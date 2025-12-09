# mamba_decoder.py
import torch
import torch.nn as nn
from mamba_ssm import Mamba


"""Mamba-based TTS decoder module.

Architecture: Replaces ControlSpeech's Transformer decoder with MAVE's Mamba decoder

From MAVE paper (arXiv:2510.04738):
- `Mamba(d_model)` constructs a layer-like callable.
- Calling signature: `out, new_state = mamba(x)` for full-sequence or
    `out, new_state = mamba(x, state)` for step-wise updates.
- `x` has shape (B, T, d_model) and `out` has same shape; `new_state` is an
    opaque per-layer state that can be stored and passed back for incremental
    decoding. The decoder below relies on those semantics.

Layer Structure (per MAVE):
- Mamba SSM: Efficient sequence modeling (O(n) vs Transformer's O(n²))
- Cross-Attention: Attend to styled_frames (content+style+timing from ControlSpeech pipeline)
- FFN: Feed-forward transformation

Inputs (per ControlSpeech + MAVE):
- audio_tokens: Codec tokens for teacher forcing
- styled_frames: Output from StyleConditioningPipeline (already contains style via cross-attentions)
- ref_hidden: Voice prompt embeddings (timbre from reference speaker)

This file includes:
- `MambaTTSDecoderLayer` : single layer (Mamba + cross-attn + FFN)
- `MambaTTSDecoder` : stacked decoder with utilities for full-sequence and
    single-step (autoregressive) decoding. Use `decode_step` for single-token
    generation (it manages per-layer Mamba states and positional indexing).
"""


class MambaTTSDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.norm_mamba = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model)

        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )

        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(
        self,
        x,
        styled_frames,
        styled_mask=None,
        mamba_state=None,
    ):
        """
        Args:
            x: (B, T, d_model) - decoder input (embedded audio tokens)
            styled_frames: (B, T_frame, d_model) - style-conditioned frames from pipeline
            styled_mask: (B, T_frame) - mask for styled_frames (True=valid, False=pad)
            mamba_state: Optional Mamba state for autoregressive decoding
        """
        # 1) Mamba: main sequence modeling
        h = self.norm_mamba(x)
        if mamba_state is None:
            h_mamba, new_state = self.mamba(h)
        else:
            h_mamba, new_state = self.mamba(h, mamba_state)
        x = x + h_mamba

        # 2) Cross-attention to styled frames (contains content+style+timing)
        h = self.norm_cross(x)
        key_padding_mask = None
        if styled_mask is not None:
            # Convert True=valid to False=valid for PyTorch's convention
            key_padding_mask = ~styled_mask

        attn_out, _ = self.cross_attn(
            query=h,
            key=styled_frames,
            value=styled_frames,
            key_padding_mask=key_padding_mask,
        )
        x = x + attn_out

        # 3) FFN
        h = self.norm_ff(x)
        ff_out = self.ff(h)
        x = x + ff_out

        return x, new_state


class MambaTTSDecoder(nn.Module):
    def __init__(
        self,
        vocab_size_audio,
        d_model=512,
        n_layers=6,  # Per ControlSpeech Appendix F
        n_heads=8,
        d_ff=2048,
        max_len=8192,  # allow flattened multi-quantizer codec sequences
        num_quantizers=6,  # FACodec uses 6 quantizers (1 prosody + 2 content + 3 residual)
    ):
        super().__init__()
        self.vocab_size_audio = vocab_size_audio
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size_audio, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        # Quantizer embeddings: Custom addition for multi-quantizer FACodec (not in papers)
        # Helps distinguish between prosody/content/residual quantizers
        self.quant_embed = nn.Embedding(num_quantizers, d_model)

        self.layers = nn.ModuleList([
            MambaTTSDecoderLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size_audio)

    def forward(self, audio_tokens, styled_frames, styled_mask=None, ref_hidden=None, ref_mask=None):
        """
        Forward pass for teacher-forced training.
        
        Args:
            audio_tokens: (B, Q*T) or (B, Q, T) - codec tokens for teacher forcing
            styled_frames: (B, T_frame, d_model) - style-conditioned frames from pipeline
                Contains: content (from text) + style (from SMSD cross-attentions) + timing (from length regulation)
            styled_mask: (B, T_frame) bool - True=valid, False=padding
            ref_hidden: (B, T_ref, d_model) - voice prompt embeddings for timbre cloning
            ref_mask: (B, T_ref) bool - True=valid, False=padding
        
        Returns:
            logits: (B, T_audio, vocab_size_audio) - predicted codec token logits
        """
        if audio_tokens.dim() == 3:
            # audio_tokens: (B, Q, T)
            B, Q, T = audio_tokens.shape
            audio_tokens = audio_tokens.reshape(B, Q * T)
            quant_ids = torch.arange(Q, device=audio_tokens.device).repeat_interleave(T)
            quant_ids = quant_ids.unsqueeze(0).expand(B, -1)
        elif audio_tokens.dim() == 2:
            B, T = audio_tokens.shape
            quant_ids = torch.zeros_like(audio_tokens)
        else:
            raise ValueError("audio_tokens must be (B, T) or (B, Q, T)")
        device = audio_tokens.device

        # Validate styled_frames
        assert styled_frames.dim() == 3 and styled_frames.shape[0] == B, (
            f"styled_frames must be (B, T_frame, d_model), got {styled_frames.shape}"
        )
        
        # Validate mask if provided
        if styled_mask is not None:
            assert styled_mask.dim() == 2 and styled_mask.shape[0] == B, (
                "styled_mask must be shape (B, T_frame) with dtype=bool"
            )

        # Prepend reference embeddings for timbre conditioning (MAVE-style)
        conditioning_frames = styled_frames
        conditioning_mask = styled_mask
        
        if ref_hidden is not None:
            # ref_hidden: (B, T_ref, d_model)
            assert ref_hidden.dim() == 3 and ref_hidden.shape[0] == B, (
                "ref_hidden must be (B, T_ref, d_model)"
            )
            if ref_mask is None:
                ref_mask = torch.ones(B, ref_hidden.shape[1], dtype=torch.bool, device=device)
            else:
                assert ref_mask.dim() == 2 and ref_mask.shape[0] == B, (
                    "ref_mask must be (B, T_ref) bool"
                )

            # Concatenate: [ref_hidden || styled_frames] so decoder can attend to both
            conditioning_frames = torch.cat([ref_hidden, styled_frames], dim=1)
            if conditioning_mask is None:
                conditioning_mask = ref_mask
            else:
                conditioning_mask = torch.cat([ref_mask, conditioning_mask], dim=1)

        tok = self.token_embed(audio_tokens)
        qemb = self.quant_embed(quant_ids)
        pos_ids = torch.arange(T, device=device)
        pos = self.pos_embed(pos_ids)[None, :, :].expand(B, T, -1)
        x = tok + pos + qemb

        mamba_states = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            x, new_state = layer(
                x=x,
                styled_frames=conditioning_frames,
                styled_mask=conditioning_mask,
                mamba_state=mamba_states[i],
            )
            mamba_states[i] = new_state

        x = self.norm_out(x)
        logits = self.head(x)
        return logits

    def decode_step(
        self,
        last_token,
        styled_frames,
        mamba_states,
        step_index: int,
        styled_mask=None,
        ref_hidden=None,
        ref_mask=None,
    ):
        """Generate logits for a single autoregressive step.

        Args:
            last_token: (B, 1) int tensor containing the most recent audio token.
            styled_frames: (B, T_frame, d_model) - style-conditioned frames from pipeline.
            mamba_states: list of per-layer states (length == n_layers). Each
                entry may be None for the first step.
            step_index: int, absolute position index of this token (0-based).
            styled_mask: optional (B, T_frame) boolean mask for styled_frames padding.
            ref_hidden: optional (B, T_ref, d_model) - voice prompt embeddings.
            ref_mask: optional (B, T_ref) bool - mask for ref_hidden.

        Returns:
            logits: (B, 1, vocab_size_audio)
            new_states: list of updated per-layer mamba states
        """
        B_local = last_token.shape[0]
        device = last_token.device

        # embed token and position
        tok = self.token_embed(last_token)
        pos_id = torch.tensor([step_index], device=device)
        pos = self.pos_embed(pos_id)[None, :, :].expand(B_local, 1, -1)
        x = tok + pos

        # Prepend reference embeddings to styled_frames for timbre conditioning
        conditioning_frames = styled_frames
        conditioning_mask = styled_mask
        
        if ref_hidden is not None:
            assert ref_hidden.dim() == 3 and ref_hidden.shape[0] == B_local, (
                "ref_hidden must be (B, T_ref, d_model)"
            )
            if ref_mask is None:
                ref_mask = torch.ones(B_local, ref_hidden.shape[1], dtype=torch.bool, device=device)
            else:
                assert ref_mask.dim() == 2 and ref_mask.shape[0] == B_local, (
                    "ref_mask must be (B, T_ref) bool"
                )

            conditioning_frames = torch.cat([ref_hidden, styled_frames], dim=1)
            if conditioning_mask is None:
                conditioning_mask = ref_mask
            else:
                conditioning_mask = torch.cat([ref_mask, conditioning_mask], dim=1)

        new_states = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            x, new_state = layer(
                x=x,
                styled_frames=conditioning_frames,
                styled_mask=conditioning_mask,
                mamba_state=mamba_states[i] if mamba_states is not None else None,
            )
            new_states[i] = new_state

        x = self.norm_out(x)
        logits = self.head(x)
        return logits, new_states
