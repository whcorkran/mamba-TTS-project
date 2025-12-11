# mamba_decoder.py
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from mamba_ssm.utils.generation import InferenceParams


"""Mamba-based TTS decoder module.

Architecture: Replaces ControlSpeech's Transformer decoder with MAVE's Mamba decoder

From MAVE paper (arXiv:2510.04738):
- `Mamba(d_model)` constructs a layer-like callable
- Uses mamba_ssm's InferenceParams for stateful autoregressive decoding
- `x` has shape (B, T, d_model) and `out` has same shape

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
- `MambaTTSDecoder` : stacked decoder with:
    - forward(): Full-sequence processing for training
    - create_inference_params(): Initialize state for generation
    - decode_step(): Single-token generation with InferenceParams state management
"""


class MambaTTSDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, layer_idx=None):
        super().__init__()
        self.norm_mamba = nn.LayerNorm(d_model)
        # layer_idx is required for proper state management during inference
        self.mamba = Mamba(d_model, layer_idx=layer_idx)

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
        inference_params=None,
    ):
        """
        Args:
            x: (B, T, d_model) - decoder input (embedded audio tokens)
            styled_frames: (B, T_frame, d_model) - style-conditioned frames from pipeline
            styled_mask: (B, T_frame) - mask for styled_frames (True=valid, False=pad)
            inference_params: Optional InferenceParams for autoregressive decoding
                              (state is managed internally by mamba_ssm)
        """
        # 1) Mamba: main sequence modeling
        # mamba_ssm.Mamba returns just output tensor; state is managed via inference_params
        h = self.norm_mamba(x)
        h_mamba = self.mamba(h, inference_params=inference_params)
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

        return x


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

        self.n_layers = n_layers
        self.layers = nn.ModuleList([
            MambaTTSDecoderLayer(d_model, n_heads, d_ff, layer_idx=i)
            for i in range(n_layers)
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

        # Training: no inference_params needed, process full sequence
        for layer in self.layers:
            x = layer(
                x=x,
                styled_frames=conditioning_frames,
                styled_mask=conditioning_mask,
            )

        x = self.norm_out(x)
        logits = self.head(x)
        return logits

    def create_inference_params(self, max_seqlen: int, batch_size: int):
        """Create InferenceParams for autoregressive decoding.
        
        Call this once before starting generation, then pass the same
        inference_params to each decode_step call.
        
        Args:
            max_seqlen: Maximum sequence length for generation
            batch_size: Batch size
            
        Returns:
            InferenceParams object to pass to decode_step
        """
        inference_params = InferenceParams(
            max_seqlen=max_seqlen,
            max_batch_size=batch_size,
        )
        # Allocate KV cache for each Mamba layer
        for layer in self.layers:
            layer.mamba.allocate_inference_cache(
                batch_size=batch_size,
                max_seqlen=max_seqlen,
                dtype=next(self.parameters()).dtype,
            )
        return inference_params

    def decode_step(
        self,
        last_token,
        styled_frames,
        inference_params,
        step_index: int,
        styled_mask=None,
        ref_hidden=None,
        ref_mask=None,
    ):
        """Generate logits for a single autoregressive step.

        Args:
            last_token: (B, 1) int tensor containing the most recent audio token.
            styled_frames: (B, T_frame, d_model) - style-conditioned frames from pipeline.
            inference_params: InferenceParams object from create_inference_params().
                              Manages state internally across calls.
            step_index: int, absolute position index of this token (0-based).
            styled_mask: optional (B, T_frame) boolean mask for styled_frames padding.
            ref_hidden: optional (B, T_ref, d_model) - voice prompt embeddings.
            ref_mask: optional (B, T_ref) bool - mask for ref_hidden.

        Returns:
            logits: (B, 1, vocab_size_audio)
        """
        B_local = last_token.shape[0]
        device = last_token.device

        # Update sequence length in inference_params
        inference_params.seqlen_offset = step_index

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

        for layer in self.layers:
            x = layer(
                x=x,
                styled_frames=conditioning_frames,
                styled_mask=conditioning_mask,
                inference_params=inference_params,
            )

        x = self.norm_out(x)
        logits = self.head(x)
        return logits
