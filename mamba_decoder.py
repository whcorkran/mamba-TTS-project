# mamba_decoder.py
import torch
import torch.nn as nn
from mamba_ssm import Mamba


"""Mamba-based TTS decoder module.

From MAVE paper:
- `Mamba(d_model)` constructs a layer-like callable.
- Calling signature: `out, new_state = mamba(x)` for full-sequence or
    `out, new_state = mamba(x, state)` for step-wise updates.
- `x` has shape (B, T, d_model) and `out` has same shape; `new_state` is an
    opaque per-layer state that can be stored and passed back for incremental
    decoding. The decoder below relies on those semantics.

This file includes:
- `MambaTTSDecoderLayer` : single layer (Mamba + cross-attn + FiLM + FFN)
- `MambaTTSDecoder` : stacked decoder with utilities for full-sequence and
    single-step (autoregressive) decoding. Use `decode_step` for single-token
    generation (it manages per-layer Mamba states and positional indexing).
"""


class MambaTTSDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, d_style):
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

        self.style_mlp = nn.Sequential(
            nn.Linear(d_style, 2 * d_model),
            nn.Tanh(),
        )

    def forward(
        self,
        x,
        text_hidden,
        z_style,
        text_mask=None,
        mamba_state=None,
    ):
        # 1) Mamba: main sequence modeling
        h = self.norm_mamba(x)
        if mamba_state is None:
            h_mamba, new_state = self.mamba(h)
        else:
            h_mamba, new_state = self.mamba(h, mamba_state)
        x = x + h_mamba

        # 2) Cross-attention to text
        h = self.norm_cross(x)
        key_padding_mask = None
        if text_mask is not None:
            key_padding_mask = ~text_mask

        attn_out, _ = self.cross_attn(
            query=h,
            key=text_hidden,
            value=text_hidden,
            key_padding_mask=key_padding_mask,
        )
        x = x + attn_out

        # 3) Style FiLM + FFN
        h = self.norm_ff(x)
        gamma_beta = self.style_mlp(z_style)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        h = gamma * h + beta

        ff_out = self.ff(h)
        x = x + ff_out

        return x, new_state


class MambaTTSDecoder(nn.Module):
    def __init__(
        self,
        vocab_size_audio,
        d_model=512,
        n_layers=8,
        n_heads=8,
        d_ff=2048,
        d_style=256,
        max_len=8192,  # allow flattened multi-quantizer codec sequences
        num_quantizers=1,
    ):
        super().__init__()
        self.vocab_size_audio = vocab_size_audio
        self.token_embed = nn.Embedding(vocab_size_audio, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.quant_embed = nn.Embedding(num_quantizers, d_model)

        self.layers = nn.ModuleList([
            MambaTTSDecoderLayer(d_model, n_heads, d_ff, d_style)
            for _ in range(n_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size_audio)

    def forward(self, audio_tokens, text_hidden, z_style, text_mask=None, ref_hidden=None, ref_mask=None):
        """
        audio_tokens: either
            - (B, T_audio) int codec ids (single quantizer)
            - (B, Q, T_audio) int codec ids (multi-quantizer; flattened internally)
        text_hidden: (B, T_text, d_model) text encoder outputs
        z_style: (B, d_style) style/timbre embedding
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

        # text_mask: optional (B, T_text) bool. If ref_hidden is provided we
        # will prepend ref_hidden to text_hidden and update text_mask.
        if text_mask is not None:
            assert text_mask.dim() == 2 and text_mask.shape[0] == B, (
                "text_mask must be shape (B, T_text) with dtype=bool"
            )

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

            # Prepend reference embeddings so cross-attention can attend to them
            text_hidden = torch.cat([ref_hidden, text_hidden], dim=1)
            if text_mask is None:
                text_mask = ref_mask
            else:
                text_mask = torch.cat([ref_mask, text_mask], dim=1)

        tok = self.token_embed(audio_tokens)
        qemb = self.quant_embed(quant_ids)
        pos_ids = torch.arange(T, device=device)
        pos = self.pos_embed(pos_ids)[None, :, :].expand(B, T, -1)
        x = tok + pos + qemb

        mamba_states = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            x, new_state = layer(
                x=x,
                text_hidden=text_hidden,
                z_style=z_style,
                text_mask=text_mask,
                mamba_state=mamba_states[i],
            )
            mamba_states[i] = new_state

        x = self.norm_out(x)
        logits = self.head(x)
        return logits

    def decode_step(
        self,
        last_token,
        text_hidden,
        z_style,
        mamba_states,
        step_index: int,
        text_mask=None,
        ref_hidden=None,
        ref_mask=None,
    ):
        """Generate logits for a single autoregressive step.

        Args:
            last_token: (B, 1) int tensor containing the most recent audio token.
            text_hidden: (B, T_text, d_model) encoder outputs.
            z_style: (B, d_style) style embedding.
            mamba_states: list of per-layer states (length == n_layers). Each
                entry may be None for the first step.
            step_index: int, absolute position index of this token (0-based).
            text_mask: optional (B, T_text) boolean mask for text padding.

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

        # If reference embeddings are provided, prepend them to text_hidden and
        # update text_mask so cross-attention sees reference tokens the same
        # way as in `forward`.
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

            text_hidden = torch.cat([ref_hidden, text_hidden], dim=1)
            if text_mask is None:
                text_mask = ref_mask
            else:
                text_mask = torch.cat([ref_mask, text_mask], dim=1)

        new_states = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            x, new_state = layer(
                x=x,
                text_hidden=text_hidden,
                z_style=z_style,
                text_mask=text_mask,
                mamba_state=mamba_states[i] if mamba_states is not None else None,
            )
            new_states[i] = new_state

        x = self.norm_out(x)
        logits = self.head(x)
        return logits, new_states
