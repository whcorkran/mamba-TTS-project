# mamba_decoder.py
import torch
import torch.nn as nn
from mamba_ssm import Mamba


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

        # style conditioning
        self.style_mlp = nn.Sequential(
            nn.Linear(d_style, 2 * d_model),
            nn.Tanh(),
        )

    def forward(
        self,
        x,              # (B, T_audio, d_model)
        text_hidden,    # (B, T_text, d_model)
        z_style,        # (B, d_style)
        text_mask=None,
        mamba_state=None,
    ):
        # 1) Mamba: main sequence modeling
        h = self.norm_mamba(x)
        if mamba_state is None:
            h_mamba, new_state = self.mamba(h)         # full sequence
        else:
            h_mamba, new_state = self.mamba(h, mamba_state)  # step-wise
        x = x + h_mamba

        # 2) Cross-attention to text
        h = self.norm_cross(x)
        key_padding_mask = None
        if text_mask is not None:
            key_padding_mask = ~text_mask  # True at padding positions

        attn_out, _ = self.cross_attn(
            query=h,
            key=text_hidden,
            value=text_hidden,
            key_padding_mask=key_padding_mask,
        )
        x = x + attn_out

        # 3) Style FiLM + FFN
        h = self.norm_ff(x)
        gamma_beta = self.style_mlp(z_style)  # (B, 2*d_model)
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
        max_len=4096,
    ):
        super().__init__()
        self.vocab_size_audio = vocab_size_audio
        self.token_embed = nn.Embedding(vocab_size_audio, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

        self.layers = nn.ModuleList([
            MambaTTSDecoderLayer(d_model, n_heads, d_ff, d_style)
            for _ in range(n_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size_audio)

    def forward(self, audio_tokens, text_hidden, z_style, text_mask=None):
        """
        audio_tokens: (B, T_audio) int codec ids (teacher forcing input)
        text_hidden: (B, T_text, d_model) text encoder outputs
        z_style: (B, d_style) style/timbre embedding
        """
        B, T = audio_tokens.shape
        device = audio_tokens.device

        tok = self.token_embed(audio_tokens)                      # (B, T, d_model)
        pos_ids = torch.arange(T, device=device)
        pos = self.pos_embed(pos_ids)[None, :, :].expand(B, T, -1)
        x = tok + pos

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
        logits = self.head(x)  # (B, T, vocab_size_audio)
        return logits
