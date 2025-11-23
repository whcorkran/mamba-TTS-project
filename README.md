# mamba-TTS-project

## Project Setup
```bash
bash setup.sh
```

## Mamba-based Decoder (mamba_decoder.py)

What it does
- Autoregressive audio-token decoder built from stacked Mamba SSM blocks.
- After each Mamba block the layer performs cross-attention to conditioning features (text and optional reference tokens), then applies optional FiLM (feature-wise linear modulation) and an FFN.

Simple inputs
- audio_tokens: LongTensor (B, T_audio) — teacher-forcing token ids for forward.
- last_token: LongTensor (B, 1) — single-step input for `decode_step`.
- text_hidden: FloatTensor (B, T_text, d_model) — text encoder outputs (keys/values for cross-attn).
- z_style: FloatTensor (B, d_style) — global style/timbre vector used by FiLM (optional if we only use ref tokens).
- text_mask: BoolTensor (B, T_text) — padding mask for `text_hidden` (True = padding).
- ref_hidden (optional): FloatTensor (B, T_ref, d_model) — reference token embeddings prepended to `text_hidden`.
- ref_mask (optional): BoolTensor (B, T_ref) — mask for `ref_hidden`.

Outputs
- forward(...) -> logits: FloatTensor (B, T_audio, vocab_size_audio).
- decode_step(...) -> (logits: FloatTensor (B, 1, vocab_size_audio), new_states: list) where `new_states` are per-layer Mamba states for incremental decoding.

Notes
- Passing `ref_hidden` implements MAVE-style reference-token conditioning (cross-attn sees `[ref || text]`).
- Mamba runs on GPU with the package mamba-ssm


