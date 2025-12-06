"""
Training script that ties together:
- VccmTTSDataset for (voice prompt, style prompt, text prompt, target wav)
- Text processing/encoder + duration predictor
- SMSD style model
- FACodec encoder for codec targets
- Mamba-based decoder

Loss = w_codec * L_codec + w_dur * L_dur + w_smsd * L_smsd
"""

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio
import tempfile

from dataset import VccmTTSDataset
from text_encoder import TextProcessor, TextEncoder, DurationPredictor
from smsd import SMSD
from style_cross_attention import StyleConditioningPipeline
from mamba_decoder import MambaTTSDecoder
from data_utils.audio_encoder import FACodecEncoder


def codec_ce_loss(logits: torch.Tensor, targets: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """
    Cross-entropy over flattened codec tokens.
    logits: (B, T, vocab)
    targets: (B, T) long
    """
    B, T, V = logits.shape
    return F.cross_entropy(
        logits.view(B * T, V),
        targets.view(B * T),
        ignore_index=pad_id,
    )


def build_models(device: torch.device):
    d_model = 512
    d_style = 256
    vocab_path = Path("phoneme_vocab.json")
    text_processor = TextProcessor(vocab_path=str(vocab_path))

    text_encoder = TextEncoder(
        vocab_size=text_processor.vocab_size,
        d_model=d_model,
    ).to(device)

    dur_predictor = DurationPredictor(d_model=d_model).to(device)
    smsd = SMSD(style_dim=d_style).to(device)
    style_pipe = StyleConditioningPipeline(d_style=d_style, d_model=d_model).to(device)

    # FACodec uses 5 quantizers with codebook sizes set in the encoder; default vocab per codebook is 10.
    # We model a single stream by flattening quantizers, so we set vocab_size_audio = 10.
    decoder = MambaTTSDecoder(
        vocab_size_audio=10,
        d_model=d_model,
        d_style=d_style,
        num_quantizers=5,
    ).to(device)

    codec_encoder = FACodecEncoder()
    return text_processor, text_encoder, dur_predictor, smsd, style_pipe, decoder, codec_encoder


def prepare_text(batch_texts, text_processor: TextProcessor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert text prompts to padded phoneme IDs and masks.
    Returns phoneme_ids (B, T_text) long and mask (B, T_text) bool where True=pad.
    """
    phoneme_ids, _, mask = text_processor.batch_process(batch_texts, pad_to_max=True)
    phoneme_ids = phoneme_ids.to(device)
    mask = mask.to(device) if mask is not None else None
    return phoneme_ids, mask


def heuristic_durations(text_mask: torch.Tensor, target_frames: int) -> torch.Tensor:
    """
    Simple fallback duration targets: evenly divide total frames across phonemes.
    text_mask: (B, T_text) bool padding mask
    target_frames: int total target length (frames / codec tokens)
    """
    B, T = text_mask.shape
    lengths = (~text_mask).sum(dim=1).clamp(min=1)
    per_ph = torch.div(target_frames, lengths, rounding_mode="floor").clamp(min=1)
    durations = torch.zeros_like(text_mask, dtype=torch.float)
    for b in range(B):
        durations[b, : lengths[b]] = per_ph[b]
    return durations


def encode_waveforms_to_facodec(wavs: torch.Tensor, encoder: FACodecEncoder, sample_rate: int = 16000):
    """
    Encode a batch of waveforms (B, 1, T) to FACodec tokens by writing temp WAVs.
    Returns codec_tokens: (B, T_codec, C) and spk_embs.
    """
    wavs = wavs.detach().cpu()
    paths = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, wav in enumerate(wavs):
            path = Path(tmpdir) / f"tmp_{i}.wav"
            torchaudio.save(str(path), wav, sample_rate)
            paths.append(str(path))
        codec_tokens, spk_embs = encoder.encode(paths)
    return codec_tokens, spk_embs


def embed_codec_tokens(tokens_3d: torch.Tensor, decoder: MambaTTSDecoder):
    """
    tokens_3d: (B, Q, T) long codec ids
    Returns embedded ref_hidden (B, Q*T, d_model) and flattened mask (B, Q*T) bool where True=pad.
    """
    B, Q, T_ref = tokens_3d.shape
    flat = tokens_3d.reshape(B, Q * T_ref)
    quant_ids = torch.arange(Q, device=flat.device).repeat_interleave(T_ref).unsqueeze(0).expand(B, -1)
    pos_ids = torch.arange(T_ref, device=flat.device).repeat(Q)
    pos = decoder.pos_embed(pos_ids)[None, :, :].expand(B, -1, -1)
    tok = decoder.token_embed(flat)
    qemb = decoder.quant_embed(quant_ids)
    ref_hidden = tok + pos + qemb

    # mask: True for padding (token==0)
    mask = (tokens_3d == 0).reshape(B, Q * T_ref)
    return ref_hidden, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=10, help="short run for sanity check")
    parser.add_argument("--w_codec", type=float, default=1.0)
    parser.add_argument("--w_dur", type=float, default=0.1)
    parser.add_argument("--w_smsd", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    text_processor, text_encoder, dur_predictor, smsd, style_pipe, decoder, codec_encoder = build_models(device)

    dataset = VccmTTSDataset()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn)

    optim = torch.optim.Adam(
        list(text_encoder.parameters())
        + list(dur_predictor.parameters())
        + list(smsd.parameters())
        + list(style_pipe.parameters())
        + list(decoder.parameters()),
        lr=args.lr,
    )

    step = 0
    decoder.train()
    text_encoder.train()
    dur_predictor.train()
    smsd.train()
    style_pipe.train()

    for batch in loader:
        if step >= args.max_steps:
            break

        inputs, target_wav = batch  # target_wav: (B, 1, T_audio)
        text_prompts = inputs["text_prompt"]
        style_prompts = inputs["style_prompt"]

        # Codec targets
        with torch.no_grad():
            codec_tokens, spk_embs = encode_waveforms_to_facodec(target_wav, codec_encoder)  # (B, T, C)
        codec_tokens = codec_tokens.to(device).long()
        B, T_codec, C = codec_tokens.shape
        audio_tokens_3d = codec_tokens.permute(0, 2, 1)  # (B, Q, T)
        audio_tokens = audio_tokens_3d.reshape(B, -1)  # flatten quantizers into one sequence
        codec_pad_mask = (audio_tokens_3d == 0).reshape(B, -1)
        pad_id = 0  # FACodec pads with zeros

        # Text 
        phoneme_ids, text_mask = prepare_text(text_prompts, text_processor, device)
        text_hidden = text_encoder(phoneme_ids, mask=text_mask)  # (B, T_text, d_model)

        # Style 
        spk_embs = spk_embs.to(device) if spk_embs is not None else None
        loss_smsd = smsd(style_prompts, y_true=spk_embs) if spk_embs is not None else torch.tensor(0.0, device=device)
        with torch.no_grad():
            style_emb = smsd(style_prompts)
        style_emb = style_emb.to(device)

        # Duration 
        log_dur_pred = dur_predictor(text_hidden, mask=text_mask)  # (B, T_text)
        if text_mask is None:
            text_mask = torch.zeros_like(log_dur_pred, dtype=torch.bool)
        durations_target = heuristic_durations(text_mask, target_frames=audio_tokens.shape[1])
        loss_dur = dur_predictor.compute_loss(log_dur_pred, durations_target.to(device), mask=text_mask)
        durations_for_lr = torch.exp(log_dur_pred).detach()

        # Style conditioning + length regulation 
        styled_frames, frame_lengths, style_K, style_V = style_pipe(
            text_hidden, style_emb, durations_for_lr, text_mask=text_mask
        )
        max_frame = styled_frames.shape[1]
        ref_mask = torch.arange(max_frame, device=device)[None, :].expand(B, -1) >= frame_lengths[:, None]

        # Voice prompt as reference
        with torch.no_grad():
            voice_codec, _ = encode_waveforms_to_facodec(inputs["voice_waveform"], codec_encoder)
        voice_codec = voice_codec.to(device).long()  # (B, T_ref, C)
        voice_tokens_3d = voice_codec.permute(0, 2, 1)  # (B, Q, T_ref)
        ref_hidden, voice_mask = embed_codec_tokens(voice_tokens_3d, decoder)

        # Decoder 
        logits = decoder(
            audio_tokens,
            text_hidden=text_hidden,
            z_style=style_emb,
            text_mask=text_mask,
            ref_hidden=ref_hidden,
            ref_mask=voice_mask,
        )
        loss_codec = codec_ce_loss(logits, audio_tokens, pad_id=pad_id)

        loss_total = args.w_codec * loss_codec + args.w_dur * loss_dur + args.w_smsd * loss_smsd

        optim.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optim.step()

        print(
            f"step {step} | "
            f"loss_total={loss_total.item():.4f} "
            f"codec={loss_codec.item():.4f} dur={loss_dur.item():.4f} smsd={loss_smsd.item():.4f}"
        )
        step += 1


if __name__ == "__main__":
    main()
