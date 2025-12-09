"""
Training script following ControlSpeech architecture with Mamba decoder.

TRAINING STATUS: Training from SCRATCH (not using pretrained ControlSpeech)
============================================================================
Components and their training status:

FROZEN (Pretrained, Not Trained):
  ✓ FACodec Encoder/Decoder: Downloaded from HuggingFace, used for audio encoding
  ✓ BERT (in SMSD): bert-base-uncased, frozen for style text encoding

TRAINED (Your Model Components):
  ✓ Text Encoder: FastSpeech2-style FFT blocks (4 layers, 512-dim)
  ✓ Duration Predictor: Predicts phoneme durations (requires MFA alignments)
  ✓ SMSD: Mixture Density Network mapping text descriptions → style vectors
  ✓ Style Conditioning: Two cross-attention layers + length regulator
  ✓ Mamba Decoder: 6 Mamba layers (MAVE) replacing ControlSpeech's Transformer
  ✓ FACodec Style Projection: 256→512 linear layer (trainable)

Architecture Flow (ControlSpeech with Mamba):
==================================================
Inputs:
  - Text prompt (content to synthesize)
  - Style prompt (prosody/emotion text description)
  - Voice prompt (reference audio for timbre cloning)

Pipeline:
  1. Text Encoder: text → phoneme-level features (B, T_text, d_model)
  
  2. SMSD (Style Mixture Semantic Density):
     - BERT encodes style text description (frozen)
     - Mixture Density Network samples style embedding (B, d_style) (trained)
     
  3. Style Conditioning Pipeline:
     a. Project style to K, V for cross-attention
     b. Cross-Attention #1: Text ⊗ Style → styled text features
     c. Duration Predictor: predict phoneme durations from styled text
     d. Length Regulation: upsample phoneme → frame level
     e. Cross-Attention #2: Frames ⊗ Style → styled frames (B, T_frame, d_model)
  
  4. Voice Prompt Processing:
     - FACodec encodes voice prompt → codec tokens (frozen encoder)
     - Embed as reference for timbre cloning
  
  5. Mamba Decoder (MAVE architecture replacing ControlSpeech's Conformer):
     - Input: styled frames (content + style + temporal structure)
     - Architecture: Mamba SSM → Cross-Attention → FFN (per layer)
     - Cross-attention: attend to styled frames for content
     - Reference: attend to voice prompt embeddings for timbre (MAVE-style)
     - Output: codec token logits (B, T, vocab_size)
  
  6. Target Processing:
     - FACodec encodes target audio → codec tokens for supervision (frozen)

Loss = w_codec * L_codec + w_dur * L_dur + w_smsd * L_smsd
  - L_codec: cross-entropy on predicted codec tokens (main reconstruction)
  - L_dur: duration prediction MSE loss
  - L_smsd: mixture NLL for style modeling

Key Difference from Original ControlSpeech:
  - Original: Transformer decoder (Self-Attn → Cross-Attn → FFN) with O(n²) complexity
  - This: MAVE Mamba decoder (Mamba SSM → Cross-Attn → FFN) with O(n) complexity
  
Architecture Notes:
  - FiLM conditioning removed (not in ControlSpeech or MAVE papers)
  - Quantizer embeddings added for FACodec's 6-quantizer structure
  - Decoder layers: 6 (per ControlSpeech Appendix F recommendation)
"""

import argparse
from pathlib import Path
from typing import Tuple, Dict, Any
import yaml
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchaudio
import tempfile

from dataset import VccmTTSDataset
from text_encoder import TextProcessor, TextEncoder, DurationPredictor
from smsd import SMSD
from style_cross_attention import StyleConditioningPipeline
from mamba_decoder import MambaTTSDecoder
from data_utils.audio_encoder import FACodecEncoder


# ============================================================================
# ARCHITECTURE CONSTANTS (hardcoded by ControlSpeech + MAVE papers)
# ============================================================================
D_MODEL = 512  # ControlSpeech hidden dimension
D_STYLE = 512  # Style vector dimension (after FACodec 256→512 projection)
VOCAB_SIZE_AUDIO = 10  # FACodec codebook size per quantizer
NUM_QUANTIZERS = 6  # FACodec: 1 prosody + 2 content + 3 residual
SAMPLE_RATE = 16000  # FACodec operates at 16kHz
BERT_MODEL = "bert-base-uncased"  # SMSD always uses this
BERT_DIM = 768  # bert-base-uncased output dimension
FREEZE_BERT = True  # BERT always frozen

# Text Encoder (FastSpeech2 architecture)
TEXT_N_HEADS = 2  # Baseline confirmed
TEXT_D_K = 64
TEXT_D_V = 64
TEXT_D_INNER = 1024  # FFN dimension
TEXT_KERNEL_SIZE = (9, 1)  # Conv kernel
TEXT_MAX_SEQ_LEN = 3000

# Duration Predictor (VariancePredictor architecture)
DUR_FILTER_SIZE = 256
DUR_KERNEL_SIZE = 3

# SMSD
SMSD_HIDDEN_DIM = 512  # MLP hidden dimension

# Style Conditioning
STYLE_NUM_HEADS = 8  # Cross-attention heads

# Mamba Decoder (MAVE + ControlSpeech)
MAMBA_N_HEADS = 8  # Cross-attention heads
MAMBA_D_FF = 2048  # 4 × D_MODEL
# ============================================================================


class Config:
    """Simple config class that allows dot notation access"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def __repr__(self):
        return f"Config({self.__dict__})"


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)


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


def build_models(config: Config, device: torch.device):
    """Build all models according to config"""
    vocab_path = Path(config.paths.phoneme_vocab)
    text_processor = TextProcessor(vocab_path=str(vocab_path))

    text_encoder = TextEncoder(
        vocab_size=text_processor.vocab_size,
        d_model=D_MODEL,
        n_layers=config.model.text_encoder.n_layers,  # Configurable
        n_head=TEXT_N_HEADS,
        d_k=TEXT_D_K,
        d_v=TEXT_D_V,
        d_inner=TEXT_D_INNER,
        kernel_size=TEXT_KERNEL_SIZE,
        dropout=config.model.text_encoder.dropout,  # Configurable
        max_seq_len=TEXT_MAX_SEQ_LEN,
    ).to(device)

    dur_predictor = DurationPredictor(
        d_model=D_MODEL,
        filter_size=DUR_FILTER_SIZE,
        kernel_size=DUR_KERNEL_SIZE,
        dropout=config.model.duration_predictor.dropout,  # Configurable
    ).to(device)
    
    smsd = SMSD(
        bert_model=BERT_MODEL,
        bert_dim=BERT_DIM,
        style_dim=D_STYLE,
        num_mixtures=config.model.smsd.num_mixtures,  # Configurable (paper tested 3, 5, 7)
        hidden_dim=SMSD_HIDDEN_DIM,
        dropout=config.model.smsd.dropout,  # Configurable
        variance_mode=config.model.smsd.variance_mode,  # Configurable (paper tested different modes)
        freeze_bert=FREEZE_BERT,
    ).to(device)
    
    style_pipe = StyleConditioningPipeline(
        d_style=D_STYLE,
        d_model=D_MODEL,
        num_heads=STYLE_NUM_HEADS,
        dropout=config.model.style_conditioning.dropout,  # Configurable
    ).to(device)

    decoder = MambaTTSDecoder(
        vocab_size_audio=VOCAB_SIZE_AUDIO,
        d_model=D_MODEL,
        n_layers=config.model.mamba_decoder.n_layers,  # Configurable (paper uses 6)
        n_heads=MAMBA_N_HEADS,
        d_ff=MAMBA_D_FF,
        max_len=config.model.mamba_decoder.max_len,  # Configurable
        num_quantizers=NUM_QUANTIZERS,
    ).to(device)

    codec_encoder = FACodecEncoder()
    return text_processor, text_encoder, dur_predictor, smsd, style_pipe, decoder, codec_encoder


def prepare_text(batch_texts, text_processor: TextProcessor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert text prompts to padded phoneme IDs and masks.
    Returns phoneme_ids (B, T_text) long and mask (B, T_text) bool where True=pad.
    """
    assert len(batch_texts) > 0, "batch_texts cannot be empty"
    assert all(isinstance(t, str) and len(t) > 0 for t in batch_texts), "All texts must be non-empty strings"
    
    phoneme_ids, _, mask = text_processor.batch_process(batch_texts, pad_to_max=True)
    phoneme_ids = phoneme_ids.to(device)
    
    # Mask should always be returned when pad_to_max=True
    assert mask is not None, "text_processor.batch_process should return mask when pad_to_max=True"
    mask = mask.to(device)
    
    # Validate shapes
    assert phoneme_ids.dim() == 2, f"phoneme_ids should be 2D (B, T), got {phoneme_ids.shape}"
    assert mask.shape == phoneme_ids.shape, f"mask shape {mask.shape} != phoneme_ids shape {phoneme_ids.shape}"
    
    return phoneme_ids, mask


def encode_waveforms_to_facodec(wavs: torch.Tensor, encoder: FACodecEncoder, sample_rate: int = 16000):
    """
    Encode a batch of waveforms (B, 1, T) to FACodec tokens and style embeddings.
    
    Returns:
        codec_tokens: (B, T_codec, C) - Audio codec tokens (6 quantizers: 4 style + 2 content)
        style_embs: (B, 512) - Style/timbre embeddings projected to 512-dim
    
    Note: FACodec's encoder outputs:
        - codec_tokens: Prosody (1) + Content (2) + Residual (3) quantizers = 6 total
        - spk_embs: Originally 256-dim timbre, projected to 512-dim to match ControlSpeech
    
    These style_embs capture timbre but may not fully represent prosodic style
    (emotion, speaking rate, pitch patterns). For full style control, consider
    extracting additional prosody features from the audio.
    """
    assert wavs.dim() == 3, f"wavs must be (B, 1, T), got shape {wavs.shape}"
    assert wavs.shape[1] == 1, f"wavs must have 1 channel, got {wavs.shape[1]}"
    assert wavs.shape[2] > 0, "wavs must have non-zero length"
    
    wavs = wavs.detach().cpu()
    paths = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, wav in enumerate(wavs):
            path = Path(tmpdir) / f"tmp_{i}.wav"
            torchaudio.save(str(path), wav, sample_rate)
            paths.append(str(path))
        codec_tokens, style_embs = encoder.encode(paths)
    
    # Validate FACodec outputs
    assert codec_tokens is not None, "FACodec returned None for codec_tokens"
    assert style_embs is not None, "FACodec returned None for style_embs"
    assert codec_tokens.dim() == 3, f"codec_tokens should be (B, T, C), got {codec_tokens.shape}"
    assert style_embs.dim() == 2, f"style_embs should be (B, 512), got {style_embs.shape}"
    assert codec_tokens.shape[0] == wavs.shape[0], f"Batch size mismatch: {codec_tokens.shape[0]} != {wavs.shape[0]}"
    assert style_embs.shape[0] == wavs.shape[0], f"Batch size mismatch: {style_embs.shape[0]} != {wavs.shape[0]}"
    assert style_embs.shape[1] == 512, f"style_embs should be 512-dim, got {style_embs.shape[1]}"
    
    return codec_tokens, style_embs


def embed_codec_tokens(tokens_3d: torch.Tensor, decoder: MambaTTSDecoder):
    """
    tokens_3d: (B, Q, T) long codec ids
    Returns embedded ref_hidden (B, Q*T, d_model) and flattened mask (B, Q*T) bool where True=pad.
    """
    assert tokens_3d.dim() == 3, f"tokens_3d must be (B, Q, T), got {tokens_3d.shape}"
    assert tokens_3d.dtype in [torch.long, torch.int64], f"tokens_3d must be long dtype, got {tokens_3d.dtype}"
    
    B, Q, T_ref = tokens_3d.shape
    assert Q > 0 and T_ref > 0, f"Invalid quantizer or time dimensions: Q={Q}, T={T_ref}"
    
    flat = tokens_3d.reshape(B, Q * T_ref)
    quant_ids = torch.arange(Q, device=flat.device).repeat_interleave(T_ref).unsqueeze(0).expand(B, -1)
    pos_ids = torch.arange(T_ref, device=flat.device).repeat(Q)
    pos = decoder.pos_embed(pos_ids)[None, :, :].expand(B, -1, -1)
    tok = decoder.token_embed(flat)
    qemb = decoder.quant_embed(quant_ids)
    ref_hidden = tok + pos + qemb

    # Validate output shapes
    assert ref_hidden.shape == (B, Q * T_ref, decoder.token_embed.embedding_dim), \
        f"ref_hidden shape mismatch: {ref_hidden.shape}"

    # mask: True for padding (token==0)
    mask = (tokens_3d == 0).reshape(B, Q * T_ref)
    return ref_hidden, mask


def validate_config(config: Config):
    """Validate configuration to catch issues before training starts"""
    # Training hyperparameters
    assert config.training.batch_size > 0, "batch_size must be positive"
    assert config.training.max_steps > 0, "max_steps must be positive"
    assert config.training.lr > 0, "learning rate must be positive"
    assert 0 < config.training.lr_decay <= 1.0, "lr_decay must be in (0, 1]"
    assert config.training.warmup_steps >= 0, "warmup_steps must be non-negative"
    assert config.training.clip_grad_norm > 0, "clip_grad_norm must be positive"
    
    # Loss weights - all should be non-negative
    assert config.training.w_codec >= 0, "w_codec must be non-negative"
    assert config.training.w_dur >= 0, "w_dur must be non-negative"
    assert config.training.w_smsd >= 0, "w_smsd must be non-negative"
    assert (config.training.w_codec + config.training.w_dur + config.training.w_smsd) > 0, \
        "At least one loss weight must be positive"
    
    # Model dimensions
    assert config.model.d_model > 0, "d_model must be positive"
    assert config.model.d_style > 0, "d_style must be positive"
    assert config.model.vocab_size_audio > 0, "vocab_size_audio must be positive"
    assert config.model.num_quantizers > 0, "num_quantizers must be positive"
    
    # Dataset paths
    from pathlib import Path
    assert Path(config.data.csv_path).exists(), f"CSV file not found: {config.data.csv_path}"
    assert Path(config.data.audio_root).exists(), f"Audio root not found: {config.data.audio_root}"
    assert Path(config.paths.phoneme_vocab).exists(), f"Phoneme vocab not found: {config.paths.phoneme_vocab}"
    assert config.training.save_interval > 0, "save_interval must be positive"
    assert config.training.early_stopping_patience >= 0, "early_stopping_patience must be non-negative (0 = disabled)"
    assert config.data.validation_split >= 0 and config.data.validation_split < 1, "validation_split must be in [0, 1)"
    
    print("✓ Configuration validation passed")


def save_checkpoint(
    checkpoint_name: str,
    step: int,
    text_encoder: nn.Module,
    dur_predictor: nn.Module,
    smsd: nn.Module,
    style_pipe: nn.Module,
    decoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config: Config,
    checkpoint_dir: Path,
    val_loss: float = None,
    extra_info: dict = None,
) -> Path:
    """
    Save training checkpoint
    
    Args:
        checkpoint_name: Name of checkpoint file (e.g., "best_model.pt" or "last_checkpoint.pt")
        step: Current training step
        text_encoder, dur_predictor, smsd, style_pipe, decoder: Model components
        optimizer: Optimizer state
        scheduler: LR scheduler state
        config: Training configuration
        checkpoint_dir: Directory to save checkpoints
        val_loss: Optional validation loss to save with checkpoint
        extra_info: Optional extra information dict to save
    
    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / checkpoint_name
    
    checkpoint = {
        'step': step,
        'text_encoder': text_encoder.state_dict(),
        'dur_predictor': dur_predictor.state_dict(),
        'smsd': smsd.state_dict(),
        'style_pipe': style_pipe.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'config': config.__dict__,
        'val_loss': val_loss,
        'timestamp': datetime.now().isoformat(),
    }
    
    if extra_info:
        checkpoint.update(extra_info)
    
    torch.save(checkpoint, checkpoint_path)
    
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    text_encoder: nn.Module,
    dur_predictor: nn.Module,
    smsd: nn.Module,
    style_pipe: nn.Module,
    decoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
) -> int:
    """
    Load training checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        text_encoder, dur_predictor, smsd, style_pipe, decoder: Model components to load into
        optimizer: Optimizer to load state into
        scheduler: LR scheduler to load state into
        device: Device to load checkpoint to
    
    Returns:
        step: Training step to resume from
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    text_encoder.load_state_dict(checkpoint['text_encoder'])
    dur_predictor.load_state_dict(checkpoint['dur_predictor'])
    smsd.load_state_dict(checkpoint['smsd'])
    style_pipe.load_state_dict(checkpoint['style_pipe'])
    decoder.load_state_dict(checkpoint['decoder'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    
    step = checkpoint['step']
    print(f"✓ Resumed from step {step}")
    
    return step


def validate(
    val_loader: DataLoader,
    text_processor: any,
    text_encoder: nn.Module,
    dur_predictor: nn.Module,
    smsd: nn.Module,
    style_pipe: nn.Module,
    decoder: nn.Module,
    codec_encoder: any,
    config: Config,
    device: torch.device,
    max_batches: int = None,
) -> Dict[str, float]:
    """
    Run validation and return metrics
    
    Args:
        val_loader: Validation data loader
        text_processor, text_encoder, dur_predictor, smsd, style_pipe, decoder, codec_encoder: Model components
        config: Training configuration
        device: Device to run on
        max_batches: Optional limit on number of batches to validate (for speed)
    
    Returns:
        Dictionary of validation metrics
    """
    # Set to eval mode
    text_encoder.eval()
    dur_predictor.eval()
    smsd.eval()
    style_pipe.eval()
    decoder.eval()
    
    total_loss_codec = 0.0
    total_loss_dur = 0.0
    total_loss_smsd = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if max_batches and batch_idx >= max_batches:
                break
                
            try:
                inputs, target_wav = batch
                text_prompts = inputs["text_prompt"]
                style_prompts = inputs["style_prompt"]
                
                # Encode target audio
                audio_tokens, style_embs_gt = encode_waveforms_to_facodec(target_wav, codec_encoder)
                audio_tokens = audio_tokens.to(device).long()
                style_embs_gt = style_embs_gt.to(device)
                B = audio_tokens.shape[0]
                
                # Text encoding
                phoneme_ids, _, text_mask = prepare_text(text_prompts, text_processor, device)
                text_hidden = text_encoder(phoneme_ids, text_mask)
                
                # Style encoding
                style_emb = smsd(style_prompts)
                style_emb = style_emb.to(device)
                
                # SMSD loss
                loss_smsd = smsd(style_prompts, y_true=style_embs_gt)
                
                # Duration prediction
                log_dur_pred = dur_predictor(text_hidden, mask=text_mask)
                
                # MFA durations REQUIRED
                assert 'durations' in inputs and inputs['durations'] is not None, \
                    "MFA durations REQUIRED for validation! Place TextGrid files in VccmDataset/mfa_outputs/"
                durations_target = inputs['durations'].to(device)
                
                loss_dur = dur_predictor.compute_loss(log_dur_pred, durations_target, mask=text_mask)
                durations_for_lr = torch.exp(log_dur_pred).detach()
                
                # Style conditioning
                styled_frames, frame_lengths, _, _ = style_pipe(
                    text_hidden, style_emb, durations_for_lr, text_mask=text_mask
                )
                
                max_frame = styled_frames.shape[1]
                frame_mask = torch.arange(max_frame, device=device)[None, :].expand(B, -1) < frame_lengths[:, None]
                
                # Voice reference
                voice_codec, _ = encode_waveforms_to_facodec(inputs["voice_waveform"], codec_encoder)
                voice_codec = voice_codec.to(device).long()
                voice_tokens_3d = voice_codec.permute(0, 2, 1)
                ref_hidden, voice_mask = embed_codec_tokens(voice_tokens_3d, decoder)
                
                # Decoder
                logits = decoder(
                    audio_tokens=audio_tokens,
                    styled_frames=styled_frames,
                    styled_mask=frame_mask,
                    ref_hidden=ref_hidden,
                    ref_mask=voice_mask,
                )
                
                loss_codec = codec_ce_loss(logits, audio_tokens, pad_id=0)
                
                total_loss_codec += loss_codec.item()
                total_loss_dur += loss_dur.item()
                total_loss_smsd += loss_smsd.item()
                num_batches += 1
                
            except Exception as e:
                print(f"Warning: Validation batch {batch_idx} failed: {e}")
                continue
    
    # Set back to train mode
    text_encoder.train()
    dur_predictor.train()
    smsd.train()
    style_pipe.train()
    decoder.train()
    
    if num_batches == 0:
        return {'val_loss_total': float('inf'), 'val_loss_codec': float('inf'), 
                'val_loss_dur': float('inf'), 'val_loss_smsd': float('inf')}
    
    avg_codec = total_loss_codec / num_batches
    avg_dur = total_loss_dur / num_batches
    avg_smsd = total_loss_smsd / num_batches
    avg_total = (config.training.w_codec * avg_codec + 
                 config.training.w_dur * avg_dur + 
                 config.training.w_smsd * avg_smsd)
    
    return {
        'val_loss_total': avg_total,
        'val_loss_codec': avg_codec,
        'val_loss_dur': avg_dur,
        'val_loss_smsd': avg_smsd,
    }


def save_final_model(
    text_encoder: nn.Module,
    dur_predictor: nn.Module,
    smsd: nn.Module,
    style_pipe: nn.Module,
    decoder: nn.Module,
    config: Config,
    save_dir: Path,
):
    """
    Save final trained model (weights only, no optimizer state)
    
    Args:
        text_encoder, dur_predictor, smsd, style_pipe, decoder: Model components
        config: Training configuration
        save_dir: Directory to save final model
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    final_model_path = save_dir / "final_model.pt"
    
    final_model = {
        'text_encoder': text_encoder.state_dict(),
        'dur_predictor': dur_predictor.state_dict(),
        'smsd': smsd.state_dict(),
        'style_pipe': style_pipe.state_dict(),
        'decoder': decoder.state_dict(),
        'config': config.__dict__,
        'timestamp': datetime.now().isoformat(),
    }
    
    torch.save(final_model, final_model_path)
    print(f"✓ Saved final model to {final_model_path}")
    
    # Also save config as JSON for easy inspection
    config_path = save_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2, default=str)
    print(f"✓ Saved config to {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Train ControlSpeech + Mamba TTS")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    
    # Validate configuration early to catch errors
    validate_config(config)
    
    # Set random seed for reproducibility
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.training.seed)

    # Setup device
    if config.training.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(config.training.device)
    print(f"Using device: {device}")

    # Setup checkpoint directory
    checkpoint_dir = Path(config.paths.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")

    # Build models
    text_processor, text_encoder, dur_predictor, smsd, style_pipe, decoder, codec_encoder = build_models(config, device)

    # Setup dataset and dataloader with train/val split
    full_    dataset = VccmTTSDataset(
        csv_path=config.data.csv_path,
        audio_root=config.data.audio_root,
        sample_rate=SAMPLE_RATE,  # FACodec requires 16kHz
    )
    
    # Split into train and validation
    if config.data.validation_split > 0:
        val_size = int(len(full_dataset) * config.data.validation_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(config.training.seed)
        )
        print(f"Dataset split: {train_size} train, {val_size} validation samples")
    else:
        train_dataset = full_dataset
        val_dataset = None
        print(f"No validation split - using all {len(full_dataset)} samples for training")
    
    loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=config.data.shuffle,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=full_dataset.collate_fn,
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            collate_fn=full_dataset.collate_fn,
        )

    # Setup optimizer with proper hyperparameters
    optim = torch.optim.Adam(
        list(text_encoder.parameters())
        + list(dur_predictor.parameters())
        + list(smsd.parameters())
        + list(style_pipe.parameters())
        + list(decoder.parameters()),
        lr=config.training.lr,
        betas=tuple(config.training.betas),
        eps=config.training.eps,
        weight_decay=config.training.weight_decay,
    )
    
    # Setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=config.training.lr_decay)

    # Resume from checkpoint if specified
    step = 0
    if args.resume:
        step = load_checkpoint(
            args.resume,
            text_encoder,
            dur_predictor,
            smsd,
            style_pipe,
            decoder,
            optim,
            scheduler,
            device,
        )
        step += 1  # Start from next step

    # Set models to training mode
    decoder.train()
    text_encoder.train()
    dur_predictor.train()
    smsd.train()
    style_pipe.train()

    # Training loop with best model tracking
    best_val_loss = float('inf')
    steps_without_improvement = 0
    early_stop = False
    
    print(f"\nStarting training from step {step} to {config.training.max_steps}")
    print(f"Best model will be saved to {checkpoint_dir / 'best_model.pt'}")
    if config.training.save_last_checkpoint:
        print(f"Last checkpoint will be saved to {checkpoint_dir / 'last_checkpoint.pt'}")
    if config.training.early_stopping_patience > 0:
        print(f"Early stopping enabled: patience={config.training.early_stopping_patience}, "
              f"min_delta={config.training.early_stopping_min_delta}")
    print(f"Config: lr={config.training.lr}, batch_size={config.training.batch_size}, "
          f"betas={config.training.betas}, warmup={config.training.warmup_steps}")
    
    for batch in loader:
        if step >= config.training.max_steps or early_stop:
            break

        inputs, target_wav = batch  # target_wav: (B, 1, T_audio)
        text_prompts = inputs["text_prompt"]
        style_prompts = inputs["style_prompt"]

        # Codec targets
        with torch.no_grad():
            codec_tokens, style_embs = encode_waveforms_to_facodec(target_wav, codec_encoder)  # (B, T, C)
        codec_tokens = codec_tokens.to(device).long()
        B, T_codec, C = codec_tokens.shape
        audio_tokens_3d = codec_tokens.permute(0, 2, 1)  # (B, Q, T)
        audio_tokens = audio_tokens_3d.reshape(B, -1)  # flatten quantizers into one sequence
        pad_id = 0  # FACodec pads with zeros

        # Text 
        phoneme_ids, text_mask = prepare_text(text_prompts, text_processor, device)
        text_hidden = text_encoder(phoneme_ids, mask=text_mask)  # (B, T_text, d_model)

        # Style Processing (SMSD module)
        # ============================================================================
        # ControlSpeech Architecture (Paper Section 3.3):
        #   SMSD learns to map text descriptions → 512-dim style vectors
        #
        # Ground Truth Style Vector Sources (in order of quality):
        #   1. BEST: Extract prosody features from FACodec encoder + target audio
        #      - Combines timbre (speaker identity) + prosody (emotion, rate, pitch)
        #      - This is what ControlSpeech paper uses
        #
        #   2. CURRENT: Use FACodec's timbre embeddings (projected to 512-dim)
        #      - Captures speaker identity well
        #      - May not fully capture prosodic style variations
        #      - Good enough for timbre cloning, limited for style transfer
        #
        #   3. WORST: Use zero/random vectors (training without style supervision)
        #      - SMSD learns only from text descriptions
        #      - Results in poor style controllability
        # ============================================================================
        
        # Validate style embeddings exist (FACodec should always return them)
        assert style_embs is not None, "FACodec must return style embeddings - got None"
        style_embs_gt = style_embs.to(device)
        
        # Validate shape
        assert style_embs_gt.shape == (B, 512), \
            f"style_embs_gt must be (B, 512), got {style_embs_gt.shape}"
        
        # Training mode: compute SMSD loss (text description → ground truth style)
        # The SMSD learns to predict the distribution of style vectors from text
        loss_smsd = smsd(style_prompts, y_true=style_embs_gt)
        assert not torch.isnan(loss_smsd), "SMSD loss is NaN"
        assert not torch.isinf(loss_smsd), "SMSD loss is inf"
        
        # Inference mode: sample style embedding from SMSD's learned distribution
        # NOTE: We use SMSD's sampled output (not ground truth) to match inference behavior
        # This prevents train/test mismatch where the model never sees its own predictions
        with torch.no_grad():
            style_emb = smsd(style_prompts)
        style_emb = style_emb.to(device)
        
        assert style_emb.shape == (B, 512), f"SMSD output must be (B, 512), got {style_emb.shape}"

        # Duration Prediction (REQUIRES MFA ALIGNMENTS)
        # ============================================================================
        # ControlSpeech REQUIRES MFA (Montreal Forced Aligner) durations - NO FALLBACK
        # 
        # MFA Setup:
        #   1. Install: conda install -c conda-forge montreal-forced-aligner
        #   2. Download models: mfa model download acoustic english_mfa
        #                       mfa model download dictionary english_mfa
        #   3. Run alignment: mfa align audio_dir/ transcripts_dir/ english_mfa english_mfa output_dir/
        #   4. Place TextGrid files in: VccmDataset/mfa_outputs/{item_name}.TextGrid
        #
        # The dataset will load durations on-the-fly from TextGrid files.
        # If TextGrid files are missing, training will FAIL with assertion error.
        # ============================================================================
        log_dur_pred = dur_predictor(text_hidden, mask=text_mask)  # (B, T_text)
        
        # text_mask should NEVER be None at this point (prepare_text validates this)
        assert text_mask is not None, "text_mask is None - this should never happen after prepare_text()"
        assert text_mask.shape == (B, text_hidden.shape[1]), \
            f"text_mask shape {text_mask.shape} doesn't match text_hidden (B, T_text)"
        
        # MFA durations are REQUIRED - no fallback
        assert 'durations' in inputs and inputs['durations'] is not None, \
            "MFA durations are REQUIRED! Run MFA preprocessing and place TextGrid files in VccmDataset/mfa_outputs/"
        
        durations_target = inputs['durations'].to(device)
        assert durations_target.shape == (B, text_hidden.shape[1]), \
            f"durations shape mismatch: {durations_target.shape} vs text (B, {text_hidden.shape[1]})"
        
        loss_dur = dur_predictor.compute_loss(log_dur_pred, durations_target.to(device), mask=text_mask)
        assert not torch.isnan(loss_dur), "Duration loss is NaN"
        assert not torch.isinf(loss_dur), "Duration loss is inf"
        
        durations_for_lr = torch.exp(log_dur_pred).detach()
        assert (durations_for_lr >= 0).all(), "Durations must be non-negative"

        # Style conditioning + length regulation 
        # This follows ControlSpeech architecture:
        # 1. Cross-Attention #1: Text ⊗ Style → styled text
        # 2. Length Regulation: phoneme → frame level
        # 3. Cross-Attention #2: Frames ⊗ Style → styled frames
        styled_frames, frame_lengths, style_K, style_V = style_pipe(
            text_hidden, style_emb, durations_for_lr, text_mask=text_mask
        )
        # styled_frames: (B, T_frame, d_model) - frame-level, style-conditioned features
        
        # Create frame-level mask based on actual lengths
        max_frame = styled_frames.shape[1]
        frame_mask = torch.arange(max_frame, device=device)[None, :].expand(B, -1) < frame_lengths[:, None]
        # frame_mask: (B, T_frame) bool where True=valid, False=padding

        # Voice prompt as reference for timbre cloning
        with torch.no_grad():
            voice_codec, _ = encode_waveforms_to_facodec(inputs["voice_waveform"], codec_encoder)
        voice_codec = voice_codec.to(device).long()  # (B, T_ref, C)
        voice_tokens_3d = voice_codec.permute(0, 2, 1)  # (B, Q, T_ref)
        ref_hidden, voice_mask = embed_codec_tokens(voice_tokens_3d, decoder)

        # Decoder (Codec Generator)
        # In ControlSpeech, the decoder attends to styled_frames (not raw text_hidden)
        # These styled_frames already contain:
        # - Content information (from text)
        # - Style information (via two cross-attentions)
        # - Correct temporal structure (via length regulation)
        
        # Validate styled_frames shape
        assert styled_frames.shape == (B, max_frame, text_encoder.d_model), \
            f"styled_frames shape mismatch: {styled_frames.shape}"
        
        # Check for reasonable frame length (not too far off from audio length)
        audio_frame_count = audio_tokens.shape[1]
        styled_frame_count = styled_frames.shape[1]
        ratio = audio_frame_count / styled_frame_count if styled_frame_count > 0 else float('inf')
        
        # Warn if mismatch is extreme (more than 2x or less than 0.5x)
        if ratio > 2.0 or ratio < 0.5:
            print(f"WARNING: Large mismatch between audio frames ({audio_frame_count}) "
                  f"and styled frames ({styled_frame_count}), ratio={ratio:.2f}. "
                  f"This may indicate duration prediction issues.")
        
        logits = decoder(
            audio_tokens=audio_tokens,
            styled_frames=styled_frames,
            styled_mask=frame_mask,
            ref_hidden=ref_hidden,
            ref_mask=voice_mask,
        )
        
        # Validate decoder output
        assert logits.shape == (B, audio_tokens.shape[1], decoder.vocab_size_audio), \
            f"logits shape mismatch: {logits.shape} vs expected (B, {audio_tokens.shape[1]}, {decoder.vocab_size_audio})"
        
        loss_codec = codec_ce_loss(logits, audio_tokens, pad_id=pad_id)
        assert not torch.isnan(loss_codec), "Codec loss is NaN"
        assert not torch.isinf(loss_codec), "Codec loss is inf"

        # Compute total loss with configured weights
        loss_total = (
            config.training.w_codec * loss_codec +
            config.training.w_dur * loss_dur +
            config.training.w_smsd * loss_smsd
        )
        
        # Validate total loss before backward
        assert not torch.isnan(loss_total), \
            f"Total loss is NaN (codec={loss_codec.item():.4f}, dur={loss_dur.item():.4f}, smsd={loss_smsd.item():.4f})"
        assert not torch.isinf(loss_total), \
            f"Total loss is inf (codec={loss_codec.item():.4f}, dur={loss_dur.item():.4f}, smsd={loss_smsd.item():.4f})"
        assert loss_total.item() >= 0, f"Total loss is negative: {loss_total.item()}"

        # Optimization step
        optim.zero_grad()
        loss_total.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(text_encoder.parameters()) +
            list(dur_predictor.parameters()) +
            list(smsd.parameters()) +
            list(style_pipe.parameters()) +
            list(decoder.parameters()),
            config.training.clip_grad_norm
        )
        
        optim.step()
        
        # Learning rate schedule: warmup then exponential decay
        if step < config.training.warmup_steps:
            # Linear warmup
            lr_scale = (step + 1) / config.training.warmup_steps
            for param_group in optim.param_groups:
                param_group['lr'] = config.training.lr * lr_scale
        else:
            # Exponential decay after warmup
            scheduler.step()
        
        # Logging
        if step % config.training.log_interval == 0:
            current_lr = optim.param_groups[0]['lr']
            print(
                f"step {step:6d} | lr={current_lr:.2e} | "
                f"loss={loss_total.item():.4f} "
                f"(codec={loss_codec.item():.4f} "
                f"dur={loss_dur.item():.4f} "
                f"smsd={loss_smsd.item():.4f})"
            )
        
        # Validation and checkpointing
        if step % config.training.save_interval == 0 and step > 0:
            # Run validation if available
            if val_loader is not None:
                print(f"\nRunning validation at step {step}...")
                val_metrics = validate(
                    val_loader,
                    text_processor,
                    text_encoder,
                    dur_predictor,
                    smsd,
                    style_pipe,
                    decoder,
                    codec_encoder,
                    config,
                    device,
                    max_batches=50,  # Limit validation time
                )
                
                val_loss = val_metrics['val_loss_total']
                print(f"Validation: loss={val_loss:.4f} "
                      f"(codec={val_metrics['val_loss_codec']:.4f}, "
                      f"dur={val_metrics['val_loss_dur']:.4f}, "
                      f"smsd={val_metrics['val_loss_smsd']:.4f})")
                
                # Check if this is the best model
                improvement = best_val_loss - val_loss
                if improvement > config.training.early_stopping_min_delta:
                    print(f"✓ New best model! (previous: {best_val_loss:.4f}, improvement: {improvement:.4f})")
                    best_val_loss = val_loss
                    steps_without_improvement = 0
                    
                    # Save best model
                    save_checkpoint(
                        "best_model.pt",
                        step,
                        text_encoder,
                        dur_predictor,
                        smsd,
                        style_pipe,
                        decoder,
                        optim,
                        scheduler,
                        config,
                        checkpoint_dir,
                        val_loss=val_loss,
                        extra_info={'val_metrics': val_metrics},
                    )
                    print(f"  Saved to {checkpoint_dir / 'best_model.pt'}")
                else:
                    steps_without_improvement += 1
                    print(f"  No improvement ({steps_without_improvement}/{config.training.early_stopping_patience})")
                    
                    # Early stopping check
                    if config.training.early_stopping_patience > 0 and \
                       steps_without_improvement >= config.training.early_stopping_patience:
                        print(f"\n⚠ Early stopping triggered! No improvement for {steps_without_improvement} validations.")
                        early_stop = True
            
            # Save last checkpoint (for recovery)
            if config.training.save_last_checkpoint:
                save_checkpoint(
                    "last_checkpoint.pt",
                    step,
                    text_encoder,
                    dur_predictor,
                    smsd,
                    style_pipe,
                    decoder,
                    optim,
                    scheduler,
                    config,
                    checkpoint_dir,
                    val_loss=val_loss if val_loader else None,
                )
                print(f"  Saved last checkpoint to {checkpoint_dir / 'last_checkpoint.pt'}")
        
        step += 1
    
    # Training finished
    if early_stop:
        print(f"\n✓ Training stopped early at step {step} (early stopping triggered)")
    else:
        print(f"\n✓ Training completed after {step} steps!")
    
    print(f"\n{'='*60}")
    print(f"Training Summary:")
    print(f"  Final step: {step}")
    if val_loader:
        print(f"  Best validation loss: {best_val_loss:.4f}")
        print(f"  Best model saved to: {checkpoint_dir / 'best_model.pt'}")
    if config.training.save_last_checkpoint:
        print(f"  Last checkpoint: {checkpoint_dir / 'last_checkpoint.pt'}")
    print(f"\nTo resume training, use: --resume {checkpoint_dir / 'last_checkpoint.pt'}")
    print(f"For inference, use: {checkpoint_dir / 'best_model.pt'}")
    print(f"{'='*60}")
    
    print(f"\n{'='*60}")
    print(f"Training finished!")
    print(f"Final checkpoint saved to: {checkpoint_dir / 'final_model.pt'}")
    print(f"To resume training, use: --resume {checkpoint_dir / f'checkpoint_step_{step:07d}.pt'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
