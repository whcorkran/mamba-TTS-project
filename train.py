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
from typing import Dict
import yaml
import json
from datetime import datetime
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import PreprocessedTTSDataset
from text_encoder import TextProcessor, TextEncoder, DurationPredictor
from smsd import SMSD
from style_cross_attention import StyleConditioningPipeline
from mamba_decoder import MambaTTSDecoder


# ============================================================================
# ARCHITECTURE CONSTANTS (hardcoded by ControlSpeech + MAVE papers)
# ============================================================================
D_MODEL = 512  # ControlSpeech hidden dimension
D_STYLE = 512  # Style vector dimension (after FACodec 256→512 projection)
VOCAB_SIZE_AUDIO = 1024  # FACodec codebook size per quantizer
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
    print("\n" + "=" * 60)
    print("Building models...")
    print("=" * 60)
    
    print("  [1/6] Loading phoneme vocabulary...", end=" ", flush=True)
    vocab_path = Path(config.paths.phoneme_vocab)
    text_processor = TextProcessor(vocab_path=str(vocab_path))
    print(f"✓ ({text_processor.vocab_size} tokens)")

    print("  [2/6] Building TextEncoder...", end=" ", flush=True)
    text_encoder = TextEncoder(
        vocab_size=text_processor.vocab_size,
        d_model=D_MODEL,
        n_layers=config.model.text_encoder.n_layers,
        n_head=TEXT_N_HEADS,
        d_k=TEXT_D_K,
        d_v=TEXT_D_V,
        d_inner=TEXT_D_INNER,
        kernel_size=TEXT_KERNEL_SIZE,
        dropout=config.model.text_encoder.dropout,
        max_seq_len=TEXT_MAX_SEQ_LEN,
    ).to(device)
    print(f"✓ ({config.model.text_encoder.n_layers} layers)")

    print("  [3/6] Building DurationPredictor...", end=" ", flush=True)
    dur_predictor = DurationPredictor(
        d_model=D_MODEL,
        filter_size=DUR_FILTER_SIZE,
        kernel_size=DUR_KERNEL_SIZE,
        dropout=config.model.duration_predictor.dropout,
    ).to(device)
    print("✓")
    
    print("  [4/6] Building SMSD (downloading BERT if needed)...", flush=True)
    print("        This may take a minute on first run...")
    smsd = SMSD(
        bert_model=BERT_MODEL,
        bert_dim=BERT_DIM,
        style_dim=D_STYLE,
        num_mixtures=config.model.smsd.num_mixtures,
        hidden_dim=SMSD_HIDDEN_DIM,
        dropout=config.model.smsd.dropout,
        variance_mode=config.model.smsd.variance_mode,
        freeze_bert=FREEZE_BERT,
    ).to(device)
    print(f"        ✓ SMSD ready ({config.model.smsd.num_mixtures} mixtures)")
    
    print("  [5/6] Building StyleConditioningPipeline...", end=" ", flush=True)
    style_pipe = StyleConditioningPipeline(
        d_style=D_STYLE,
        d_model=D_MODEL,
        num_heads=STYLE_NUM_HEADS,
        dropout=config.model.style_conditioning.dropout,
    ).to(device)
    print("✓")

    print("  [6/6] Building MambaTTSDecoder...", end=" ", flush=True)
    decoder = MambaTTSDecoder(
        vocab_size_audio=VOCAB_SIZE_AUDIO,
        d_model=D_MODEL,
        n_layers=config.model.mamba_decoder.n_layers,
        n_heads=MAMBA_N_HEADS,
        d_ff=MAMBA_D_FF,
        max_len=config.model.mamba_decoder.max_len,
        num_quantizers=NUM_QUANTIZERS,
    ).to(device)
    print(f"✓ ({config.model.mamba_decoder.n_layers} layers)")
    
    # Apply torch.compile for faster training (PyTorch 2.0+)
    # NOTE: Disabled due to Inductor/Dynamo issues with symbolic shape tracing
    # causing AssertionError with negative values in shape calculations.
    # Uncomment below if you want to try torch.compile with fixed shapes:
    # print("\nApplying torch.compile() for optimized execution...")
    # text_encoder = torch.compile(text_encoder, dynamic=False)
    # dur_predictor = torch.compile(dur_predictor, dynamic=False)
    # style_pipe = torch.compile(style_pipe, dynamic=False)
    # decoder = torch.compile(decoder, dynamic=False)
    # print("✓ Models compiled (first batch will be slow due to compilation)")
    print("\nSkipping torch.compile() (dynamic shapes not supported)")
    
    # Count parameters
    total_params = sum(
        sum(p.numel() for p in model.parameters())
        for model in [text_encoder, dur_predictor, smsd, style_pipe, decoder]
    )
    trainable_params = sum(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
        for model in [text_encoder, dur_predictor, smsd, style_pipe, decoder]
    )
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print("=" * 60)

    return text_processor, text_encoder, dur_predictor, smsd, style_pipe, decoder


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
    
    # Dataset paths (updated for preprocessed data)
    from pathlib import Path
    processed_dir = Path(config.data.processed_dir)
    assert processed_dir.exists(), f"Processed data directory not found: {processed_dir}"
    assert (processed_dir / "metadata.json").exists(), f"Metadata not found: {processed_dir / 'metadata.json'}"
    assert (processed_dir / "tensors").exists(), f"Tensors directory not found: {processed_dir / 'tensors'}"
    
    mfa_root = Path(config.data.mfa_root)
    if config.data.require_mfa:
        assert mfa_root.exists(), f"MFA root not found: {mfa_root}"
    
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
    print(f"\n{'=' * 60}")
    print(f"Loading checkpoint from {checkpoint_path}")
    print(f"{'=' * 60}")
    
    print("  Loading checkpoint file...", end=" ", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print("✓")
    
    print("  Restoring model weights...", end=" ", flush=True)
    text_encoder.load_state_dict(checkpoint['text_encoder'])
    dur_predictor.load_state_dict(checkpoint['dur_predictor'])
    smsd.load_state_dict(checkpoint['smsd'])
    style_pipe.load_state_dict(checkpoint['style_pipe'])
    decoder.load_state_dict(checkpoint['decoder'])
    print("✓")
    
    print("  Restoring optimizer state...", end=" ", flush=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    print("✓")
    
    step = checkpoint['step']
    print(f"\n✓ Resumed from step {step}")
    if 'val_loss' in checkpoint and checkpoint['val_loss'] is not None:
        print(f"  Last validation loss: {checkpoint['val_loss']:.4f}")
    if 'timestamp' in checkpoint:
        print(f"  Checkpoint saved at: {checkpoint['timestamp']}")
    print(f"{'=' * 60}")
    
    return step


def validate(
    val_loader: DataLoader,
    text_encoder: nn.Module,
    dur_predictor: nn.Module,
    smsd: nn.Module,
    style_pipe: nn.Module,
    decoder: nn.Module,
    config: Config,
    device: torch.device,
    max_batches: int = None,
) -> Dict[str, float]:
    """
    Run validation and return metrics (using preprocessed data)
    
    Args:
        val_loader: Validation data loader
        text_encoder, dur_predictor, smsd, style_pipe, decoder: Model components
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
    
    # Determine total batches for progress bar
    total_batches = min(max_batches, len(val_loader)) if max_batches else len(val_loader)
    
    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=total_batches, desc="Validating", leave=False)
        for batch_idx, batch in pbar:
            if max_batches and batch_idx >= max_batches:
                break
                
            try:
                inputs, target_codec = batch
                
                # Move tensors to device
                phoneme_ids = inputs['phoneme_ids'].to(device)
                phoneme_mask = inputs['phoneme_mask'].to(device)
                style_embs_gt = inputs['style_emb'].to(device)
                voice_codec = inputs['voice_codec'].to(device)
                target_codec = target_codec.to(device)
                
                style_prompts = inputs['style_prompt']
                B = phoneme_ids.shape[0]
                
                # Convert target codec to audio tokens
                audio_tokens_3d = target_codec.permute(0, 2, 1).long()
                audio_tokens = audio_tokens_3d.reshape(B, -1)
                
                # Text encoding
                text_hidden = text_encoder(phoneme_ids, mask=phoneme_mask)
                
                # Style encoding and loss
                style_emb = smsd(style_prompts)
                style_emb = style_emb.to(device)
                loss_smsd = smsd(style_prompts, y_true=style_embs_gt)
                
                # Duration prediction
                log_dur_pred = dur_predictor(text_hidden, mask=phoneme_mask)
                
                durations_target = inputs['durations']
                if durations_target is not None:
                    durations_target = durations_target.to(device)
                    # Handle length mismatch
                    dur_len = durations_target.shape[1]
                    ph_len = phoneme_ids.shape[1]
                    if dur_len != ph_len:
                        if dur_len < ph_len:
                            durations_target = F.pad(durations_target, (0, ph_len - dur_len), value=0)
                        else:
                            durations_target = durations_target[:, :ph_len]
                    loss_dur = dur_predictor.compute_loss(log_dur_pred, durations_target, mask=phoneme_mask)
                else:
                    loss_dur = torch.tensor(0.0, device=device)
                
                durations_for_lr = torch.exp(log_dur_pred).detach()
                
                # Style conditioning
                styled_frames, frame_lengths, _, _ = style_pipe(
                    text_hidden, style_emb, durations_for_lr, text_mask=phoneme_mask
                )
                
                max_frame = styled_frames.shape[1]
                frame_mask = torch.arange(max_frame, device=device)[None, :].expand(B, -1) < frame_lengths[:, None]
                
                # Voice reference
                voice_tokens_3d = voice_codec.permute(0, 2, 1).long()
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
                
                # Update progress bar
                pbar.set_postfix({
                    'codec': f'{loss_codec.item():.3f}',
                    'dur': f'{loss_dur.item():.3f}',
                    'smsd': f'{loss_smsd.item():.3f}'
                })
                
            except Exception as e:
                print(f"\nWarning: Validation batch {batch_idx} failed: {e}")
                continue
        
        pbar.close()
    
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
        # Enable cuDNN benchmark for faster convolutions
        torch.backends.cudnn.benchmark = True

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

    # Build models (no FACodec encoder needed - we use preprocessed tensors)
    text_processor, text_encoder, dur_predictor, smsd, style_pipe, decoder = build_models(config, device)

    # Setup dataset using preprocessed tensors
    print("\n" + "=" * 60)
    print("Loading dataset...")
    print("=" * 60)
    full_dataset = PreprocessedTTSDataset(
        processed_dir=config.data.processed_dir,
        mfa_root=config.data.mfa_root,
        require_mfa=config.data.require_mfa,
        sample_rate=SAMPLE_RATE,
        cache_in_ram=getattr(config.data, 'cache_in_ram', False),
    )
    
    # Split into train and validation
    print("\nSplitting dataset...", end=" ", flush=True)
    if config.data.validation_split > 0:
        val_size = int(len(full_dataset) * config.data.validation_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(config.training.seed)
        )
        print(f"✓ ({train_size:,} train, {val_size:,} validation)")
    else:
        train_dataset = full_dataset
        val_dataset = None
        print(f"✓ (all {len(full_dataset):,} samples for training, no validation)")
    
    print("Creating data loaders...", end=" ", flush=True)
    loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=config.data.shuffle,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=full_dataset.collate_fn,
        prefetch_factor=getattr(config.data, 'prefetch_factor', 2) if config.data.num_workers > 0 else None,
        persistent_workers=config.data.num_workers > 0,
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
            prefetch_factor=getattr(config.data, 'prefetch_factor', 2) if config.data.num_workers > 0 else None,
            persistent_workers=config.data.num_workers > 0,
        )
    print(f"✓ (batch_size={config.training.batch_size}, num_workers={config.data.num_workers})")
    print("=" * 60)

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
    
    print(f"\n{'=' * 60}")
    print(f"Starting training from step {step} to {config.training.max_steps}")
    print(f"{'=' * 60}")
    print(f"  Best model: {checkpoint_dir / 'best_model.pt'}")
    if config.training.save_last_checkpoint:
        print(f"  Last checkpoint: {checkpoint_dir / 'last_checkpoint.pt'}")
    if config.training.early_stopping_patience > 0:
        print(f"  Early stopping: patience={config.training.early_stopping_patience}, "
              f"min_delta={config.training.early_stopping_min_delta}")
    print(f"  Hyperparameters: lr={config.training.lr}, batch_size={config.training.batch_size}, "
          f"warmup={config.training.warmup_steps}")
    print(f"  Logging every {config.training.log_interval} steps, "
          f"checkpointing every {config.training.save_interval} steps")
    print(f"{'=' * 60}\n")
    
    # Calculate approximate epochs
    steps_per_epoch = len(loader)
    total_epochs = (config.training.max_steps - step) / steps_per_epoch
    print(f"Dataset: {len(train_dataset)} samples, {steps_per_epoch} batches/epoch, ~{total_epochs:.1f} epochs total\n")
    
    # Create progress bar for training
    pbar = tqdm(
        total=config.training.max_steps - step,
        initial=0,
        desc="Training",
        unit="step",
        dynamic_ncols=True,
    )
    
    # Track metrics for progress bar
    recent_losses = {'codec': [], 'dur': [], 'smsd': [], 'total': []}
    smoothing_window = 50
    
    epoch = 0
    first_batch = True
    
    while step < config.training.max_steps and not early_stop:
        epoch += 1
        tqdm.write(f"\n--- Epoch {epoch} starting (step {step}) ---")
        
        for batch in loader:
            if step >= config.training.max_steps or early_stop:
                break
            
            # First batch may be slow due to CUDA kernel compilation
            if first_batch:
                tqdm.write("Processing first batch (CUDA compilation may take a moment)...")
                first_batch = False

            # ============================================================================
            # Batch format from PreprocessedTTSDataset:
            #   inputs: dict with phoneme_ids, phoneme_mask, text_prompt, style_prompt,
            #           style_emb (precomputed), durations, voice_codec
            #   target_codec: (B, T_target, 6) preprocessed FACodec tokens
            # ============================================================================
            inputs, target_codec = batch
            
            # Validate batch structure
            assert isinstance(inputs, dict), f"inputs must be dict, got {type(inputs)}"
            required_keys = ['phoneme_ids', 'phoneme_mask', 'style_emb', 'voice_codec', 
                            'text_prompt', 'style_prompt', 'durations']
            for key in required_keys:
                assert key in inputs, f"Missing required key '{key}' in batch inputs"
            
            # Move tensors to device
            phoneme_ids = inputs['phoneme_ids'].to(device)           # (B, T_phonemes)
            phoneme_mask = inputs['phoneme_mask'].to(device)         # (B, T_phonemes) True=pad
            style_embs_gt = inputs['style_emb'].to(device)           # (B, 512) precomputed
            voice_codec = inputs['voice_codec'].to(device)           # (B, T_ref, 6)
            target_codec = target_codec.to(device)                   # (B, T_target, 6)
            
            text_prompts = inputs['text_prompt']                     # List[str]
            style_prompts = inputs['style_prompt']                   # List[str]
            
            B = phoneme_ids.shape[0]
            assert B > 0, "Batch size is 0"
            
            # Validate tensor shapes
            assert phoneme_ids.dim() == 2, f"phoneme_ids should be (B, T), got {phoneme_ids.shape}"
            assert phoneme_mask.shape == phoneme_ids.shape, \
                f"phoneme_mask shape {phoneme_mask.shape} != phoneme_ids shape {phoneme_ids.shape}"
            assert style_embs_gt.shape == (B, 512), f"style_embs_gt should be (B, 512), got {style_embs_gt.shape}"
            assert voice_codec.dim() == 3 and voice_codec.shape[2] == 6, \
                f"voice_codec should be (B, T_ref, 6), got {voice_codec.shape}"
            assert target_codec.dim() == 3 and target_codec.shape[2] == 6, \
                f"target_codec should be (B, T_target, 6), got {target_codec.shape}"
            assert len(text_prompts) == B, f"text_prompts length {len(text_prompts)} != batch size {B}"
            assert len(style_prompts) == B, f"style_prompts length {len(style_prompts)} != batch size {B}"
            
            # Convert target codec to audio tokens for decoder
            # target_codec: (B, T, 6) -> audio_tokens: (B, 6*T)
            audio_tokens_3d = target_codec.permute(0, 2, 1).long()   # (B, 6, T)
            audio_tokens = audio_tokens_3d.reshape(B, -1)            # (B, 6*T)
            pad_id = 0

            # Use BF16 mixed precision for faster training on A100
            # BF16 has same exponent range as FP32, so no GradScaler needed
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # Text encoding using preprocessed phoneme IDs
                text_hidden = text_encoder(phoneme_ids, mask=phoneme_mask)  # (B, T_text, d_model)

                # Style Processing (SMSD module)
                # ============================================================================
                # The preprocessed style_emb is the ground truth from FACodec.
                # SMSD learns to predict this from style text descriptions.
                # ============================================================================
                
                # Training mode: compute SMSD loss (text description → ground truth style)
                loss_smsd = smsd(style_prompts, y_true=style_embs_gt)
                assert not torch.isnan(loss_smsd), "SMSD loss is NaN"
                assert not torch.isinf(loss_smsd), "SMSD loss is inf"
                
                # Inference mode: sample style embedding from SMSD's learned distribution
                # NOTE: We use SMSD's sampled output (not ground truth) to match inference behavior
                with torch.no_grad():
                    style_emb = smsd(style_prompts)
                style_emb = style_emb.to(device)

                # Duration Prediction (REQUIRES MFA ALIGNMENTS)
                # ============================================================================
                # Durations come from MFA TextGrid files, loaded by PreprocessedTTSDataset.
                # Dataset is configured with require_mfa=True, so all samples have durations.
                # ============================================================================
                log_dur_pred = dur_predictor(text_hidden, mask=phoneme_mask)  # (B, T_text)
                
                # MFA durations are REQUIRED - dataset filters samples without them
                durations_target = inputs['durations']
                if durations_target is None:
                    raise RuntimeError(
                        "MFA durations are None! This should not happen with require_mfa=True. "
                        "Check your MFA TextGrid files in VccmDataset/mfa_outputs/"
                    )
                durations_target = durations_target.to(device)
                
                # Handle potential length mismatch between durations and phoneme_ids
                # (MFA may have different phoneme count than our tokenizer)
                dur_len = durations_target.shape[1]
                ph_len = phoneme_ids.shape[1]
                if dur_len != ph_len:
                    # Log warning for significant mismatches (> 20% difference)
                    mismatch_ratio = abs(dur_len - ph_len) / max(dur_len, ph_len)
                    if mismatch_ratio > 0.2:
                        print(f"WARNING: Large duration/phoneme length mismatch: "
                              f"dur={dur_len}, ph={ph_len} ({mismatch_ratio:.1%} diff). "
                              f"This may indicate tokenizer/MFA alignment issues.")
                    
                    # Pad or truncate durations to match phoneme length
                    if dur_len < ph_len:
                        durations_target = F.pad(durations_target, (0, ph_len - dur_len), value=0)
                    else:
                        durations_target = durations_target[:, :ph_len]
                
                assert durations_target.shape == (B, ph_len), \
                    f"Duration shape mismatch after adjustment: {durations_target.shape} vs expected ({B}, {ph_len})"
                
                loss_dur = dur_predictor.compute_loss(log_dur_pred, durations_target, mask=phoneme_mask)
                assert not torch.isnan(loss_dur), "Duration loss is NaN"
                assert not torch.isinf(loss_dur), "Duration loss is inf"
                
                durations_for_lr = torch.exp(log_dur_pred).detach()
                # Ensure minimum duration of 1 frame per phoneme to avoid zero-length outputs
                durations_for_lr = durations_for_lr.clamp(min=1.0)

                # Style conditioning + length regulation 
                styled_frames, frame_lengths, style_K, style_V = style_pipe(
                    text_hidden, style_emb, durations_for_lr, text_mask=phoneme_mask
                )
                # styled_frames: (B, T_frame, d_model) - frame-level, style-conditioned features
                
                # Validate style pipeline outputs
                assert styled_frames.dim() == 3 and styled_frames.shape[0] == B, \
                    f"styled_frames should be (B, T_frame, d_model), got {styled_frames.shape}"
                assert styled_frames.shape[2] == D_MODEL, \
                    f"styled_frames d_model mismatch: {styled_frames.shape[2]} != {D_MODEL}"
                assert frame_lengths.shape == (B,), f"frame_lengths should be (B,), got {frame_lengths.shape}"
                assert (frame_lengths > 0).all(), "All frame_lengths must be positive"
                
                # Create frame-level mask based on actual lengths
                max_frame = styled_frames.shape[1]
                frame_mask = torch.arange(max_frame, device=device)[None, :].expand(B, -1) < frame_lengths[:, None]

                # Voice prompt as reference for timbre cloning
                # voice_codec is already preprocessed: (B, T_ref, 6)
                assert voice_codec.shape[0] == B, f"voice_codec batch size mismatch: {voice_codec.shape[0]} != {B}"
                voice_tokens_3d = voice_codec.permute(0, 2, 1).long()  # (B, 6, T_ref)
                assert voice_tokens_3d.shape[1] == NUM_QUANTIZERS, \
                    f"voice_codec should have {NUM_QUANTIZERS} quantizers, got {voice_tokens_3d.shape[1]}"
                ref_hidden, voice_mask = embed_codec_tokens(voice_tokens_3d, decoder)

                # Decoder forward pass
                logits = decoder(
                    audio_tokens=audio_tokens,
                    styled_frames=styled_frames,
                    styled_mask=frame_mask,
                    ref_hidden=ref_hidden,
                    ref_mask=voice_mask,
                )
                
                # Validate decoder output
                expected_logits_shape = (B, audio_tokens.shape[1], VOCAB_SIZE_AUDIO)
                assert logits.shape == expected_logits_shape, \
                    f"Decoder logits shape mismatch: {logits.shape} != expected {expected_logits_shape}"
                assert not torch.isnan(logits).any(), "Decoder produced NaN logits"
                assert not torch.isinf(logits).any(), "Decoder produced Inf logits"
                
                loss_codec = codec_ce_loss(logits, audio_tokens, pad_id=pad_id)
                assert not torch.isnan(loss_codec), "Codec loss is NaN"
                assert not torch.isinf(loss_codec), "Codec loss is inf"

                # Compute total loss with configured weights
                loss_total = (
                    config.training.w_codec * loss_codec +
                    config.training.w_dur * loss_dur +
                    config.training.w_smsd * loss_smsd
                )
            
            # Validate total loss before backward (outside autocast)
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
            
            # Track losses for smoothed progress bar display
            recent_losses['codec'].append(loss_codec.item())
            recent_losses['dur'].append(loss_dur.item())
            recent_losses['smsd'].append(loss_smsd.item())
            recent_losses['total'].append(loss_total.item())
            
            # Keep only recent losses for smoothing
            for key in recent_losses:
                if len(recent_losses[key]) > smoothing_window:
                    recent_losses[key] = recent_losses[key][-smoothing_window:]
            
            # Update progress bar
            current_lr = optim.param_groups[0]['lr']
            avg_loss = sum(recent_losses['total']) / len(recent_losses['total'])
            pbar.set_postfix({
                'loss': f'{avg_loss:.3f}',
                'lr': f'{current_lr:.1e}',
                'epoch': epoch,
            })
            pbar.update(1)
            
            # Detailed logging at intervals
            if step % config.training.log_interval == 0:
                tqdm.write(
                    f"[Step {step:6d}] lr={current_lr:.2e} | "
                    f"loss={loss_total.item():.4f} "
                    f"(codec={loss_codec.item():.4f}, "
                    f"dur={loss_dur.item():.4f}, "
                    f"smsd={loss_smsd.item():.4f})"
                )
            
            # Validation and checkpointing
            if step % config.training.save_interval == 0 and step > 0:
                # Run validation if available
                if val_loader is not None:
                    tqdm.write(f"\n{'=' * 40}")
                    tqdm.write(f"Running validation at step {step}...")
                    val_metrics = validate(
                        val_loader,
                        text_encoder,
                        dur_predictor,
                        smsd,
                        style_pipe,
                        decoder,
                        config,
                        device,
                        max_batches=50,  # Limit validation time
                    )
                    
                    val_loss = val_metrics['val_loss_total']
                    tqdm.write(f"Validation: loss={val_loss:.4f} "
                               f"(codec={val_metrics['val_loss_codec']:.4f}, "
                               f"dur={val_metrics['val_loss_dur']:.4f}, "
                               f"smsd={val_metrics['val_loss_smsd']:.4f})")
                    
                    # Check if this is the best model
                    improvement = best_val_loss - val_loss
                    if improvement > config.training.early_stopping_min_delta:
                        tqdm.write(f"✓ New best model! (previous: {best_val_loss:.4f}, improvement: {improvement:.4f})")
                        best_val_loss = val_loss
                        steps_without_improvement = 0
                        
                        # Save best model
                        tqdm.write(f"  Saving best model...")
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
                        tqdm.write(f"  ✓ Saved to {checkpoint_dir / 'best_model.pt'}")
                    else:
                        steps_without_improvement += 1
                        tqdm.write(f"  No improvement ({steps_without_improvement}/{config.training.early_stopping_patience})")
                        
                        # Early stopping check
                        if config.training.early_stopping_patience > 0 and \
                           steps_without_improvement >= config.training.early_stopping_patience:
                            tqdm.write(f"\n⚠ Early stopping triggered! No improvement for {steps_without_improvement} validations.")
                            early_stop = True
                    
                    tqdm.write(f"{'=' * 40}\n")
                
                # Save last checkpoint (for recovery)
                if config.training.save_last_checkpoint:
                    tqdm.write(f"  Saving checkpoint...")
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
                    tqdm.write(f"  ✓ Saved to {checkpoint_dir / 'last_checkpoint.pt'}")
            
            step += 1
    
    # Close progress bar
    pbar.close()
    
    # Training finished
    print("\n")
    if early_stop:
        print(f"✓ Training stopped early at step {step} (early stopping triggered)")
    else:
        print(f"✓ Training completed after {step} steps!")
    
    print(f"\n{'='*60}")
    print(f"Training Summary:")
    print(f"  Final step: {step}")
    print(f"  Epochs completed: {epoch}")
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
