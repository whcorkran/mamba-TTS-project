"""
Fix style embeddings: Replace BERT embeddings (768-dim) with FACodec speaker embeddings (512-dim).

This script only updates the _style.pt files without redoing phoneme or codec processing.
Per ControlSpeech paper, ground truth style vectors come from FACodec speaker embeddings.

Usage:
    python -m data_utils.fix_style_embeddings \
        --processed_dir processed_data/ \
        --audio_dir VccmDataset/audio_extracted \
        --batch_size 32
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from tqdm import tqdm

from .audio_encoder import FACodecEncoder


class StyleEmbeddingFixer:
    """Extract FACodec speaker embeddings and update _style.pt files."""
    
    def __init__(
        self,
        processed_dir: str,
        audio_dir: str,
        device: str = 'cuda',
        batch_size: int = 32,
        num_load_workers: int = 8,
    ):
        print(f"\n{'='*60}")
        print("Style Embedding Fixer")
        print("Replacing BERT (768-dim) with FACodec speaker (512-dim)")
        print(f"{'='*60}")
        
        self.processed_dir = Path(processed_dir)
        self.audio_dir = Path(audio_dir)
        self.device = device
        self.batch_size = batch_size
        self.num_load_workers = num_load_workers
        
        # Load metadata
        print(f"\n[1/2] Loading metadata...")
        metadata_path = self.processed_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"  ✓ Loaded {len(self.metadata):,} samples from {metadata_path}")
        
        # Initialize FACodec encoder
        print(f"\n[2/2] Initializing FACodec encoder...")
        print(f"  Loading FACodec models to {device}...")
        self.encoder = FACodecEncoder()
        self.encoder.fa_encoder = self.encoder.fa_encoder.to(device)
        self.encoder.fa_decoder = self.encoder.fa_decoder.to(device)
        print(f"  ✓ FACodec encoder ready on {device}")
        
        # Audio loading utilities
        import io
        import soundfile as sf
        import torchaudio
        self._io = io
        self._sf = sf
        self._torchaudio = torchaudio
        
        print(f"\nConfiguration:")
        print(f"  Processed dir: {self.processed_dir}")
        print(f"  Audio dir: {self.audio_dir}")
        print(f"  Batch size: {batch_size}")
        print(f"  Load workers: {num_load_workers}")
        print(f"{'='*60}\n")
    
    def _safe_name(self, item_name: str) -> str:
        """Convert item_name to safe filename."""
        return item_name.replace('/', '_').replace(' ', '_')
    
    def _item_name_to_audio_path(self, item_name: str) -> Path:
        """Convert item_name to audio file path."""
        # Handle different naming conventions
        # e.g., "ESD-Dataset-0020-Happy-0020_000928" -> "ESD/Dataset/0020/Happy/0020_000928.wav"
        parts = item_name.split('-')
        if len(parts) >= 2:
            # Reconstruct path with / separators
            path_str = '/'.join(parts) + '.wav'
        else:
            path_str = item_name + '.wav'
        return self.audio_dir / path_str
    
    def _load_audio(self, args: Tuple[int, str]) -> Tuple[int, Optional[torch.Tensor]]:
        """Load a single audio file."""
        idx, item_name = args
        try:
            audio_path = self._item_name_to_audio_path(item_name)
            if not audio_path.exists():
                return idx, None
            
            audio_data, sr = self._sf.read(str(audio_path))
            audio = torch.from_numpy(audio_data).float()
            
            if audio.ndim == 2:  # stereo -> mono
                audio = audio.mean(dim=1)
            
            if sr != 16000:
                audio = self._torchaudio.functional.resample(
                    audio.unsqueeze(0), sr, 16000
                ).squeeze(0)
            
            return idx, audio
        except Exception as e:
            return idx, None
    
    def _extract_style_batch(self, audio_batch: torch.Tensor) -> torch.Tensor:
        """Extract FACodec speaker embeddings from audio batch."""
        with torch.no_grad():
            enc = self.encoder.fa_encoder(audio_batch)
            prosody = self.encoder.fa_encoder.get_prosody_feature(audio_batch)
            
            # Align lengths
            if prosody.shape[2] != enc.shape[2]:
                if prosody.shape[2] > enc.shape[2]:
                    prosody = prosody[:, :, :enc.shape[2]]
                else:
                    pad_len = enc.shape[2] - prosody.shape[2]
                    prosody = torch.nn.functional.pad(prosody, (0, pad_len))
            
            _, _, _, _, spk_embs = self.encoder.fa_decoder(
                enc, prosody, eval_vq=False, vq=True
            )
            
            # Project 256 -> 512 (as per ControlSpeech paper)
            spk_embs_projected = self.encoder.style_projection(spk_embs)
            
            return spk_embs_projected
    
    def fix_all(self) -> Tuple[int, int]:
        """
        Fix all style embeddings.
        
        Returns:
            (num_fixed, num_skipped)
        """
        tensors_dir = self.processed_dir / "tensors"
        
        # All existing files need fixing (they have 768-dim BERT embeddings)
        print("Collecting files to process...")
        items_to_fix = []
        for i, meta in enumerate(self.metadata):
            item_name = meta['item_name']
            safe_name = self._safe_name(item_name)
            style_path = tensors_dir / f"{safe_name}_style.pt"
            items_to_fix.append((i, item_name, style_path))
        
        num_batches = (len(items_to_fix) + self.batch_size - 1) // self.batch_size
        
        print(f"\n{'='*60}")
        print(f"Processing {len(items_to_fix):,} files in {num_batches:,} batches")
        print(f"  BERT embeddings (768-dim) → FACodec speaker (512-dim)")
        print(f"  Batch size: {self.batch_size}")
        print(f"{'='*60}\n")
        
        num_fixed = 0
        num_skipped = 0
        
        # Progress bar with detailed stats
        pbar = tqdm(
            total=len(items_to_fix),
            desc="Extracting FACodec style embeddings",
            unit="file",
            dynamic_ncols=True,
        )
        
        with ThreadPoolExecutor(max_workers=self.num_load_workers) as executor:
            for batch_start in range(0, len(items_to_fix), self.batch_size):
                batch_items = items_to_fix[batch_start:batch_start + self.batch_size]
                
                # Load audio in parallel
                load_args = [(j, item[1]) for j, item in enumerate(batch_items)]
                futures = [executor.submit(self._load_audio, args) for args in load_args]
                
                audio_tensors = []
                valid_batch_items = []
                
                for future, item in zip(futures, batch_items):
                    local_idx, audio = future.result()
                    if audio is not None:
                        audio_tensors.append(audio)
                        valid_batch_items.append(item)
                    else:
                        num_skipped += 1
                        pbar.update(1)
                
                if not audio_tensors:
                    continue
                
                # Pad and batch
                max_len = max(t.shape[0] for t in audio_tensors)
                padded = [
                    torch.nn.functional.pad(t, (0, max_len - t.shape[0]))
                    for t in audio_tensors
                ]
                audio_batch = torch.stack(padded, dim=0).unsqueeze(1).to(self.device)
                
                # Extract style embeddings
                style_embs = self._extract_style_batch(audio_batch)
                
                # Update progress bar stats
                pbar.set_postfix({
                    'fixed': num_fixed,
                    'skipped': num_skipped,
                    'batch': f'{len(audio_tensors)}/{len(batch_items)}'
                })
                
                # Save each
                for j, (_, item_name, style_path) in enumerate(valid_batch_items):
                    style_emb = style_embs[j].cpu()
                    torch.save(style_emb, style_path)
                    num_fixed += 1
                    pbar.update(1)
        
        pbar.close()
        return num_fixed, num_skipped


def main():
    parser = argparse.ArgumentParser(
        description="Fix style embeddings: BERT (768) -> FACodec (512)"
    )
    parser.add_argument("--processed_dir", type=str, default="processed_data",
                        help="Path to preprocessed data directory")
    parser.add_argument("--audio_dir", type=str, default="VccmDataset/audio_extracted",
                        help="Path to extracted audio directory")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for GPU processing")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of workers for audio loading")
    args = parser.parse_args()
    
    fixer = StyleEmbeddingFixer(
        processed_dir=args.processed_dir,
        audio_dir=args.audio_dir,
        batch_size=args.batch_size,
        num_load_workers=args.num_workers,
    )
    
    num_fixed, num_skipped = fixer.fix_all()
    
    print(f"\n{'='*60}")
    print("Style embedding fix complete!")
    print(f"  Fixed: {num_fixed}")
    print(f"  Skipped (audio not found): {num_skipped}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

