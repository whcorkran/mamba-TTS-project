"""
Dataset preprocessing pipeline.
Preprocesses CSV dataset: audio encoding, text-to-phoneme, style embeddings.
Saves processed data for PyTorch DataLoader.
"""

#    python -m data_utils.preprocess --csv_path VccmDataset/controlspeech_train.csv \
#        --output_dir processed_data/ \
#        --tarball VccmDataset/TextrolSpeech_data.tar.gz \
#        --phoneme_vocab_path . \
#        --debug

import csv
import json
import tarfile
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple

import torch
import numpy as np
from tqdm import tqdm

from .text_processor import TxtProcessor, BertModel, StyleProcessor
from .audio_encoder import FACodecEncoder


class DatasetPreprocessor:
    """Preprocesses dataset: audio, phonemes, style embeddings."""
    
    def __init__(
        self,
        output_dir: str,
        tarball_paths: Union[str, List[str]],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        sample_rate: int = 16000,
        debug: bool = False,
        phoneme_vocab_path: str = ".",
    ):
        """
        Initialize preprocessor.
        
        Args:
            output_dir: Directory to save preprocessed data
            tarball_paths: Path(s) to tarball(s) containing audio files.
                           Can be a single path, a list of paths, or comma-separated paths.
            device: Device for model inference
            sample_rate: Target sample rate for audio
            debug: If True, only process first 128 samples
            phoneme_vocab_path: Path to phoneme_vocab.json file or directory containing it
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.debug = debug
        
        # Load phoneme vocabulary
        vocab_path = Path(phoneme_vocab_path)
        if vocab_path.is_dir():
            vocab_path = vocab_path / "phoneme_vocab.json"
        if not vocab_path.exists():
            raise FileNotFoundError(f"Phoneme vocabulary not found at {vocab_path}")
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.phoneme_vocab = json.load(f)
        # Build phoneme-to-index mapping for tokenization
        self.phoneme_to_idx = {ph: idx for idx, ph in enumerate(self.phoneme_vocab)}
        print(f"Loaded phoneme vocabulary ({len(self.phoneme_vocab)} tokens) from {vocab_path}")
        
        # Normalize tarball_paths to a list
        if isinstance(tarball_paths, str):
            # Support comma-separated paths
            tarball_paths = [p.strip() for p in tarball_paths.split(",") if p.strip()]
        
        # Open tarballs and build combined audio index
        # Maps audio filename -> (tarfile object, TarInfo)
        self.tarballs: List[tarfile.TarFile] = []
        self.audio_index: Dict[str, Tuple[tarfile.TarFile, tarfile.TarInfo]] = {}
        
        for tarball_path in tarball_paths:
            print(f"Loading audio files: {tarball_path}")
            tar = tarfile.open(tarball_path, "r:*")
            self.tarballs.append(tar)
            
            count = 0
            for m in tar.getmembers():
                if m.isfile() and m.name.endswith(".wav"):
                    if m.name not in self.audio_index:
                        self.audio_index[m.name] = (tar, m)
                        count += 1
                    # If duplicate, first tarball wins (could warn here if needed)
            print(f"  Found {count} audio files in {Path(tarball_path).name}")
        
        print(f"  Total: {len(self.audio_index)} unique audio files across {len(self.tarballs)} tarball(s)")
        
        # Initialize processors
        self.txt_processor = TxtProcessor()
        
        self.style_model = BertModel()
        self.style_processor = StyleProcessor(self.style_model)
        
        self.audio_encoder = FACodecEncoder()
    
    def __del__(self):
        """Close all tarballs on cleanup."""
        if hasattr(self, 'tarballs'):
            for tar in self.tarballs:
                    tar.close()

    
    def item_name_to_path(self, item_name: str) -> str:
        """
        Convert item_name to tarball path.
        Uses same logic as dataset.py: replace '-' with '/' and add '.wav'.
        """
        return str(Path(item_name.replace("-", "/")).with_suffix(".wav"))
    
    def process_text(self, text: str) -> dict:
        """Convert text to phonemes using TxtProcessor."""
        ph, txt, word, ph2word, ph_gb_word = self.txt_processor.txt_to_ph(text)
        phonemes = ph.split()
        
        # Convert phonemes to integer indices for embedding
        unk_idx = self.phoneme_to_idx.get('<PAD>', 0)  # Fallback for unknown phonemes
        phoneme_ids = [self.phoneme_to_idx.get(p, unk_idx) for p in phonemes]
        
        return {
            'phonemes': phonemes,
            'phoneme_ids': phoneme_ids,
            'phoneme_str': ph,
            'cleaned_text': txt,
            'words': word.split(),
            'ph2word': ph2word,
        }
    
    def process_style(self, style_prompt: str) -> torch.Tensor:
        """Embed style prompt using BERT."""
        with torch.no_grad():
            style_emb = self.style_processor.embed(style_prompt)
        return style_emb.cpu().numpy()
    
    def process_audio(self, wav_path: str) -> Tuple[np.ndarray, np.ndarray]:
        try:
            # Look up file in tarball index
            if wav_path not in self.audio_index:
                print(f"  Audio not found in tarball: {wav_path}")
                return None, None
            
            tar, member = self.audio_index[wav_path]
            
            # Extract audio bytes from tarball
            f = tar.extractfile(member)
            if f is None:
                print(f"  Could not extract: {wav_path}")
                return None, None
            audio_bytes = f.read()
            
            # Encode with FACodec (pass bytes directly)
            with torch.no_grad():
                codec, spk_emb = self.audio_encoder.encode(audio_bytes)
            
            return codec.cpu().numpy(), spk_emb.cpu().numpy()
    
        except Exception as e:
            print(f"  Audio encoding error: {e}")
            return None, None
    
    def process_row(self, row: dict) -> Optional[dict]:
        item_name = row['item_name']
        audio_path = self.item_name_to_path(item_name)
        if audio_path is None:
            return None  # Skip if audio not found
        
        text_data = self.process_text(row['txt'])
        
        style_emb = self.process_style(row['style_prompt'])
        
        codec_tokens, spk_emb = self.process_audio(audio_path)
        if codec_tokens is None:
            return None  # Skip if audio encoding failed
        
        return {
            'item_name': item_name,
            'text': row['txt'],
            'phonemes': text_data['phonemes'],
            'phoneme_ids': text_data['phoneme_ids'],
            'phoneme_str': text_data['phoneme_str'],
            'ph2word': text_data['ph2word'],
            'style_emb': style_emb,
            'style_prompt': row['style_prompt'],
            'emotion': row.get('emotion', ''),
            'gender': row.get('gender', ''),
            'speaker': row.get('spk', ''),
            'dur_label': row.get('dur', ''),
            'pitch_label': row.get('pitch', ''),
            'energy_label': row.get('energy', ''),
            'codec_tokens': codec_tokens,
            'spk_emb': spk_emb,
        }
    
    def preprocess(self, csv_path: str, flush_every: int = 100):
        """
        Preprocess entire dataset from CSV.
        Writes to disk every `flush_every` samples to avoid OOM.
        
        Args:
            csv_path: Path to CSV file
            flush_every: Number of samples to buffer before writing to disk
            
        Returns:
            Total number of processed items
        """
        print(f"\nPreprocessing dataset from {csv_path}")
        
        # Read CSV
        with open(csv_path, 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        print(f"Found {len(rows)} rows in CSV")
        
        # Setup output directories
        tensors_dir = self.output_dir / "tensors"
        tensors_dir.mkdir(exist_ok=True)
        
        # Process each row, flushing to disk periodically
        buffer = []
        all_metadata = []
        skipped = 0
        errors = 0
        total_processed = 0
        
        rows_to_process = rows[:10] if self.debug else rows
        
        for row in tqdm(rows_to_process, desc="Processing"):
            try:
                item = self.process_row(row)
                if item is not None:
                    buffer.append(item)
                    total_processed += 1
                    
                    # Flush buffer to disk when full
                    if len(buffer) >= flush_every:
                        self._flush_buffer(buffer, tensors_dir, all_metadata)
                        buffer.clear()
                else:
                    skipped += 1
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"\nError processing {row.get('item_name', 'unknown')}: {e}")
        
        # Flush any remaining items
        if buffer:
            self._flush_buffer(buffer, tensors_dir, all_metadata)
            buffer.clear()
        
        # Save metadata (accumulated throughout processing)
        meta_path = self.output_dir / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        print("\n" + "="*50)
        print("Preprocessing complete:")
        print(f"  Processed: {total_processed}")
        print(f"  Skipped (audio not found): {skipped}")
        print(f"  Errors: {errors}")
        print(f"  Total rows: {len(rows_to_process)}")
        print(f"  Saved metadata to {meta_path}")
        print(f"  Saved tensors to {tensors_dir}")
        print("="*50)
        
        return total_processed
    
    def _flush_buffer(self, buffer: list, tensors_dir: Path, all_metadata: list):
        """Write buffered items to disk and accumulate metadata."""
        for item in buffer:
            item_name = item['item_name'].replace('/', '_').replace(' ', '_')
            
            # Save phoneme ids as integer tensor
            torch.save(item['phoneme_ids'], tensors_dir / f"{item_name}_phonemes.pt")
            
            # Save style embedding
            torch.save(item['style_emb'], tensors_dir / f"{item_name}_style.pt")
            
            # Save audio codec
            if item.get('codec_tokens') is not None:
                torch.save(item['codec_tokens'], tensors_dir / f"{item_name}_codec.pt")
            
            # Save speaker embedding
            if item.get('spk_emb') is not None:
                torch.save(item['spk_emb'], tensors_dir / f"{item_name}_spk_emb.pt")
            
            # Accumulate metadata (small, kept in memory)
            all_metadata.append({
                'item_name': item['item_name'],
                'text': item['text'],
                'phonemes': item['phonemes'],
                'phoneme_str': item['phoneme_str'],
                'ph2word': item['ph2word'],
                'style_prompt': item['style_prompt'],
                'emotion': item.get('emotion', ''),
                'gender': item.get('gender', ''),
                'speaker': item.get('speaker', ''),
                'dur_label': item.get('dur_label', ''),
                'pitch_label': item.get('pitch_label', ''),
                'energy_label': item.get('energy_label', ''),
            })


def preprocess_dataset(
    csv_path: str,
    output_dir: str,
    tarball_paths: Union[str, List[str]],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    sample_rate: int = 16000,
    debug: bool = False,
    phoneme_vocab_path: str = ".",
):
    """
    Convenience function to preprocess dataset.
    
    Args:
        csv_path: Path to CSV file
        output_dir: Directory to save preprocessed data
        tarball_paths: Path(s) to tarball(s) containing audio files.
                       Can be a single path, a list of paths, or comma-separated paths.
        device: Device for model inference
        sample_rate: Target sample rate for audio
        debug: If True, only process first 128 samples
        phoneme_vocab_path: Path to phoneme_vocab.json file or directory containing it
        
    Returns:
        List of processed items
    """
    preprocessor = DatasetPreprocessor(
        output_dir=output_dir,
        tarball_paths=tarball_paths,
        device=device,
        sample_rate=sample_rate,
        debug=debug,
        phoneme_vocab_path=phoneme_vocab_path,
    )
    return preprocessor.preprocess(csv_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess TTS dataset")
    parser.add_argument("--csv_path", type=str, required=True, 
                        help="Path to CSV file")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Output directory for preprocessed data")
    parser.add_argument("--tarball", type=str, nargs='+', required=True, 
                        help="Path(s) to audio tarball(s). Can specify multiple: "
                             "--tarball file1.tar file2.tar or use comma-separated: "
                             "--tarball 'file1.tar,file2.tar'")
    parser.add_argument("--phoneme_vocab_path", type=str, default=".",
                        help="Path to phoneme_vocab.json or directory containing it (default: '.')")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Target sample rate (default: 16000)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for inference (default: cuda if available)")
    parser.add_argument("--debug", action='store_true', default=False,
                        help="Use debug dataset")
    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Flatten tarball paths: handle both multiple args and comma-separated
    tarball_paths = []
    for path in args.tarball:
        tarball_paths.extend(p.strip() for p in path.split(",") if p.strip())
    
    preprocess_dataset(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        tarball_paths=tarball_paths,
        device=device,
        sample_rate=args.sample_rate,
        debug=args.debug,
        phoneme_vocab_path=args.phoneme_vocab_path,
    )
