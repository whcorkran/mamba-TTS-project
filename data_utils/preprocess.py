"""
Dataset preprocessing pipeline.
Preprocesses CSV dataset: audio encoding, text-to-phoneme, style embeddings.
Saves processed data for PyTorch DataLoader.
"""
import csv
import json
import tarfile
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple

import torch
from tqdm import tqdm

from .text_processor import TxtProcessor, BertModel, StyleProcessor
from .audio_encoder import FACodecEncoder, BaseAudioPreprocessor


class DatasetPreprocessor:
    """Preprocesses dataset: audio, phonemes, style embeddings."""
    
    def __init__(
        self,
        output_dir: str,
        tarball_paths: Union[str, List[str]],
        phoneme_vocab_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        sample_rate: int = 16000,
        debug: bool = False,
    ):
        """
        Initialize preprocessor.
        
        Args:
            output_dir: Directory to save preprocessed data
            tarball_paths: Path(s) to tarball(s) containing audio files.
                           Can be a single path, a list of paths, or comma-separated paths.
            phoneme_vocab_path: Path to phoneme vocabulary JSON file
            device: Device for model inference
            sample_rate: Target sample rate for audio
            debug: If True, only process first 128 samples
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.sample_rate = sample_rate
        self.debug = debug
        
        # Load phoneme vocabulary
        print(f"Loading phoneme vocabulary: {phoneme_vocab_path}")
        with open(phoneme_vocab_path, 'r', encoding='utf-8') as f:
            self.phoneme_vocab: List[str] = json.load(f)
        self.phoneme_to_idx: Dict[str, int] = {p: i for i, p in enumerate(self.phoneme_vocab)}
        print(f"  Loaded {len(self.phoneme_vocab)} phonemes")
        
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
        self.audio_preprocessor = BaseAudioPreprocessor(sample_rate=sample_rate)
    
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
        
        # Convert phonemes to integer indices
        phoneme_ids = []
        for p in phonemes:
            if p in self.phoneme_to_idx:
                phoneme_ids.append(self.phoneme_to_idx[p])
            else:
                # Unknown phoneme - use <PAD> (index 0) as fallback
                print(f"  Warning: Unknown phoneme '{p}', using <PAD>")
                phoneme_ids.append(0)
        
        return {
            'phonemes': phonemes,
            'phoneme_ids': torch.tensor(phoneme_ids, dtype=torch.long),
            'phoneme_str': ph,
            'cleaned_text': txt,
            'words': word.split(),
            'ph2word': ph2word,
        }
    
    def process_style(self, style_prompt: str) -> torch.Tensor:
        """Embed style prompt using BERT."""
        with torch.no_grad():
            style_emb = self.style_processor.embed(style_prompt)
        return style_emb.cpu()
    
    def process_audio(self, wav_path: str) -> torch.Tensor:
        try:
            # Encode with FACodec
            with torch.no_grad():
                codec, spk_emb = self.audio_encoder.encode(wav_path)
            
            return codec.cpu()
    
        except Exception as e:
            print(f"  Audio encoding error: {e}")
            return None
    
    def process_row(self, row: dict) -> Optional[dict]:
        item_name = row['item_name']
        audio_path = self.item_name_to_path(item_name)
        if audio_path is None:
            return None  # Skip if audio not found
        
        text_data = self.process_text(row['txt'])
        
        style_emb = self.process_style(row['style_prompt'])
        
        audio_data = self.process_audio(audio_path)
        if audio_data is None:
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
            'audio': audio_data,
        }
    
    def preprocess(self, csv_path: str):
        """
        Preprocess entire dataset from CSV.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            List of processed items
        """
        print(f"\nPreprocessing dataset from {csv_path}")
        
        # Read CSV
        with open(csv_path, 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        print(f"Found {len(rows)} rows in CSV")
        
        # Process each row
        processed = []
        skipped = 0
        errors = 0
        
        for row in tqdm(rows[:128] if self.debug else rows, desc="Processing"):
            try:
                item = self.process_row(row)
                if item is not None:
                    processed.append(item)
                else:
                    skipped += 1
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"\nError processing {row.get('item_name', 'unknown')}: {e}")
        
        print("\n" + "="*50)
        print("Preprocessing complete:")
        print(f"  Processed: {len(processed)}")
        print(f"  Skipped (audio not found): {skipped}")
        print(f"  Errors: {errors}")
        print(f"  Total rows: {len(rows)}")
        print("="*50)
        
        self.save_processed(processed)
        
        return processed
    
    def save_processed(self, processed: list):
        """
        Save processed data in DataLoader-friendly format as PyTorch tensors.
        
        Directory structure:
            output_dir/
            ├── metadata.json          # Sample index with text metadata
            ├── style_embeddings/      # BERT style embeddings
            │   └── {item_name}.pt
            ├── phoneme_ids/           # Phoneme integer sequences
            │   └── {item_name}.pt
            └── audio_codecs/          # FACodec audio tensors
                └── {item_name}.pt
        """
        print(f"\nSaving to {self.output_dir}")
        
        # Create subdirectories for each tensor type
        style_dir = self.output_dir / "style_embeddings"
        phoneme_dir = self.output_dir / "phoneme_ids"
        audio_dir = self.output_dir / "audio_codecs"
        
        style_dir.mkdir(exist_ok=True)
        phoneme_dir.mkdir(exist_ok=True)
        audio_dir.mkdir(exist_ok=True)
        
        # Save metadata (JSON-serializable parts)
        metadata = []
        for item in processed:
            item_name = item['item_name'].replace('/', '_').replace(' ', '_')
            meta = {
                'item_name': item['item_name'],
                'file_key': item_name,  # Sanitized name for file lookups
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
            }
            metadata.append(meta)
        
        meta_path = self.output_dir / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved metadata ({len(metadata)} items) to {meta_path}")
        
        # Save tensors organized by type
        for item in tqdm(processed, desc="Saving tensors"):
            item_name = item['item_name'].replace('/', '_').replace(' ', '_')
            
            # Save style embeddings
            style_path = style_dir / f"{item_name}.pt"
            torch.save(item['style_emb'], style_path)
            
            # Save phoneme IDs
            phoneme_path = phoneme_dir / f"{item_name}.pt"
            torch.save(item['phoneme_ids'], phoneme_path)
            
            # Save audio codecs
            if item.get('audio') is not None:
                audio_path = audio_dir / f"{item_name}.pt"
                torch.save(item['audio'], audio_path)
        
        print(f"  Saved style embeddings to {style_dir}")
        print(f"  Saved phoneme IDs to {phoneme_dir}")
        print(f"  Saved audio codecs to {audio_dir}")


def preprocess_dataset(
    csv_path: str,
    output_dir: str,
    tarball_paths: Union[str, List[str]],
    phoneme_vocab_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    sample_rate: int = 16000,
    debug: bool = False,
):
    """
    Convenience function to preprocess dataset.
    
    Args:
        csv_path: Path to CSV file
        output_dir: Directory to save preprocessed data
        tarball_paths: Path(s) to tarball(s) containing audio files.
                       Can be a single path, a list of paths, or comma-separated paths.
        phoneme_vocab_path: Path to phoneme vocabulary JSON file
        device: Device for model inference
        sample_rate: Target sample rate for audio
        debug: If True, only process first 128 samples
        
    Returns:
        List of processed items
    """
    preprocessor = DatasetPreprocessor(
        output_dir=output_dir,
        tarball_paths=tarball_paths,
        phoneme_vocab_path=phoneme_vocab_path,
        device=device,
        sample_rate=sample_rate,
        debug=debug,
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
    parser.add_argument("--phoneme_vocab", type=str, required=True,
                        help="Path to phoneme vocabulary JSON file")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Target sample rate (default: 16000)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for inference (default: cuda if available)")
    parser.add_argument("--debug", action='store_true', default=False,
                        help="Use debug dataset (process only first 128 samples)")
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
        phoneme_vocab_path=args.phoneme_vocab,
        device=device,
        sample_rate=args.sample_rate,
        debug=args.debug,
    )
