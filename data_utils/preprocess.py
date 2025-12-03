"""
Dataset preprocessing pipeline.
Preprocesses CSV dataset: audio encoding, text-to-phoneme, style embeddings.
Saves processed data for PyTorch DataLoader.
"""
import io
import csv
import json
import tarfile
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple

import librosa
import torch
import torchaudio
import numpy as np
from tqdm import tqdm

from .text_processor import TxtProcessor, BertModel, StyleProcessor
from .audio_encoder import FACodecEncoder, BaseAudioPreprocessor
from .phonemes import SPECIAL_TOKENS


class DatasetPreprocessor:
    """Preprocesses dataset: audio, phonemes, style embeddings."""
    
    def __init__(
        self,
        output_dir: str,
        tarball_paths: Union[str, List[str]],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        sample_rate: int = 16000,
    ):
        """
        Initialize preprocessor.
        
        Args:
            output_dir: Directory to save preprocessed data
            tarball_paths: Path(s) to tarball(s) containing audio files.
                           Can be a single path, a list of paths, or comma-separated paths.
            device: Device for model inference
            sample_rate: Target sample rate for audio
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.sample_rate = sample_rate
        
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
        
        # Phoneme vocabulary (accumulated during processing)
        self.phoneme_set = set()
    
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
        self.phoneme_set.update(phonemes)
        
        return {
            'phonemes': phonemes,
            'phoneme_str': ph,
            'cleaned_text': txt,
            'words': word.split(),
            'ph2word': ph2word,
        }
    
    def process_style(self, style_prompt: str) -> np.ndarray:
        """Embed style prompt using BERT."""
        with torch.no_grad():
            style_emb = self.style_processor.embed(style_prompt)
        return style_emb.cpu().numpy()
    
    def process_audio(self, wav_path: str) -> np.ndarray:
        try:
            # Encode with FACodec
            with torch.no_grad():
                codec, spk_emb = self.audio_encoder.encode(wav_path)
            
            return codec.cpu().numpy()
    
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
        
        for row in tqdm(rows, desc="Processing"):
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
        print(f"\nSaving to {self.output_dir}")

        # Reuse phoneme vocabulary logic from phonemes.py
        vocab = SPECIAL_TOKENS.copy()
        vocab.extend(sorted(p for p in self.phoneme_set if p not in SPECIAL_TOKENS))

        vocab_path = self.output_dir / "phoneme_vocab.json"
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, indent=2)
        print(f"  Saved vocabulary ({len(vocab)}) to {vocab_path}")
        
        # Save metadata (JSON-serializable parts)
        metadata = []
        for item in processed:
            meta = {
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
            }
            metadata.append(meta)
        
        meta_path = self.output_dir / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved metadata ({len(metadata)} items) to {meta_path}")
        
        # Save tensors
        tensors_dir = self.output_dir / "tensors"
        tensors_dir.mkdir(exist_ok=True)
        
        for item in tqdm(processed, desc="Saving tensors"):
            item_name = item['item_name'].replace('/', '_').replace(' ', '_')
            style_path = tensors_dir / f"{item_name}_style.npy"
            np.save(style_path, item['style_emb'])
            
            if item.get('audio') and item['audio'].get('codec') is not None:
                codec_path = tensors_dir / f"{item_name}_codec.npy"
                np.save(codec_path, item['audio']['codec'])
                
                if item['audio'].get('spk_emb') is not None:
                    spk_path = tensors_dir / f"{item_name}_spk.npy"
                    np.save(spk_path, item['audio']['spk_emb'])
        
        print("  Saved tensors to", tensors_dir)


def preprocess_dataset(
    csv_path: str,
    output_dir: str,
    tarball_paths: Union[str, List[str]],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    sample_rate: int = 16000,
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
        
    Returns:
        List of processed items
    """
    preprocessor = DatasetPreprocessor(
        output_dir=output_dir,
        tarball_paths=tarball_paths,
        device=device,
        sample_rate=sample_rate,
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
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Target sample rate (default: 16000)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for inference (default: cuda if available)")
    
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
    )
