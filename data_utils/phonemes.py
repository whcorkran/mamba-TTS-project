"""
Phoneme vocabulary builder for TTS dataset.
Processes dataset to collect all unique phonemes and saves vocabulary JSON.
"""
import csv
import json
import argparse
from tqdm import tqdm

from data_utils.text_processor import TxtProcessor


SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "|", "!", ",", ".", ":", ";", "?"]


def build_phoneme_vocabulary(csv_path: str, output_path: str = "phoneme_vocab.json", text_column: str = "txt"):
    """Build phoneme vocabulary from dataset CSV."""
    print(f"Building phoneme vocabulary from {csv_path}...")
    
    phoneme_set = set()
    processor = TxtProcessor()
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    
    print(f"Processing {len(rows)} rows...")
    
    for row in tqdm(rows, desc="Converting to phonemes"):
        text = row.get(text_column, "").strip()
        if not text:
            continue
        
        try:
            ph, _, _, _, _ = processor.txt_to_ph(text)
            phoneme_set.update(ph.split())
        except Exception as e:
            print(f"Error: {e}")
    
    # Build vocabulary: special tokens first, then sorted phonemes
    vocab = SPECIAL_TOKENS.copy()
    vocab.extend(sorted(p for p in phoneme_set if p not in SPECIAL_TOKENS))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2)
    
    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Saved to: {output_path}")
    
    return vocab


def load_phoneme_vocabulary(vocab_path: str) -> dict:
    """Load phoneme vocabulary from JSON file."""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_list = json.load(f)
    return {ph: idx for idx, ph in enumerate(vocab_list)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build phoneme vocabulary from dataset")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--output_path", type=str, default="phoneme_vocab.json", help="Output path")
    parser.add_argument("--text_column", type=str, default="txt", help="Text column name")
    
    args = parser.parse_args()
    build_phoneme_vocabulary(args.csv_path, args.output_path, args.text_column)
