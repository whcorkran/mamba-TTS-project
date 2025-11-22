"""
Phoneme vocabulary builder for TTS dataset.
Processes the entire dataset to collect all unique phonemes and saves a vocabulary JSON.
"""
import os
import csv
import json
from typing import Set, List, Dict
import nltk

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Dummy tqdm that just returns the iterable
    def tqdm(iterable, desc=None, disable=False):
        return iterable

from preprocess import text_to_phoneme


def build_phoneme_vocabulary(
    csv_path: str,
    output_path: str = "phoneme_vocab.json",
    text_column: str = "txt",
    include_special_tokens: bool = True,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Build phoneme vocabulary from dataset CSV.
    
    Args:
        csv_path: Path to CSV file with text data
        output_path: Path to save the phoneme vocabulary JSON
        text_column: Name of the column containing text to phonemize
        include_special_tokens: If True, include <BOS>, <EOS>, |, and punctuation
        verbose: If True, print progress and statistics
    
    Returns:
        Dictionary mapping phoneme to ID (0-indexed, with 0 reserved for padding/UNK)
    """
    if verbose:
        print(f"Building phoneme vocabulary from {csv_path}...")
    
    # Set to collect all unique phonemes
    phoneme_set: Set[str] = set()
    
    # Special tokens that should be included
    special_tokens = []
    if include_special_tokens:
        special_tokens = ["<BOS>", "<EOS>", "|", "!", ",", ".", ":", ";", "?"]
        phoneme_set.update(special_tokens)
    
    # Read CSV and process all texts
    total_rows = 0
    processed_rows = 0
    errors = 0
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Count total rows for progress bar
            rows_list = list(reader)
            total_rows = len(rows_list)
            
            if verbose:
                print(f"Found {total_rows} rows in CSV")
            
            # Process each row
            for row in tqdm(rows_list, desc="Processing texts", disable=not verbose):
                if text_column not in row:
                    if verbose:
                        print(f"Warning: Column '{text_column}' not found in CSV. Available columns: {list(row.keys())}")
                    break
                
                text = row[text_column].strip()
                if not text:
                    continue
                
                try:
                    # Convert text to phonemes
                    result = text_to_phoneme(text)
                    
                    # Extract phonemes from the phoneme string
                    phoneme_text = result['ph']
                    if phoneme_text:
                        phonemes = phoneme_text.split()
                        phoneme_set.update(phonemes)
                    
                    processed_rows += 1
                    
                except Exception as e:
                    errors += 1
                    if verbose and errors <= 5:  # Only print first 5 errors
                        print(f"Error processing text '{text[:50]}...': {e}")
    
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading CSV file: {e}")
    
    # Convert to sorted list (special tokens first, then phonemes)
    all_phonemes: List[str] = []
    
    # Add special tokens first if included
    if include_special_tokens:
        # Order: padding/UNK (implicit), then special tokens
        all_phonemes.extend([token for token in special_tokens if token in phoneme_set])
    
    # Add regular phonemes (sorted alphabetically)
    regular_phonemes = sorted([ph for ph in phoneme_set if ph not in special_tokens])
    all_phonemes.extend(regular_phonemes)
    
    # Create vocabulary dictionary
    # ID 0 is reserved for padding/UNK
    # IDs 1+ are for actual phonemes
    vocab_dict = {phoneme: idx + 1 for idx, phoneme in enumerate(all_phonemes)}
    
    # Save to JSON
    # Save as list for compatibility with existing code
    vocab_list = ["<PAD>"] + all_phonemes  # Index 0 is padding
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_list, f, ensure_ascii=False, indent=2)
    
    if verbose:
        print("\nVocabulary statistics:")
        print(f"  Total unique phonemes: {len(all_phonemes)}")
        print(f"  Special tokens: {len([t for t in all_phonemes if t in special_tokens])}")
        print(f"  Regular phonemes: {len([p for p in all_phonemes if p not in special_tokens])}")
        print(f"  Processed rows: {processed_rows}/{total_rows}")
        print(f"  Errors: {errors}")
        print(f"  Vocabulary saved to: {output_path}")
        print(f"  Vocabulary size (including padding): {len(vocab_list)}")
        print(f"\nFirst 20 phonemes: {vocab_list[:20]}")
        print(f"Last 10 phonemes: {vocab_list[-10:]}")
    
    return vocab_dict


def load_phoneme_vocabulary(vocab_path: str) -> Dict[str, int]:
    """
    Load phoneme vocabulary from JSON file.
    
    Args:
        vocab_path: Path to phoneme vocabulary JSON file
    
    Returns:
        Dictionary mapping phoneme to ID
    """
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_list = json.load(f)
    
    # Convert list to dict (skip index 0 which is padding)
    vocab_dict = {phoneme: idx for idx, phoneme in enumerate(vocab_list)}
    
    return vocab_dict


def get_phoneme_id(phoneme: str, vocab_dict: Dict[str, int], unk_id: int = 0) -> int:
    """
    Get ID for a phoneme from vocabulary.
    
    Args:
        phoneme: Phoneme string
        vocab_dict: Phoneme vocabulary dictionary
        unk_id: ID to return for unknown phonemes (default: 0 for padding)
    
    Returns:
        Phoneme ID
    """
    return vocab_dict.get(phoneme, unk_id)


if __name__ == "__main__":
    import argparse

    nltk.download('averaged_perceptron_tagger_eng') 

    parser = argparse.ArgumentParser(description="Build phoneme vocabulary from dataset")
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to CSV file with text data"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="phoneme_vocab.json",
        help="Path to save phoneme vocabulary JSON (default: phoneme_vocab.json)"
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="txt",
        help="Name of column containing text (default: txt)"
    )
    parser.add_argument(
        "--no_special_tokens",
        action="store_true",
        help="Exclude special tokens from vocabulary"
    )
    
    args = parser.parse_args()
    
    build_phoneme_vocabulary(
        csv_path=args.csv_path,
        output_path=args.output_path,
        text_column=args.text_column,
        include_special_tokens=not args.no_special_tokens,
        verbose=True,
    )

