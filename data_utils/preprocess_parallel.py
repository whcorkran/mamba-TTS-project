"""
Parallel dataset preprocessing pipeline.
Uses multiprocessing for CPU-bound tasks and batched GPU inference.

Key optimizations:
1. Multiprocessing pool for text/phoneme processing (CPU-bound)
2. Batched GPU inference for FACodec and BERT style embeddings
3. ThreadPoolExecutor for async file I/O
4. Pipeline parallelism: overlap CPU and GPU work

Example usage:
    python -m data_utils.preprocess_parallel \\
        --csv_path VccmDataset/controlspeech_train.csv \\
        --output_dir processed_data/ \\
        --tarball VccmDataset/TextrolSpeech_data.tar.gz \\
        --phoneme_vocab_path . \\
        --cpu_workers 6 \\
        --gpu_batch_size 16 \\
        --io_workers 4
"""

import csv
import json
import tarfile
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import torch
import numpy as np
from tqdm import tqdm

# Text processing imports (lightweight, can be imported in workers)
from .text_processor import TxtProcessor, BertModel
from .audio_encoder import FACodecEncoder


# =============================================================================
# CPU-bound text processing (runs in separate processes)
# =============================================================================

def _init_text_processor():
    """Initialize TxtProcessor in worker process (called once per worker)."""
    global _txt_processor
    _txt_processor = TxtProcessor()


def _process_single_text(args: Tuple[int, str, str, dict]) -> Tuple[int, Optional[dict]]:
    """
    Process a single text sample in worker process.
    
    Args:
        args: (index, item_name, text, phoneme_to_idx_dict)
    
    Returns:
        (index, processed_data) or (index, None) on error
    """
    global _txt_processor
    idx, item_name, text, phoneme_to_idx = args
    
    try:
        ph, txt, word, ph2word, ph_gb_word = _txt_processor.txt_to_ph(text)
        phonemes = ph.split()
        
        # Convert phonemes to indices
        unk_idx = phoneme_to_idx.get('<PAD>', 0)
        phoneme_ids = [phoneme_to_idx.get(p, unk_idx) for p in phonemes]
        
        return idx, {
            'item_name': item_name,
            'text': text,
            'phonemes': phonemes,
            'phoneme_ids': phoneme_ids,
            'phoneme_str': ph,
            'cleaned_text': txt,
            'words': word.split(),
            'ph2word': ph2word,
        }
    except Exception:
        return idx, None


def process_text_parallel(
    rows: List[dict],
    phoneme_to_idx: dict,
    num_workers: int = None,
    text_column: str = 'txt',
    name_column: str = 'item_name',
) -> List[Optional[dict]]:
    """
    Process text to phonemes in parallel using multiprocessing.
    
    Args:
        rows: List of CSV row dictionaries
        phoneme_to_idx: Phoneme vocabulary mapping
        num_workers: Number of worker processes (default: CPU count - 1)
        text_column: Name of text column in CSV
        name_column: Name of item name column in CSV
    
    Returns:
        List of processed text data (same order as input rows)
    """
    num_workers = num_workers or max(1, cpu_count() - 1)
    
    # Prepare arguments for workers
    work_items = [
        (i, row[name_column], row[text_column], phoneme_to_idx)
        for i, row in enumerate(rows)
    ]
    
    results = [None] * len(rows)
    
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_text_processor
    ) as executor:
        futures = {executor.submit(_process_single_text, item): item[0] for item in work_items}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Text processing"):
            idx, result = future.result()
            results[idx] = result
    
    return results


# =============================================================================
# Batched GPU inference
# =============================================================================

class BatchedStyleProcessor:
    """Style processor with batched inference."""
    
    def __init__(self, device: str = 'cuda'):
        self.bert = BertModel()
        self.bert.model = self.bert.model.to(device)
        self.device = device
    
    def embed_batch(self, style_prompts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Embed multiple style prompts in batches.
        
        Args:
            style_prompts: List of style prompt strings
            batch_size: Batch size for inference
        
        Returns:
            List of numpy arrays (style embeddings)
        """
        results = []
        
        for i in range(0, len(style_prompts), batch_size):
            batch = style_prompts[i:i + batch_size]
            
            # Preprocess all prompts
            from .text_processor import TxtProcessor
            processed = [TxtProcessor.preprocess_text(p) for p in batch]
            
            with torch.no_grad():
                tok = self.bert.tokenizer(
                    processed,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                out = self.bert.model(**tok)
                embeddings = out.last_hidden_state[:, 0]  # CLS token
                
                for emb in embeddings:
                    results.append(emb.cpu().numpy())
        
        return results


class BatchedAudioEncoder:
    """FACodec encoder with batched inference."""
    
    def __init__(self, max_seq_len: int = 1024, device: str = 'cuda'):
        self.max_seq_len = max_seq_len
        self.encoder = FACodecEncoder(max_seq_len=max_seq_len)
        self.encoder.fa_encoder = self.encoder.fa_encoder.to(device)
        self.encoder.fa_decoder = self.encoder.fa_decoder.to(device)
        self.device = device
    
    def encode_batch(
        self,
        audio_items: List[Tuple[str, bytes]],  # (path, audio_bytes)
        batch_size: int = 8,
        sr: int = 16000,
    ) -> List[Optional[np.ndarray]]:
        """
        Encode multiple audio files in batches.
        
        Args:
            audio_items: List of (path, audio_bytes) tuples
            batch_size: Batch size for inference
            sr: Sample rate
        
        Returns:
            List of numpy arrays (codec tokens) or None for failures
        """
        import io
        import soundfile as sf
        import torchaudio
        
        results = [None] * len(audio_items)
        
        for i in range(0, len(audio_items), batch_size):
            batch_items = audio_items[i:i + batch_size]
            batch_indices = list(range(i, min(i + batch_size, len(audio_items))))
            
            # Load and preprocess audio
            audio_tensors = []
            valid_indices = []
            
            for j, (path, audio_bytes) in enumerate(batch_items):
                try:
                    if audio_bytes is None:
                        continue
                    
                    audio_data, orig_sr = sf.read(io.BytesIO(audio_bytes))
                    audio = torch.from_numpy(audio_data).float()
                    
                    if audio.ndim == 2:  # stereo -> mono
                        audio = audio.mean(dim=1)
                    
                    if orig_sr != sr:
                        audio = torchaudio.functional.resample(
                            audio.unsqueeze(0), orig_sr, sr
                        ).squeeze(0)
                    
                    audio_tensors.append(audio)
                    valid_indices.append(batch_indices[j])
                except Exception:
                    continue
            
            if not audio_tensors:
                continue
            
            # Pad to same length and batch
            max_len = max(t.shape[0] for t in audio_tensors)
            padded = [
                torch.nn.functional.pad(t, (0, max_len - t.shape[0]))
                for t in audio_tensors
            ]
            audio_batch = torch.stack(padded, dim=0).unsqueeze(1).to(self.device)  # (B, 1, T)
            
            # Encode batch
            with torch.no_grad():
                enc = self.encoder.fa_encoder(audio_batch)
                vq_pos_emb, vq_id, _, quantized, spk_embs = self.encoder.fa_decoder(
                    enc, eval_vq=False, vq=True
                )
                
                # Process each sample in batch
                for j, global_idx in enumerate(valid_indices):
                    try:
                        # Extract this sample's codes
                        Qc = vq_id[0:1, j:j+1, :]
                        Qp = vq_id[1:2, j:j+1, :]
                        Qr = vq_id[2:, j:j+1, :]
                        
                        # Pad/truncate
                        def pad_or_trunc(q, max_len=self.max_seq_len):
                            if q.shape[2] > max_len:
                                return q[:, :, :max_len]
                            elif q.shape[2] < max_len:
                                pad_len = max_len - q.shape[2]
                                return torch.cat([
                                    q, torch.zeros(q.shape[0], q.shape[1], pad_len, device=q.device)
                                ], dim=2)
                            return q
                        
                        Qc = pad_or_trunc(Qc)
                        Qp = pad_or_trunc(Qp)
                        Qr = pad_or_trunc(Qr)
                        
                        Ys = torch.cat((Qp, Qr), dim=0)
                        codec = torch.cat((Ys, Qc), dim=0)
                        
                        # (5, 1, T) -> (1, T, 5) -> (T, 5)
                        results[global_idx] = codec.permute(1, 2, 0).squeeze(0).cpu().numpy()
                    except Exception:
                        continue
        
        return results


# =============================================================================
# Async I/O for file writes
# =============================================================================

class AsyncTensorWriter:
    """Async tensor writing using thread pool."""
    
    def __init__(self, output_dir: Path, max_workers: int = 4):
        self.output_dir = output_dir
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = []
    
    def _write_tensor(self, tensor: torch.Tensor, path: Path):
        """Write a single tensor to disk."""
        torch.save(tensor, path)
    
    def submit(self, item_name: str, phoneme_ids: list, style_emb: np.ndarray, audio: np.ndarray):
        """Submit tensors for async writing."""
        safe_name = item_name.replace('/', '_').replace(' ', '_')
        tensors_dir = self.output_dir / "tensors"
        
        # Phoneme IDs
        phoneme_tensor = torch.tensor(phoneme_ids, dtype=torch.long)
        self.futures.append(
            self.executor.submit(self._write_tensor, phoneme_tensor, tensors_dir / f"{safe_name}_phonemes.pt")
        )
        
        # Style embedding
        style_tensor = torch.from_numpy(style_emb) if isinstance(style_emb, np.ndarray) else style_emb
        self.futures.append(
            self.executor.submit(self._write_tensor, style_tensor, tensors_dir / f"{safe_name}_style.pt")
        )
        
        # Audio codec
        if audio is not None:
            audio_tensor = torch.from_numpy(audio) if isinstance(audio, np.ndarray) else audio
            self.futures.append(
                self.executor.submit(self._write_tensor, audio_tensor, tensors_dir / f"{safe_name}_codec.pt")
            )
    
    def wait(self):
        """Wait for all pending writes to complete."""
        for future in self.futures:
            future.result()
        self.futures.clear()
    
    def shutdown(self):
        """Shutdown the executor."""
        self.wait()
        self.executor.shutdown(wait=True)


# =============================================================================
# Main parallel preprocessor
# =============================================================================

class ParallelDatasetPreprocessor:
    """
    Parallel preprocessor with optimized pipeline:
    1. Text processing in parallel (multiprocessing)
    2. Audio extraction from tarball (can be parallelized)
    3. Batched GPU inference for style and audio encoding
    4. Async file writes
    """
    
    def __init__(
        self,
        output_dir: str,
        tarball_paths: Union[str, List[str]],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        sample_rate: int = 16000,
        debug: bool = False,
        phoneme_vocab_path: str = ".",
        num_cpu_workers: int = None,
        gpu_batch_size: int = 16,
        io_workers: int = 4,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "tensors").mkdir(exist_ok=True)
        
        self.sample_rate = sample_rate
        self.debug = debug
        self.device = device
        self.num_cpu_workers = num_cpu_workers or max(1, cpu_count() - 2)
        self.gpu_batch_size = gpu_batch_size
        self.io_workers = io_workers
        
        # Load phoneme vocabulary
        vocab_path = Path(phoneme_vocab_path)
        if vocab_path.is_dir():
            vocab_path = vocab_path / "phoneme_vocab.json"
        if not vocab_path.exists():
            raise FileNotFoundError(f"Phoneme vocabulary not found at {vocab_path}")
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.phoneme_vocab = json.load(f)
        self.phoneme_to_idx = {ph: idx for idx, ph in enumerate(self.phoneme_vocab)}
        print(f"Loaded phoneme vocabulary ({len(self.phoneme_vocab)} tokens)")
        
        # Build tarball index
        if isinstance(tarball_paths, str):
            tarball_paths = [p.strip() for p in tarball_paths.split(",") if p.strip()]
        
        self.tarballs: List[tarfile.TarFile] = []
        self.audio_index: Dict[str, Tuple[tarfile.TarFile, tarfile.TarInfo]] = {}
        
        for tarball_path in tarball_paths:
            print(f"Indexing tarball: {tarball_path}")
            tar = tarfile.open(tarball_path, "r:*")
            self.tarballs.append(tar)
            
            for m in tar.getmembers():
                if m.isfile() and m.name.endswith(".wav"):
                    if m.name not in self.audio_index:
                        self.audio_index[m.name] = (tar, m)
        
        print(f"Indexed {len(self.audio_index)} audio files")
        
        # Initialize GPU processors (lazy)
        self._style_processor = None
        self._audio_encoder = None
    
    @property
    def style_processor(self):
        if self._style_processor is None:
            print("Initializing batched style processor...")
            self._style_processor = BatchedStyleProcessor(device=self.device)
        return self._style_processor
    
    @property
    def audio_encoder(self):
        if self._audio_encoder is None:
            print("Initializing batched audio encoder...")
            self._audio_encoder = BatchedAudioEncoder(device=self.device)
        return self._audio_encoder
    
    def __del__(self):
        if hasattr(self, 'tarballs'):
            for tar in self.tarballs:
                tar.close()
    
    def item_name_to_path(self, item_name: str) -> str:
        return str(Path(item_name.replace("-", "/")).with_suffix(".wav"))
    
    def extract_audio_bytes(self, wav_path: str) -> Optional[bytes]:
        """Extract audio bytes from tarball."""
        if wav_path not in self.audio_index:
            return None
        tar, member = self.audio_index[wav_path]
        f = tar.extractfile(member)
        if f is None:
            return None
        return f.read()
    
    def preprocess(self, csv_path: str) -> int:
        """
        Parallel preprocessing pipeline.
        
        Pipeline stages:
        1. Load CSV and prepare work items
        2. Process text to phonemes (multiprocessing)
        3. Extract audio bytes (can be parallelized)
        4. Batch style embeddings (GPU)
        5. Batch audio encoding (GPU)
        6. Async write to disk
        """
        print(f"\n{'='*60}")
        print("Starting parallel preprocessing")
        print(f"{'='*60}")
        
        # Stage 1: Load CSV
        print("\n[1/6] Loading CSV...")
        with open(csv_path, 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        
        if self.debug:
            rows = rows[:10]
        print(f"  {len(rows)} rows to process")
        
        # Stage 2: Parallel text processing
        print(f"\n[2/6] Text processing ({self.num_cpu_workers} workers)...")
        text_results = process_text_parallel(
            rows, self.phoneme_to_idx, num_workers=self.num_cpu_workers
        )
        
        # Stage 3: Extract audio bytes
        print("\n[3/6] Extracting audio from tarball...")
        audio_items = []  # (path, bytes)
        for row in tqdm(rows, desc="Audio extraction"):
            path = self.item_name_to_path(row['item_name'])
            audio_bytes = self.extract_audio_bytes(path)
            audio_items.append((path, audio_bytes))
        
        # Stage 4: Batch style embeddings
        print(f"\n[4/6] Computing style embeddings (batch_size={self.gpu_batch_size})...")
        style_prompts = [row['style_prompt'] for row in rows]
        style_embeddings = self.style_processor.embed_batch(
            style_prompts, batch_size=self.gpu_batch_size
        )
        
        # Stage 5: Batch audio encoding
        print(f"\n[5/6] Encoding audio (batch_size={self.gpu_batch_size})...")
        audio_codes = []
        for i in tqdm(range(0, len(audio_items), self.gpu_batch_size), desc="Audio encoding"):
            batch = audio_items[i:i + self.gpu_batch_size]
            batch_results = self.audio_encoder.encode_batch(
                batch, batch_size=len(batch), sr=self.sample_rate
            )
            audio_codes.extend(batch_results)
        
        # Stage 6: Async write to disk
        print(f"\n[6/6] Writing to disk ({self.io_workers} I/O workers)...")
        writer = AsyncTensorWriter(self.output_dir, max_workers=self.io_workers)
        all_metadata = []
        processed = 0
        skipped = 0
        
        for i, row in enumerate(tqdm(rows, desc="Writing")):
            text_data = text_results[i]
            style_emb = style_embeddings[i]
            audio_data = audio_codes[i]
            
            if text_data is None or audio_data is None:
                skipped += 1
                continue
            
            # Submit for async write
            writer.submit(
                row['item_name'],
                text_data['phoneme_ids'],
                style_emb,
                audio_data
            )
            
            # Accumulate metadata
            all_metadata.append({
                'item_name': row['item_name'],
                'text': row['txt'],
                'phonemes': text_data['phonemes'],
                'phoneme_str': text_data['phoneme_str'],
                'ph2word': text_data['ph2word'],
                'style_prompt': row['style_prompt'],
                'emotion': row.get('emotion', ''),
                'gender': row.get('gender', ''),
                'speaker': row.get('spk', ''),
                'dur_label': row.get('dur', ''),
                'pitch_label': row.get('pitch', ''),
                'energy_label': row.get('energy', ''),
            })
            processed += 1
        
        # Wait for all writes to complete
        writer.shutdown()
        
        # Save metadata
        meta_path = self.output_dir / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        print(f"\n{'='*60}")
        print("Preprocessing complete:")
        print(f"  Processed: {processed}")
        print(f"  Skipped: {skipped}")
        print(f"  Metadata: {meta_path}")
        print(f"{'='*60}")
        
        return processed


def preprocess_dataset_parallel(
    csv_path: str,
    output_dir: str,
    tarball_paths: Union[str, List[str]],
    device: str = None,
    sample_rate: int = 16000,
    debug: bool = False,
    phoneme_vocab_path: str = ".",
    num_cpu_workers: int = None,
    gpu_batch_size: int = 16,
    io_workers: int = 4,
) -> int:
    """Convenience function for parallel preprocessing."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    preprocessor = ParallelDatasetPreprocessor(
        output_dir=output_dir,
        tarball_paths=tarball_paths,
        device=device,
        sample_rate=sample_rate,
        debug=debug,
        phoneme_vocab_path=phoneme_vocab_path,
        num_cpu_workers=num_cpu_workers,
        gpu_batch_size=gpu_batch_size,
        io_workers=io_workers,
    )
    return preprocessor.preprocess(csv_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parallel TTS dataset preprocessing")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tarball", type=str, nargs='+', required=True)
    parser.add_argument("--phoneme_vocab_path", type=str, default=".")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--cpu_workers", type=int, default=None,
                        help="Number of CPU workers for text processing")
    parser.add_argument("--gpu_batch_size", type=int, default=16,
                        help="Batch size for GPU inference")
    parser.add_argument("--io_workers", type=int, default=4,
                        help="Number of I/O workers for file writes")
    
    args = parser.parse_args()
    
    tarball_paths = []
    for path in args.tarball:
        tarball_paths.extend(p.strip() for p in path.split(",") if p.strip())
    
    preprocess_dataset_parallel(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        tarball_paths=tarball_paths,
        device=args.device,
        sample_rate=args.sample_rate,
        debug=args.debug,
        phoneme_vocab_path=args.phoneme_vocab_path,
        num_cpu_workers=args.cpu_workers,
        gpu_batch_size=args.gpu_batch_size,
        io_workers=args.io_workers,
    )

