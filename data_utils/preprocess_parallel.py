"""
Parallel dataset preprocessing pipeline.
Uses multiprocessing for CPU-bound tasks and batched GPU inference.

Key optimizations:
1. Multiprocessing pool for text/phoneme processing (CPU-bound)
2. Batched GPU inference for FACodec and BERT style embeddings
3. ThreadPoolExecutor for async file I/O
4. Pipeline parallelism: overlap CPU and GPU work

Example usage (recommended - using extracted audio directory):
    python -m data_utils.preprocess_parallel \\
        --csv_path VccmDataset/controlspeech_train.csv \\
        --output_dir processed_data/ \\
        --audio_dir VccmDataset/audio_extracted \\
        --phoneme_vocab_path . \\
        --cpu_workers 6 \\
        --gpu_batch_size 16 \\
        --io_workers 4

Alternative (slower - using tarball):
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
        num_batches = (len(style_prompts) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(style_prompts), batch_size), total=num_batches, desc="Style embeddings"):
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
    """FACodec encoder with batched inference and prefetching for high GPU utilization."""
    
    def __init__(self, max_seq_len: int = 1024, device: str = 'cuda', num_load_workers: int = 64):
        self.max_seq_len = max_seq_len
        self.encoder = FACodecEncoder(max_seq_len=max_seq_len)
        self.encoder.fa_encoder = self.encoder.fa_encoder.to(device)
        self.encoder.fa_decoder = self.encoder.fa_decoder.to(device)
        self.device = device
        self.num_load_workers = num_load_workers
        
        # Import here to avoid issues in worker processes
        import io
        import soundfile as sf
        import torchaudio
        self._io = io
        self._sf = sf
        self._torchaudio = torchaudio
    
    def _load_single_audio(self, args: Tuple[int, str, bytes, int]) -> Tuple[int, Optional[torch.Tensor]]:
        """Load and preprocess a single audio file. Used by ThreadPoolExecutor."""
        idx, path, audio_bytes, sr = args
        try:
            if audio_bytes is None:
                return idx, None
            
            audio_data, orig_sr = self._sf.read(self._io.BytesIO(audio_bytes))
            audio = torch.from_numpy(audio_data).float()
            
            if audio.ndim == 2:  # stereo -> mono
                audio = audio.mean(dim=1)
            
            if orig_sr != sr:
                audio = self._torchaudio.functional.resample(
                    audio.unsqueeze(0), orig_sr, sr
                ).squeeze(0)
            
            return idx, audio
        except Exception:
            return idx, None
    
    def _prepare_batch(
        self, 
        batch_items: List[Tuple[str, bytes]], 
        batch_indices: List[int],
        sr: int,
        executor: ThreadPoolExecutor,
    ) -> Tuple[Optional[torch.Tensor], List[int]]:
        """
        Prepare a batch by loading audio files in parallel.
        
        Returns:
            (audio_batch tensor on GPU, list of valid global indices) or (None, [])
        """
        # Submit all load tasks in parallel
        load_args = [(j, path, audio_bytes, sr) for j, (path, audio_bytes) in enumerate(batch_items)]
        futures = [executor.submit(self._load_single_audio, args) for args in load_args]
        
        # Collect results
        audio_tensors = []
        valid_indices = []
        for future in futures:
            local_idx, audio = future.result()
            if audio is not None:
                audio_tensors.append(audio)
                valid_indices.append(batch_indices[local_idx])
        
        if not audio_tensors:
            return None, []
        
        # Pad to same length and batch
        max_len = max(t.shape[0] for t in audio_tensors)
        padded = [
            torch.nn.functional.pad(t, (0, max_len - t.shape[0]))
            for t in audio_tensors
        ]
        # Stack and move to GPU
        audio_batch = torch.stack(padded, dim=0).unsqueeze(1).to(self.device)  # (B, 1, T)
        
        return audio_batch, valid_indices
    
    def _encode_on_gpu(self, audio_batch: torch.Tensor, valid_indices: List[int]) -> Dict[int, np.ndarray]:
        """Run FACodec encoding on GPU. Returns dict mapping global_idx -> codec array."""
        results = {}
        
        with torch.no_grad():
            enc = self.encoder.fa_encoder(audio_batch)
            
            # V2 API requires prosody features
            prosody = self.encoder.fa_encoder.get_prosody_feature(audio_batch)
            
            # Align prosody length to match enc length (they can differ by 1 frame)
            if prosody.shape[2] != enc.shape[2]:
                if prosody.shape[2] > enc.shape[2]:
                    prosody = prosody[:, :, :enc.shape[2]]
                else:
                    pad_len = enc.shape[2] - prosody.shape[2]
                    prosody = torch.nn.functional.pad(prosody, (0, pad_len))
            
            vq_pos_emb, vq_id, _, quantized, spk_embs = self.encoder.fa_decoder(
                enc, prosody, eval_vq=False, vq=True
            )
            
            # Process each sample in batch
            # FACodec V2 ordering: prosody(1) + content(2) + residual(3) = 6 quantizers
            for j, global_idx in enumerate(valid_indices):
                try:
                    # Extract this sample's codes with correct V2 ordering
                    Qp = vq_id[:1, j:j+1, :]      # prosody (1, 1, T)
                    Qc = vq_id[1:3, j:j+1, :]     # content (2, 1, T)
                    Qr = vq_id[3:, j:j+1, :]      # residual (3, 1, T)
                    
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
                    
                    # Style codec = prosody + residuals (as per ControlSpeech)
                    Ys = torch.cat((Qp, Qr), dim=0)  # (4, 1, T)
                    codec = torch.cat((Ys, Qc), dim=0)  # (6, 1, T)
                    
                    # (6, 1, T) -> (1, T, 6) -> (T, 6)
                    results[global_idx] = codec.permute(1, 2, 0).squeeze(0).cpu().numpy()
                except Exception:
                    continue
        
        return results
    
    def encode_batch(
        self,
        audio_items: List[Tuple[str, bytes]],  # (path, audio_bytes)
        batch_size: int = 64,
        sr: int = 16000,
    ) -> List[Optional[np.ndarray]]:
        """
        Encode audio with parallel file loading.
        
        Uses ThreadPoolExecutor to load all files in a batch simultaneously,
        then runs GPU encoding. Simple and effective.
        """
        results = [None] * len(audio_items)
        num_batches = (len(audio_items) + batch_size - 1) // batch_size
        
        # Single executor reused across all batches
        with ThreadPoolExecutor(max_workers=self.num_load_workers) as executor:
            for batch_num in tqdm(range(num_batches), desc="Audio encoding"):
                i = batch_num * batch_size
                batch_items = audio_items[i:i + batch_size]
                batch_indices = list(range(i, min(i + batch_size, len(audio_items))))
                
                # Parallel load + prepare batch
                audio_batch, valid_indices = self._prepare_batch(
                    batch_items, batch_indices, sr, executor
                )
                
                if audio_batch is None:
                    continue
                
                # GPU encode
                batch_results = self._encode_on_gpu(audio_batch, valid_indices)
                
                # Store results
                for global_idx, codec in batch_results.items():
                    results[global_idx] = codec
        
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
        tarball_paths: Union[str, List[str]] = None,
        audio_dir: str = None,
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
        
        # Audio source: either directory or tarball(s)
        self.audio_dir = Path(audio_dir) if audio_dir else None
        self.tarballs: List[tarfile.TarFile] = []
        # Primary index: relative path -> source (Path or (tar, member))
        self.audio_index: Dict[str, Union[Path, Tuple[tarfile.TarFile, tarfile.TarInfo]]] = {}
        # Secondary index: filename -> relative path (for datasets with non-trivial path mapping)
        self.filename_index: Dict[str, str] = {}
        
        if self.audio_dir and self.audio_dir.exists():
            # Index audio files from directory (much faster)
            print(f"Indexing audio directory: {self.audio_dir}")
            audio_count = 0
            for wav_path in tqdm(self.audio_dir.rglob("*.wav"), desc="Indexing audio files"):
                # Store relative path from audio_dir as key, full path as value
                rel_path = str(wav_path.relative_to(self.audio_dir))
                self.audio_index[rel_path] = wav_path
                # Also index by filename for flexible lookup
                filename = wav_path.stem  # filename without extension
                if filename not in self.filename_index:
                    self.filename_index[filename] = rel_path
                audio_count += 1
            print(f"Indexed {audio_count:,} audio files from directory")
        elif tarball_paths:
            # Build tarball index
            if isinstance(tarball_paths, str):
                tarball_paths = [p.strip() for p in tarball_paths.split(",") if p.strip()]
            
            for tarball_path in tarball_paths:
                print(f"Indexing tarball: {tarball_path}")
                print(f"  (Reading compressed archive - this may take several minutes for large .tar.gz files...)")
                tar = tarfile.open(tarball_path, "r:*")
                self.tarballs.append(tar)
                
                # For .tar.gz, getmembers() must read entire archive first (no progress possible)
                # Use iterator approach to show progress during indexing
                audio_count = 0
                member_count = 0
                while True:
                    m = tar.next()
                    if m is None:
                        break
                    member_count += 1
                    if member_count % 10000 == 0:
                        print(f"  Scanned {member_count:,} entries, found {audio_count:,} audio files...")
                    if m.isfile() and m.name.endswith(".wav"):
                        if m.name not in self.audio_index:
                            self.audio_index[m.name] = (tar, m)
                            # Also index by filename for flexible lookup
                            filename = Path(m.name).stem
                            if filename not in self.filename_index:
                                self.filename_index[filename] = m.name
                            audio_count += 1
                print(f"  Done: scanned {member_count:,} entries, indexed {audio_count:,} audio files")
        else:
            raise ValueError("Must provide either --audio_dir or --tarball")
        
        print(f"Total indexed: {len(self.audio_index)} audio files")
        
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
    
    def item_name_to_path(self, item_name: str) -> Optional[str]:
        """
        Convert item_name to audio file path using multiple strategies.
        
        Handles various naming conventions:
        - ESD/MEAD/etc: "Dataset-SubDir-Speaker-Emotion-filename" -> "Dataset/SubDir/Speaker/Emotion/filename.wav"
        - LibriTTS: "speaker_chapter_seq_subseq" -> lookup by filename
        - CREMA: "CREMA-D-AudioWAV-filename" -> "CREMA/AudioWAV/filename.wav" (D is part of name, not dir)
        - RAVDESS: "RAVDESS-Actor_XX-aa-bb-cc-..." -> "RAVDESS/Actor_XX/aa-bb-cc-....wav"
        """
        # Strategy 1: Try direct hyphen-to-slash conversion (works for ESD, MEAD, TESS, etc.)
        direct_path = str(Path(item_name.replace("-", "/")).with_suffix(".wav"))
        if direct_path in self.audio_index:
            return direct_path
        
        # Strategy 2: Extract filename and look up in filename index
        # This handles LibriTTS (no hyphens) and mismatched path structures
        
        # For items without hyphens (LibriTTS), the whole thing is the filename
        if "-" not in item_name:
            filename = item_name
        else:
            # For hyphenated names, try to extract the actual filename
            parts = item_name.split("-")
            
            # Special case: RAVDESS - "RAVDESS-Actor_XX-aa-bb-cc-dd-ee-ff-gg"
            # Filename is everything after "Actor_XX-"
            if item_name.startswith("RAVDESS-Actor_"):
                # Find where Actor_XX ends
                actor_part = parts[1]  # "Actor_04"
                rest_start = len("RAVDESS-") + len(actor_part) + 1  # +1 for the hyphen
                filename = item_name[rest_start:]  # "03-01-05-01-02-01-04"
            
            # Special case: CREMA-D - "CREMA-D-AudioWAV-filename"
            # The "D" is part of dataset name, not a directory
            elif item_name.startswith("CREMA-D-"):
                filename = parts[-1]  # Last segment is the filename
            
            else:
                # Default: last segment is the filename
                filename = parts[-1]
        
        # Look up in filename index
        if filename in self.filename_index:
            return self.filename_index[filename]
        
        # Strategy 3: Try without the file extension in case of mismatch
        filename_no_ext = filename.replace(".wav", "")
        if filename_no_ext in self.filename_index:
            return self.filename_index[filename_no_ext]
        
        return None
    
    def extract_audio_bytes(self, wav_path: Optional[str]) -> Optional[bytes]:
        """Extract audio bytes from tarball or directory."""
        if wav_path is None or wav_path not in self.audio_index:
            return None
        
        source = self.audio_index[wav_path]
        
        # If source is a Path, read from file
        if isinstance(source, Path):
            try:
                return source.read_bytes()
            except Exception:
                return None
        
        # Otherwise it's a (tar, member) tuple
        tar, member = source
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
        
        # Stage 5: Batch audio encoding with parallel loading
        print(f"\n[5/6] Encoding audio (batch_size={self.gpu_batch_size}, parallel loading)...")
        audio_codes = self.audio_encoder.encode_batch(
            audio_items, batch_size=self.gpu_batch_size, sr=self.sample_rate
        )
        
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
    tarball_paths: Union[str, List[str]] = None,
    audio_dir: str = None,
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
        audio_dir=audio_dir,
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
    parser.add_argument("--audio_dir", type=str, default=None,
                        help="Directory containing extracted audio files (recommended, faster)")
    parser.add_argument("--tarball", type=str, nargs='+', default=None,
                        help="Tarball(s) containing audio files (slower, use --audio_dir if available)")
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
    
    if not args.audio_dir and not args.tarball:
        parser.error("Must provide either --audio_dir or --tarball")
    
    tarball_paths = []
    if args.tarball:
        for path in args.tarball:
            tarball_paths.extend(p.strip() for p in path.split(",") if p.strip())
    
    preprocess_dataset_parallel(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        audio_dir=args.audio_dir,
        tarball_paths=tarball_paths if tarball_paths else None,
        device=args.device,
        sample_rate=args.sample_rate,
        debug=args.debug,
        phoneme_vocab_path=args.phoneme_vocab_path,
        num_cpu_workers=args.cpu_workers,
        gpu_batch_size=args.gpu_batch_size,
        io_workers=args.io_workers,
    )

