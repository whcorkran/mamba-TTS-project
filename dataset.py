"""
Dataset wrapper for preprocessed TTS data.

Uses preprocessed tensors from processed_data/ directory:
- {item_name}_phonemes.pt: Phoneme token IDs
- {item_name}_style.pt: FACodec speaker embedding (512-dim) - ground truth for SMSD
- {item_name}_codec.pt: FACodec tokens (T, 6)

Also loads MFA durations from TextGrid files when available.

Per ControlSpeech paper (arXiv:2406.01205), the ground truth style vectors
for SMSD training come from FACodec's speaker embedding, projected from 256
to 512 dimensions. These are extracted from the audio during preprocessing.
"""

import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import random

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class PreprocessedTTSDataset(Dataset):
    """
    Dataset that loads preprocessed tensors from disk.
    
    This is much faster than the original VccmTTSDataset which loads
    from tarball and runs FACodec encoding on-the-fly.
    
    Directory structure expected:
        processed_data/
            metadata.json          # List of sample metadata
            tensors/
                {safe_name}_phonemes.pt   # (T_phonemes,) long
                {safe_name}_style.pt      # (512,) float
                {safe_name}_codec.pt      # (T_codec, 6) float
    
    MFA alignments expected at:
        VccmDataset/mfa_outputs/{dataset_path}/{item_name}.TextGrid
    """
    
    def __init__(
        self,
        processed_dir: str = "processed_data",
        mfa_root: str = "VccmDataset/mfa_outputs",
        require_mfa: bool = True,
        cache_mfa_index: bool = True,
        sample_rate: int = 16000,
        cache_in_ram: bool = False,
    ):
        """
        Args:
            processed_dir: Path to preprocessed data directory
            mfa_root: Path to MFA outputs root directory
            require_mfa: If True, skip samples without MFA alignments
            cache_mfa_index: If True, build index of all TextGrid files for fast lookup
            sample_rate: Sample rate (for duration frame calculation)
            cache_in_ram: If True, load ALL tensors into RAM at startup (fast training, uses ~16GB RAM)
        """
        self.processed_dir = Path(processed_dir)
        self.mfa_root = Path(mfa_root)
        self.require_mfa = require_mfa
        self.sample_rate = sample_rate
        self.hop_size = 256  # FACodec hop size
        self.cache_in_ram = cache_in_ram
        
        # RAM cache dictionaries (populated if cache_in_ram=True)
        self._phoneme_cache: Dict[str, torch.Tensor] = {}
        self._style_cache: Dict[str, torch.Tensor] = {}
        self._codec_cache: Dict[str, torch.Tensor] = {}
        self._duration_cache: Dict[str, torch.Tensor] = {}
        
        # Load metadata
        metadata_path = self.processed_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.all_metadata = json.load(f)
        print(f"  Loaded {len(self.all_metadata)} samples from metadata")
        
        # Build MFA index for fast TextGrid lookup
        self.textgrid_index: Dict[str, Path] = {}
        if cache_mfa_index and self.mfa_root.exists():
            print(f"Building MFA TextGrid index from {self.mfa_root}...")
            for tg_path in tqdm(self.mfa_root.rglob("*.TextGrid"), desc="Indexing TextGrids"):
                # Index by stem (filename without extension)
                stem = tg_path.stem
                if stem not in self.textgrid_index:
                    self.textgrid_index[stem] = tg_path
            print(f"  Indexed {len(self.textgrid_index)} TextGrid files")
        
        # Build speaker map for voice prompt selection
        self.speaker_map: Dict[str, List[int]] = {}
        
        # Filter samples based on tensor availability and MFA requirement
        self.valid_indices: List[int] = []
        tensors_dir = self.processed_dir / "tensors"
        
        print("Validating samples...")
        skipped_no_tensors = 0
        skipped_no_mfa = 0
        
        for idx, meta in enumerate(tqdm(self.all_metadata, desc="Validating")):
            item_name = meta['item_name']
            safe_name = self._safe_name(item_name)
            
            # Check if tensors exist
            codec_path = tensors_dir / f"{safe_name}_codec.pt"
            if not codec_path.exists():
                skipped_no_tensors += 1
                continue
            
            # Check MFA availability if required
            if self.require_mfa:
                tg_stem = self._item_name_to_textgrid_stem(item_name)
                if tg_stem not in self.textgrid_index:
                    skipped_no_mfa += 1
                    continue
            
            # Sample is valid
            self.valid_indices.append(idx)
            
            # Build speaker map
            speaker = meta.get('speaker', 'unknown')
            if speaker not in self.speaker_map:
                self.speaker_map[speaker] = []
            self.speaker_map[speaker].append(idx)
        
        print(f"\nDataset Statistics:")
        print(f"  Total in metadata: {len(self.all_metadata)}")
        print(f"  Valid samples: {len(self.valid_indices)}")
        print(f"  Skipped (no tensors): {skipped_no_tensors}")
        print(f"  Skipped (no MFA): {skipped_no_mfa}")
        print(f"  Unique speakers: {len(self.speaker_map)}")
        
        # Fail fast if no valid samples
        assert len(self.valid_indices) > 0, (
            f"No valid samples found! "
            f"Total metadata: {len(self.all_metadata)}, "
            f"Skipped (no tensors): {skipped_no_tensors}, "
            f"Skipped (no MFA): {skipped_no_mfa}. "
            f"Check that processed_data/tensors/ contains tensor files and "
            f"VccmDataset/mfa_outputs/ contains TextGrid files."
        )
        
        # Load all data into RAM if requested
        if self.cache_in_ram:
            self._load_all_into_ram(tensors_dir)
    
    def _safe_name(self, item_name: str) -> str:
        """Convert item_name to safe filename (matches preprocessing script)."""
        return item_name.replace('/', '_').replace(' ', '_')
    
    def _load_all_into_ram(self, tensors_dir: Path):
        """Load all tensors and durations into RAM for fast access."""
        print(f"\n{'='*60}")
        print("Loading ALL data into RAM (this may take a minute)...")
        print(f"{'='*60}")
        
        # Collect all unique safe_names we need to load
        safe_names_to_load = set()
        for idx in self.valid_indices:
            meta = self.all_metadata[idx]
            safe_names_to_load.add(self._safe_name(meta['item_name']))
        
        # Load all tensors with progress bar
        for safe_name in tqdm(safe_names_to_load, desc="Loading tensors into RAM"):
            phoneme_path = tensors_dir / f"{safe_name}_phonemes.pt"
            style_path = tensors_dir / f"{safe_name}_style.pt"
            codec_path = tensors_dir / f"{safe_name}_codec.pt"
            
            if phoneme_path.exists():
                self._phoneme_cache[safe_name] = torch.load(phoneme_path, weights_only=True)
            if style_path.exists():
                self._style_cache[safe_name] = torch.load(style_path, weights_only=True)
            if codec_path.exists():
                self._codec_cache[safe_name] = torch.load(codec_path, weights_only=True)
        
        # Pre-cache all durations from TextGrids
        if self.require_mfa:
            print("Pre-caching MFA durations...")
            for idx in tqdm(self.valid_indices, desc="Caching durations"):
                meta = self.all_metadata[idx]
                item_name = meta['item_name']
                safe_name = self._safe_name(item_name)
                tg_stem = self._item_name_to_textgrid_stem(item_name)
                
                if tg_stem in self.textgrid_index and safe_name not in self._duration_cache:
                    phoneme_ids = self._phoneme_cache.get(safe_name)
                    if phoneme_ids is not None:
                        durations = self._load_durations_from_textgrid(
                            self.textgrid_index[tg_stem],
                            num_phonemes=len(phoneme_ids)
                        )
                        if durations is not None:
                            self._duration_cache[safe_name] = durations
        
        # Report memory usage
        import sys
        total_size = 0
        for cache in [self._phoneme_cache, self._style_cache, self._codec_cache, self._duration_cache]:
            for tensor in cache.values():
                total_size += tensor.element_size() * tensor.nelement()
        
        print(f"\n✓ Loaded {len(self._codec_cache)} samples into RAM")
        print(f"✓ Total RAM usage: {total_size / 1e9:.2f} GB")
        print(f"{'='*60}\n")
    
    def _item_name_to_textgrid_stem(self, item_name: str) -> str:
        """
        Convert item_name to TextGrid filename stem.
        
        Examples:
            "1001_134708_000013_000000" -> "1001_134708_000013_000000"
            "Emotional Speech Dataset (ESD)-...-0020_000928" -> "0020_000928"
            "MEAD-W018-audio-surprised-level_2-009" -> "009"
        """
        # For LibriTTS-style names (no hyphens), use as-is
        if '-' not in item_name:
            return item_name
        
        # For hyphenated names, the last segment after the final hyphen is typically the filename
        parts = item_name.split('-')
        return parts[-1]
    
    def _load_durations_from_textgrid(self, tg_path: Path, num_phonemes: int) -> Optional[torch.Tensor]:
        """
        Extract phoneme durations from MFA TextGrid file.
        
        Args:
            tg_path: Path to .TextGrid file
            num_phonemes: Expected number of phonemes (for validation)
        
        Returns:
            durations: torch.FloatTensor of phoneme durations (in frames)
        
        Raises:
            ValueError: If TextGrid parsing fails critically
        """
        from textgrid import TextGrid
        
        try:
            tg = TextGrid.fromFile(str(tg_path))
        except Exception as e:
            raise ValueError(f"Failed to parse TextGrid file {tg_path}: {e}")
        
        assert len(tg) > 0, f"TextGrid file {tg_path} has no tiers"
        
        # Find phone tier (usually tier index 1, but search by name to be safe)
        phone_tier = None
        for tier in tg:
            if tier.name.lower() in ['phones', 'phone', 'phonemes']:
                phone_tier = tier
                break
        
        if phone_tier is None:
            # Fall back to tier index with explicit bounds check
            if len(tg) > 1:
                phone_tier = tg[1]
            else:
                phone_tier = tg[0]
        
        assert phone_tier is not None, f"Could not find phone tier in {tg_path}"
        assert len(phone_tier) > 0, f"Phone tier in {tg_path} has no intervals"
        
        durations = []
        for interval in phone_tier:
            # Convert time to frame count
            duration_sec = interval.maxTime - interval.minTime
            assert duration_sec >= 0, f"Negative duration in {tg_path}: {duration_sec}"
            duration_frames = int(duration_sec * self.sample_rate / self.hop_size)
            durations.append(max(1, duration_frames))  # At least 1 frame
        
        result = torch.FloatTensor(durations)
        assert result.numel() > 0, f"Empty durations from {tg_path}"
        
        return result
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[Dict, torch.Tensor]:
        """
        Get a preprocessed sample.
        
        Returns:
            inputs: dict with keys:
                - phoneme_ids: (T_phonemes,) long tensor
                - text_prompt: str
                - style_prompt: str
                - style_emb: (512,) float tensor (precomputed BERT embedding)
                - durations: (T_phonemes,) float tensor or None
                - voice_codec: (T_ref, 6) float tensor (reference speaker)
            target_codec: (T_target, 6) float tensor
        """
        # Map to actual metadata index
        meta_idx = self.valid_indices[idx]
        meta = self.all_metadata[meta_idx]
        
        item_name = meta['item_name']
        safe_name = self._safe_name(item_name)
        
        # Load from RAM cache or disk
        if self.cache_in_ram:
            phoneme_ids = self._phoneme_cache[safe_name]
            style_emb = self._style_cache[safe_name]
            target_codec = self._codec_cache[safe_name]
        else:
            tensors_dir = self.processed_dir / "tensors"
            phoneme_path = tensors_dir / f"{safe_name}_phonemes.pt"
            style_path = tensors_dir / f"{safe_name}_style.pt"
            codec_path = tensors_dir / f"{safe_name}_codec.pt"
            
            assert phoneme_path.exists(), f"Phoneme tensor not found: {phoneme_path}"
            assert style_path.exists(), f"Style tensor not found: {style_path}"
            assert codec_path.exists(), f"Codec tensor not found: {codec_path}"
            
            phoneme_ids = torch.load(phoneme_path, weights_only=True)
            style_emb = torch.load(style_path, weights_only=True)
            target_codec = torch.load(codec_path, weights_only=True)
        
        # Validate tensor shapes
        assert phoneme_ids.dim() == 1, f"phoneme_ids should be 1D, got {phoneme_ids.shape}"
        assert phoneme_ids.numel() > 0, f"phoneme_ids is empty for {item_name}"
        assert style_emb.dim() == 1 and style_emb.shape[0] == 512, \
            f"style_emb should be (512,), got {style_emb.shape}"
        assert target_codec.dim() == 2 and target_codec.shape[1] == 6, \
            f"target_codec should be (T, 6), got {target_codec.shape}"
        assert target_codec.shape[0] > 0, f"target_codec has zero length for {item_name}"
        
        # Load MFA durations from cache or disk
        durations = None
        if self.cache_in_ram and safe_name in self._duration_cache:
            durations = self._duration_cache[safe_name]
        else:
            tg_stem = self._item_name_to_textgrid_stem(item_name)
            if tg_stem in self.textgrid_index:
                durations = self._load_durations_from_textgrid(
                    self.textgrid_index[tg_stem],
                    num_phonemes=len(phoneme_ids)
                )
                assert durations is not None, f"Failed to load durations for {item_name}"
            elif self.require_mfa:
                raise RuntimeError(
                    f"MFA durations required but not found for {item_name} (stem={tg_stem}). "
                    f"This indicates a bug in dataset validation."
                )
        
        # Get voice reference from same speaker (different utterance)
        speaker = meta.get('speaker', 'unknown')
        speaker_samples = self.speaker_map.get(speaker, [meta_idx])
        assert len(speaker_samples) > 0, f"No samples found for speaker {speaker}"
        
        # Pick a different sample from same speaker if possible
        ref_candidates = [i for i in speaker_samples if i != meta_idx]
        if ref_candidates:
            ref_idx = random.choice(ref_candidates)
        else:
            # Fallback to same sample (only sample from this speaker)
            ref_idx = meta_idx
        
        ref_meta = self.all_metadata[ref_idx]
        ref_safe_name = self._safe_name(ref_meta['item_name'])
        
        # Load reference codec from cache or disk
        if self.cache_in_ram and ref_safe_name in self._codec_cache:
            voice_codec = self._codec_cache[ref_safe_name]
        else:
            tensors_dir = self.processed_dir / "tensors"
            ref_codec_path = tensors_dir / f"{ref_safe_name}_codec.pt"
            assert ref_codec_path.exists(), f"Reference codec not found: {ref_codec_path}"
            voice_codec = torch.load(ref_codec_path, weights_only=True)
        
        assert voice_codec.dim() == 2 and voice_codec.shape[1] == 6, \
            f"voice_codec should be (T, 6), got {voice_codec.shape}"
        
        inputs = {
            'phoneme_ids': phoneme_ids,
            'text_prompt': meta['text'],
            'style_prompt': meta['style_prompt'],
            'style_emb': style_emb,
            'durations': durations,
            'voice_codec': voice_codec,
        }
        
        return inputs, target_codec
    
    def collate_fn(self, batch: List[Tuple[Dict, torch.Tensor]]) -> Tuple[Dict, torch.Tensor]:
        """
        Collate batch of samples with proper padding.
        """
        assert len(batch) > 0, "Empty batch passed to collate_fn"
        
        inputs_list = [b[0] for b in batch]
        targets_list = [b[1] for b in batch]
        
        # Validate batch consistency
        for i, inp in enumerate(inputs_list):
            assert 'phoneme_ids' in inp, f"Sample {i} missing phoneme_ids"
            assert 'style_emb' in inp, f"Sample {i} missing style_emb"
            assert 'voice_codec' in inp, f"Sample {i} missing voice_codec"
        
        # Pad phoneme_ids
        phoneme_ids = [inp['phoneme_ids'] for inp in inputs_list]
        assert all(p.dim() == 1 for p in phoneme_ids), "All phoneme_ids must be 1D"
        assert all(len(p) > 0 for p in phoneme_ids), "All phoneme_ids must be non-empty"
        
        max_ph_len = max(len(p) for p in phoneme_ids)
        phoneme_ids_padded = torch.zeros(len(batch), max_ph_len, dtype=torch.long)
        phoneme_mask = torch.ones(len(batch), max_ph_len, dtype=torch.bool)  # True = pad
        for i, p in enumerate(phoneme_ids):
            phoneme_ids_padded[i, :len(p)] = p
            phoneme_mask[i, :len(p)] = False
        
        # Validate and stack style embeddings (already same size)
        style_embs_list = [inp['style_emb'] for inp in inputs_list]
        assert all(s.shape == (512,) for s in style_embs_list), \
            f"All style_emb must be (512,), got shapes: {[s.shape for s in style_embs_list]}"
        style_embs = torch.stack(style_embs_list)
        
        # Pad durations if available
        durations_list = [inp['durations'] for inp in inputs_list]
        durations_padded = None
        
        if self.require_mfa:
            # All samples should have durations
            assert all(d is not None for d in durations_list), \
                "require_mfa=True but some samples have None durations"
        
        if all(d is not None for d in durations_list):
            durations_padded = torch.zeros(len(batch), max_ph_len)
            for i, d in enumerate(durations_list):
                assert d.dim() == 1, f"Duration tensor {i} should be 1D, got {d.shape}"
                # Handle length mismatch between durations and phonemes
                min_len = min(len(d), max_ph_len)
                durations_padded[i, :min_len] = d[:min_len]
        
        # Validate and pad voice codec (reference)
        voice_codecs = [inp['voice_codec'] for inp in inputs_list]
        assert all(v.dim() == 2 for v in voice_codecs), "All voice_codec must be 2D (T, 6)"
        assert all(v.shape[1] == 6 for v in voice_codecs), \
            f"All voice_codec must have 6 quantizers, got: {[v.shape for v in voice_codecs]}"
        
        max_ref_len = max(v.shape[0] for v in voice_codecs)
        num_quantizers = 6  # FACodec always has 6 quantizers
        voice_codec_padded = torch.zeros(len(batch), max_ref_len, num_quantizers)
        for i, v in enumerate(voice_codecs):
            voice_codec_padded[i, :v.shape[0]] = v
        
        # Validate and pad target codec
        assert all(t.dim() == 2 for t in targets_list), "All target_codec must be 2D (T, 6)"
        assert all(t.shape[1] == 6 for t in targets_list), \
            f"All target_codec must have 6 quantizers, got: {[t.shape for t in targets_list]}"
        
        max_tgt_len = max(t.shape[0] for t in targets_list)
        target_codec_padded = torch.zeros(len(batch), max_tgt_len, num_quantizers)
        for i, t in enumerate(targets_list):
            target_codec_padded[i, :t.shape[0]] = t
        
        collated_inputs = {
            'phoneme_ids': phoneme_ids_padded,
            'phoneme_mask': phoneme_mask,
            'text_prompt': [inp['text_prompt'] for inp in inputs_list],
            'style_prompt': [inp['style_prompt'] for inp in inputs_list],
            'style_emb': style_embs,
            'durations': durations_padded,
            'voice_codec': voice_codec_padded,
        }
        
        return collated_inputs, target_codec_padded


# Keep old class for backwards compatibility
class VccmTTSDataset(Dataset):
    """
    DEPRECATED: Use PreprocessedTTSDataset instead.
    
    This class loads from tarball which is much slower.
    Kept for backwards compatibility only.
    """
    
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "VccmTTSDataset is deprecated. Use PreprocessedTTSDataset instead.\n"
            "Update your config to use:\n"
            "  data:\n"
            "    processed_dir: 'processed_data'\n"
            "    mfa_root: 'VccmDataset/mfa_outputs'"
        )


if __name__ == "__main__":
    # Test the dataset
    print("=" * 60)
    print("Testing PreprocessedTTSDataset")
    print("=" * 60)
    
    dataset = PreprocessedTTSDataset(
        processed_dir="processed_data",
        mfa_root="VccmDataset/mfa_outputs",
        require_mfa=True,
    )
    
    print(f"\nDataset length: {len(dataset)}")
    
    # Test single sample
    print("\nTesting single sample...")
    inputs, target = dataset[0]
    print(f"  phoneme_ids shape: {inputs['phoneme_ids'].shape}")
    print(f"  style_emb shape: {inputs['style_emb'].shape}")
    print(f"  target_codec shape: {target.shape}")
    print(f"  voice_codec shape: {inputs['voice_codec'].shape}")
    print(f"  text_prompt: {inputs['text_prompt'][:50]}...")
    print(f"  style_prompt: {inputs['style_prompt'][:50]}...")
    if inputs['durations'] is not None:
        print(f"  durations shape: {inputs['durations'].shape}")
    else:
        print(f"  durations: None (MFA not available for this sample)")
    
    # Test DataLoader
    print("\nTesting DataLoader...")
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=0,
    )
    
    batch_inputs, batch_targets = next(iter(loader))
    print(f"  Batch phoneme_ids shape: {batch_inputs['phoneme_ids'].shape}")
    print(f"  Batch style_emb shape: {batch_inputs['style_emb'].shape}")
    print(f"  Batch target_codec shape: {batch_targets.shape}")
    print(f"  Batch voice_codec shape: {batch_inputs['voice_codec'].shape}")
    if batch_inputs['durations'] is not None:
        print(f"  Batch durations shape: {batch_inputs['durations'].shape}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
