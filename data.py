"""
Base dataset wrapper class for TTS model training.
Provides text prompts, style prompts, and speaker prompts as samples,
with synthesized speech as labels.
"""
import os
import csv
import torch
import torchaudio
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union

from preprocess import text_to_phoneme, TxtProcessor
from pretrained_encoders import StyleEncode, AudioFACodecEncoder


class BaseDataset(Dataset):
    """
    Base dataset wrapper for TTS training.
    
    Samples:
        - text_prompt: Text to synthesize (encoded as phonemes)
        - style_prompt: Style description text (encoded as BERT embeddings)
        - speaker_prompt: Speaker identifier
    
    Labels:
        - speech: Synthesized speech audio (encoded as FACodec codes or raw audio)
    """
    
    def __init__(
        self,
        csv_path: str,
        audio_dir: Optional[str] = None,
        text_processor: Optional[TxtProcessor] = None,
        style_encoder: Optional[StyleEncode] = None,
        audio_encoder: Optional[AudioFACodecEncoder] = None,
        speaker_map: Optional[Dict[str, int]] = None,
        sample_rate: int = 16000,  # FACodec requires 16 kHz
        use_audio_codes: bool = True,
        use_cached_encodings: bool = True,
        cache_dir: Optional[str] = None,
        max_audio_length: Optional[int] = None,
        max_text_length: Optional[int] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            csv_path: Path to CSV file with columns: item_name, txt, style_prompt, spk
            audio_dir: Directory containing audio files (if None, inferred from item_name)
            text_processor: Text processor for encoding text to phonemes
            style_encoder: Style encoder for encoding style prompts
            audio_encoder: Audio encoder for encoding speech (if None, returns raw audio)
            speaker_map: Mapping from speaker names to IDs (if None, built from data)
            sample_rate: Target audio sample rate (default: 16000 Hz for FACodec)
            use_audio_codes: If True, return FACodec codes; if False, return raw audio
            use_cached_encodings: If True, cache encoded features to disk
            cache_dir: Directory for caching encoded features
            max_audio_length: Maximum audio length in samples (None = no limit)
            max_text_length: Maximum text length in phonemes (None = no limit)
        """
        self.csv_path = csv_path
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.use_audio_codes = use_audio_codes
        self.use_cached_encodings = use_cached_encodings
        self.cache_dir = cache_dir
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length
        
        # Initialize processors
        self.text_processor = text_processor if text_processor is not None else TxtProcessor()
        self.style_encoder = style_encoder
        self.audio_encoder = audio_encoder
        
        # Validate sample rate for FACodec
        if self.use_audio_codes and self.audio_encoder is not None and self.sample_rate != 16000:
            import warnings
            warnings.warn(
                f"FACodec requires 16 kHz sample rate, but got {self.sample_rate} Hz. "
                "Audio will be resampled, but this may affect quality."
            )
        
        # Load data from CSV
        self.data = self._load_csv()
        
        # Build speaker map if not provided
        if speaker_map is None:
            self.speaker_map = self._build_speaker_map()
        else:
            self.speaker_map = speaker_map
        
        # Setup cache directory
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            os.makedirs(os.path.join(self.cache_dir, 'text'), exist_ok=True)
            os.makedirs(os.path.join(self.cache_dir, 'style'), exist_ok=True)
            os.makedirs(os.path.join(self.cache_dir, 'audio'), exist_ok=True)
    
    def _load_csv(self) -> List[Dict[str, str]]:
        """Load data from CSV file."""
        data = []
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Ensure required columns exist
                required_cols = ['item_name', 'txt']
                if not all(col in row for col in required_cols):
                    continue
                
                data.append({
                    'item_name': row['item_name'],
                    'txt': row['txt'],
                    'style_prompt': row.get('style_prompt', ''),
                    'spk': row.get('spk', ''),
                })
        return data
    
    def _build_speaker_map(self) -> Dict[str, int]:
        """Build speaker ID mapping from data."""
        speakers = set()
        for item in self.data:
            if item['spk']:
                speakers.add(item['spk'])
        
        speaker_map = {spk: idx for idx, spk in enumerate(sorted(speakers))}
        return speaker_map
    
    def _get_audio_path(self, item_name: str) -> str:
        """Get path to audio file for given item_name."""
        if self.audio_dir is not None:
            # Try common audio extensions
            for ext in ['.wav', '.flac', '.mp3']:
                audio_path = os.path.join(self.audio_dir, f"{item_name}{ext}")
                if os.path.exists(audio_path):
                    return audio_path
            # If not found, return default .wav path
            return os.path.join(self.audio_dir, f"{item_name}.wav")
        else:
            # Assume audio is in same directory as CSV or item_name contains path
            return item_name
    
    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Normalize to [-1, 1] range (assuming audio is already in this range)
        # If needed, can add normalization here
        
        # Truncate if max_audio_length is set
        if self.max_audio_length is not None and waveform.shape[1] > self.max_audio_length:
            waveform = waveform[:, :self.max_audio_length]
        
        return waveform  # Shape: (1, T)
    
    def _encode_text(self, text: str, item_name: str) -> Dict[str, torch.Tensor]:
        """Encode text to phonemes."""
        cache_path = None
        if self.use_cached_encodings and self.cache_dir is not None:
            cache_path = os.path.join(self.cache_dir, 'text', f"{item_name}.pt")
            if os.path.exists(cache_path):
                return torch.load(cache_path)
        
        # Process text to phonemes
        result = text_to_phoneme(text)
        
        # Convert to tensors
        # Split phonemes
        phonemes = result['ph'].split() if result['ph'] else []
        
        # Create phoneme token IDs (simple mapping for now)
        # In practice, you'd use a proper phoneme vocabulary
        phoneme_to_id = {ph: idx + 1 for idx, ph in enumerate(set(phonemes))}
        phoneme_ids = [phoneme_to_id.get(ph, 0) for ph in phonemes]
        
        # Truncate if max_text_length is set
        if self.max_text_length is not None and len(phoneme_ids) > self.max_text_length:
            phoneme_ids = phoneme_ids[:self.max_text_length]
        
        encoded = {
            'phoneme_ids': torch.LongTensor(phoneme_ids),
            'phoneme_text': result['ph'],
            'word_text': result['word'],
            'ph2word': torch.LongTensor(result['ph2word'][:len(phoneme_ids)]),
        }
        
        # Cache if enabled
        if cache_path is not None:
            torch.save(encoded, cache_path)
        
        return encoded
    
    def _encode_style(self, style_prompt: str, item_name: str) -> torch.Tensor:
        """Encode style prompt using BERT."""
        if self.style_encoder is None:
            # Return zero vector if no style encoder
            return torch.zeros(768)
        
        cache_path = None
        if self.use_cached_encodings and self.cache_dir is not None:
            cache_path = os.path.join(self.cache_dir, 'style', f"{item_name}.pt")
            if os.path.exists(cache_path):
                return torch.load(cache_path)
        
        # Encode style
        with torch.no_grad():
            style_embed = self.style_encoder([style_prompt])
            style_embed = style_embed.squeeze(0)  # Remove batch dimension
        
        # Cache if enabled
        if cache_path is not None:
            torch.save(style_embed, cache_path)
        
        return style_embed
    
    def _encode_audio(self, waveform: torch.Tensor, item_name: str) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Encode audio using FACodec or return raw audio."""
        if not self.use_audio_codes or self.audio_encoder is None:
            return waveform
        
        cache_path = None
        if self.use_cached_encodings and self.cache_dir is not None:
            cache_path = os.path.join(self.cache_dir, 'audio', f"{item_name}.pt")
            if os.path.exists(cache_path):
                return torch.load(cache_path)
        
        # Encode audio
        with torch.no_grad():
            # Audio encoder expects (B, 1, T) or (B, T)
            waveform_batch = waveform.unsqueeze(0)  # (1, 1, T)
            encoded = self.audio_encoder(waveform_batch)
            
            # encoded is (codec, spk_embs) where codec is (T, B, C)
            codec, spk_embs = encoded
            codec = codec.squeeze(1)  # Remove batch dimension: (T, C)
            spk_embs = spk_embs.squeeze(0)  # Remove batch dimension
        
        result = (codec, spk_embs)
        
        # Cache if enabled
        if cache_path is not None:
            torch.save(result, cache_path)
        
        return result
    
    def _get_speaker_id(self, speaker: str) -> torch.Tensor:
        """Get speaker ID tensor."""
        if not speaker or speaker not in self.speaker_map:
            return torch.LongTensor([0])  # Default to speaker 0
        return torch.LongTensor([self.speaker_map[speaker]])
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Returns:
            Dictionary containing:
                - 'text_prompt': Encoded text (phoneme IDs)
                - 'style_prompt': Encoded style (BERT embedding)
                - 'speaker_prompt': Speaker ID
                - 'speech': Encoded or raw audio
                - 'item_name': Item identifier
                - Additional metadata (phoneme_text, word_text, etc.)
        """
        item = self.data[idx]
        item_name = item['item_name']
        
        # Encode text prompt
        text_encoded = self._encode_text(item['txt'], item_name)
        
        # Encode style prompt
        style_encoded = self._encode_style(item['style_prompt'], item_name)
        
        # Get speaker ID
        speaker_id = self._get_speaker_id(item['spk'])
        
        # Load and encode audio
        audio_path = self._get_audio_path(item_name)
        waveform = self._load_audio(audio_path)
        audio_encoded = self._encode_audio(waveform, item_name)
        
        # Build sample dictionary
        sample = {
            'text_prompt': text_encoded['phoneme_ids'],
            'style_prompt': style_encoded,
            'speaker_prompt': speaker_id,
            'item_name': item_name,
            'phoneme_text': text_encoded['phoneme_text'],
            'word_text': text_encoded['word_text'],
            'ph2word': text_encoded['ph2word'],
        }
        
        # Add audio (either codes or raw waveform)
        if isinstance(audio_encoded, tuple):
            # FACodec encoding: (codec, spk_embs)
            sample['speech'] = audio_encoded[0]  # Codec codes
            sample['speaker_emb'] = audio_encoded[1]  # Speaker embeddings from audio
        else:
            # Raw audio
            sample['speech'] = audio_encoded
        
        return sample


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    Pads sequences to same length within batch.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched dictionary with padded tensors
    """
    if len(batch) == 0:
        return {}
    
    # Get batch size
    batch_size = len(batch)
    
    # Extract all fields
    item_names = [item['item_name'] for item in batch]
    
    # Text prompts (variable length sequences)
    text_prompts = [item['text_prompt'] for item in batch]
    text_lengths = torch.LongTensor([len(tp) for tp in text_prompts])
    max_text_len = text_lengths.max().item()
    text_padded = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    for i, tp in enumerate(text_prompts):
        text_padded[i, :len(tp)] = tp
    
    # Style prompts (fixed size embeddings)
    style_prompts = torch.stack([item['style_prompt'] for item in batch])
    
    # Speaker prompts (single IDs)
    speaker_prompts = torch.stack([item['speaker_prompt'] for item in batch]).squeeze(-1)
    
    # Speech (variable length)
    speech_items = [item['speech'] for item in batch]
    
    # Check if speech is codec (2D: T, C) or raw audio (1D or 2D: 1, T)
    if speech_items[0].dim() == 2 and speech_items[0].shape[1] > 1:
        # Codec format: (T, C)
        speech_lengths = torch.LongTensor([s.shape[0] for s in speech_items])
        max_speech_len = speech_lengths.max().item()
        codec_dim = speech_items[0].shape[1]
        speech_padded = torch.zeros(batch_size, max_speech_len, codec_dim, dtype=speech_items[0].dtype)
        for i, s in enumerate(speech_items):
            speech_padded[i, :s.shape[0], :] = s
    else:
        # Raw audio format: (1, T) or (T,)
        speech_lengths = torch.LongTensor([s.shape[-1] if s.dim() > 1 else s.shape[0] for s in speech_items])
        max_speech_len = speech_lengths.max().item()
        if speech_items[0].dim() == 2:
            speech_padded = torch.zeros(batch_size, 1, max_speech_len, dtype=speech_items[0].dtype)
            for i, s in enumerate(speech_items):
                speech_padded[i, :, :s.shape[-1]] = s
        else:
            speech_padded = torch.zeros(batch_size, max_speech_len, dtype=speech_items[0].dtype)
            for i, s in enumerate(speech_items):
                speech_padded[i, :s.shape[0]] = s
    
    # ph2word (variable length, same as text)
    ph2word_items = [item['ph2word'] for item in batch]
    ph2word_padded = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    for i, pw in enumerate(ph2word_items):
        ph2word_padded[i, :len(pw)] = pw
    
    # Build batched dictionary
    batched = {
        'text_prompt': text_padded,
        'text_length': text_lengths,
        'style_prompt': style_prompts,
        'speaker_prompt': speaker_prompts,
        'speech': speech_padded,
        'speech_length': speech_lengths,
        'ph2word': ph2word_padded,
        'item_name': item_names,
    }
    
    # Add speaker embeddings if present
    if 'speaker_emb' in batch[0]:
        speaker_embs = torch.stack([item['speaker_emb'] for item in batch])
        batched['speaker_emb'] = speaker_embs
    
    return batched
