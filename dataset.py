"""
Dataset wrapper for VccmDataset/controlspeech_train.csv
Retrieves text prompts, style prompts, and audio samples for TTS training.
"""

import io
from pathlib import Path
import csv
import torch
import random
from torch.utils.data import Dataset
import torchaudio
import tarfile


class VccmTTSDataset(Dataset):
    """
    Dataset wrapper for ControlSpeech VccmDataset.

    Loads data from controlspeech_train.csv and retrieves:
    - Text prompt (text to synthesize)
    - Style prompt (natural language style description)
    - Audio file path and waveform
    - Metadata (emotion, speaker, pitch, duration, energy)

    CSV columns:
    - item_name: unique identifier
    - dur: duration category (low/normal/high)
    - pitch: pitch category (low/normal/high)
    - energy: energy category (low/normal/high)
    - gender: M/F
    - emotion: emotion label (happy, sad, angry, neutral, etc.)
    - spk: speaker ID
    - txt: text to synthesize
    - style_prompt: natural language description of speaking style
    """

    def __init__(
        self,
        csv_path: str = "VccmDataset/controlspeech_train.csv",
        audio_root: str = "TextrolSpeech_data.tar.gz",
        sample_rate: int = 16000,
    ):
        """
        Initialize dataset.

        Args:
            csv_path: Path to CSV file
            audio_root: Root directory for audio files
            sample_rate: Target sample rate for audio (will resample if needed)
            limit: Optional limit on number of samples to load
            filter_emotion: Optional list of emotions to include (e.g., ['happy', 'sad'])
            filter_speaker: Optional list of speaker IDs to include
        """
        self.csv_path = csv_path
        self.audio_root = audio_root
        self.sample_rate = sample_rate
        self.tar_audio = tarfile.open(self.audio_root, "r:gz")
        self.csv_file = open(self.csv_path, "r", encoding="utf-8")
        self.csv_cursor = list(csv.DictReader(self.csv_file))
        
        self.speaker_map = {}
        for row in self.csv_cursor:
            spk = row['spk']
            self.speaker_map.setdefault(spk, []).append(row['item_name'])

        self.audio_cursor = {m.name : m for m in self.tar_audio.getmembers() if m.isfile() and m.name.endswith(".wav")}

    def _wav_to_tensor(self, item_name):
        audio_path = Path(item_name.replace("-", "/")).with_suffix(".wav")
        audio_path = self.audio_cursor[str(audio_path)]
        wav = self.tar_audio.extractfile(audio_path)
        wav = io.BytesIO(wav.read())
 
        waveform, sr = torchaudio.load(wav)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            sr = self.sample_rate
        return waveform.mean(dim=0, keepdim=True)

    def __len__(self) -> int:
        return len(self.audio_cursor)

    def __getitem__(self, idx: int):
        sample = self.csv_cursor[idx]
        item_name = sample['item_name']
        speaker_examples = self.speaker_map[sample['spk']]
        speaker_name = random.choice(list(filter(lambda x: x != item_name, speaker_examples)))

        speaker_waveform = self._wav_to_tensor(speaker_name)
        target_waveform = self._wav_to_tensor(item_name)

        # Load MFA durations (REQUIRED for proper ControlSpeech training)
        # CHECK SETUP_AND_RUN.txt FOR MORE INFORMATION
        durations = None
        tg_path = Path(self.audio_root).parent / 'mfa_outputs' / f'{item_name}.TextGrid'
        if tg_path.exists():
            durations = self._load_durations_from_textgrid(tg_path, sample['txt'])
        
        return {
            'voice_waveform': speaker_waveform,
            'text_prompt': sample['txt'],
            'style_prompt': sample['style_prompt'],
            'durations': durations,  # None if MFA not available (will use fallback)
        }, target_waveform
    
    def _load_durations_from_textgrid(self, tg_path, text):
        """
        Extract phoneme durations from MFA TextGrid file.
        
        Args:
            tg_path: Path to .TextGrid file
            text: Original text (for validation)
        
        Returns:
            durations: torch.FloatTensor of phoneme durations (in frames)
        """
        try:
            from textgrid import TextGrid
            
            tg = TextGrid.fromFile(str(tg_path))
            phone_tier = tg[1]  # Usually tier 1 contains phone alignments
            
            durations = []
            hop_size = 256  # FACodec hop size
            sample_rate = self.sample_rate
            
            for interval in phone_tier:
                # Convert time to frame count
                duration_sec = interval.maxTime - interval.minTime
                duration_frames = int(duration_sec * sample_rate / hop_size)
                durations.append(max(1, duration_frames))  # At least 1 frame
            
            return torch.FloatTensor(durations)
        except Exception as e:
            print(f"Warning: Failed to load durations from {tg_path}: {e}")
            return None

    def collate_fn(self, batch):
        voices = [item[0]['voice_waveform'] for item in batch]
        texts = [item[0]['text_prompt'] for item in batch]
        styles = [item[0]['style_prompt'] for item in batch]
        targets = [item[1] for item in batch]
        
        # Collate durations if available from MFA preprocessing
        durations = [item[0].get('durations') for item in batch]
        durations_padded = None
        
        if all(d is not None for d in durations):
            # Pad durations to same length
            max_len = max(len(d) for d in durations)
            durations_padded = torch.zeros(len(batch), max_len)
            for i, d in enumerate(durations):
                durations_padded[i, :len(d)] = d
        
        result = {
            'voice_waveform': torch.stack(voices),
            'text_prompt': texts,
            'style_prompt': styles,
        }
        
        if durations_padded is not None:
            result['durations'] = durations_padded
        
        return result, torch.stack(targets)

if __name__ == "__main__":
    # Test the dataset with proper validation
    import torch
    from torch.utils.data import DataLoader

    # Use default paths from the class
    csv_path = "VccmDataset/controlspeech_train.csv"
    audio_root = "VccmDataset/TextrolSpeech_data.tar.gz"

    print("=" * 60)
    print("Testing VccmTTSDataset")
    print("=" * 60)
    
    # Test 1: Dataset initialization
    print("\n1. Initializing dataset...")
    try:
        dataset = VccmTTSDataset(
            csv_path=csv_path,
            audio_root=audio_root,
            sample_rate=16000
        )
        print("   ✓ Dataset initialized successfully")
        print(f"   Dataset length: {len(dataset)}")
    except Exception as e:
        print(f"   ✗ Failed to initialize dataset: {e}")
        raise

    # Test 2: Get a single sample
    print("\n2. Testing __getitem__...")
    try:
        sample = dataset[0]
        inputs, target = sample
        
        print("   ✓ Successfully retrieved sample")
        print(f"   Input keys: {list(inputs.keys())}")
        print(f"   Voice waveform shape: {inputs['voice_waveform'].shape}")
        print(f"   Target waveform shape: {target.shape}")
        print(f"   Text prompt (first 100 chars): {inputs['text_prompt'][:100]}...")
        print(f"   Style prompt (first 100 chars): {inputs['style_prompt'][:100]}...")
        print(f"   Voice waveform dtype: {inputs['voice_waveform'].dtype}")
        print(f"   Target waveform dtype: {target.dtype}")
    except Exception as e:
        print(f"   ✗ Failed to get sample: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Test 3: Test multiple samples
    print("\n3. Testing multiple samples...")
    try:
        num_test_samples = min(5, len(dataset))
        for i in range(num_test_samples):
            sample = dataset[i]
            inputs, target = sample
            assert inputs['voice_waveform'].shape[0] == 1, f"Expected channel dim=1, got {inputs['voice_waveform'].shape[0]}"
            assert target.shape[0] == 1, f"Expected channel dim=1, got {target.shape[0]}"
            assert isinstance(inputs['text_prompt'], str), "Text prompt should be a string"
            assert isinstance(inputs['style_prompt'], str), "Style prompt should be a string"
        print(f"   ✓ Successfully tested {num_test_samples} samples")
    except Exception as e:
        print(f"   ✗ Failed to test multiple samples: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Test 4: Test DataLoader
    print("\n4. Testing DataLoader...")
    try:
        dataloader = DataLoader(
            dataset, 
            batch_size=2, 
            shuffle=False,
            num_workers=0  # Set to 0 to avoid multiprocessing issues during testing
        )
        
        batch = next(iter(dataloader))
        inputs_batch, target_batch = batch
        
        print("   ✓ DataLoader working correctly")
        print(f"   Batch size: {len(inputs_batch['text_prompt'])}")
        print(f"   Batched voice waveform shape: {inputs_batch['voice_waveform'].shape}")
        print(f"   Batched target waveform shape: {target_batch.shape}")
    except Exception as e:
        print(f"   ✗ Failed to test DataLoader: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Test 5: Verify audio properties
    print("\n5. Verifying audio properties...")
    try:
        sample = dataset[0]
        inputs, target = sample
        voice_wav = inputs['voice_waveform']
        target_wav = target
        
        # Check that waveforms are reasonable (not all zeros, finite values)
        assert torch.any(voice_wav != 0), "Voice waveform should not be all zeros"
        assert torch.any(target_wav != 0), "Target waveform should not be all zeros"
        assert torch.all(torch.isfinite(voice_wav)), "Voice waveform should contain finite values"
        assert torch.all(torch.isfinite(target_wav)), "Target waveform should contain finite values"
        
        print("   ✓ Audio properties verified")
        print(f"   Voice waveform range: [{voice_wav.min():.4f}, {voice_wav.max():.4f}]")
        print(f"   Target waveform range: [{target_wav.min():.4f}, {target_wav.max():.4f}]")
    except Exception as e:
        print(f"   ✗ Audio verification failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
