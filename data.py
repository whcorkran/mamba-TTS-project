"""
Dataset wrapper for VccmDataset/controlspeech_train.csv
Retrieves text prompts, style prompts, and audio samples for TTS training.
"""

import os
import csv
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
import torchaudio


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
        audio_root: str = "VccmDataset/audio",
        sample_rate: int = 16000,
        limit: Optional[int] = None,
        filter_emotion: Optional[List[str]] = None,
        filter_speaker: Optional[List[str]] = None,
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
        self.file = open(self.csv_path, "r", encoding="utf-8")
        self.cursor = csv.DictReader(self.file)
        self.data = []

    def get(self, num_rows=10000):
        # Load CSV data in batch
        results = []
        for _ in range(num_rows):
            try:
                row = next(self.cursor)
                results.append(row)
            except StopIteration:
                break
        self.data = results
        return results

        print(f"Loaded {len(self.data)} samples from {self.csv_path}")

    def __len__(self) -> int:
        return len(self.data)

    def _parse_audio_path(self, item_name: str) -> str:
        """
        Parse item_name to construct audio file path.

        Handles different dataset formats:
        - ESD: "Emotional Speech Dataset (ESD)-Emotion Speech Dataset-0020-Happy-0020_000928"
        - MEAD: "MEAD-M007-audio-surprised-level_1-017"
        - SAVEE: "SAVEE-ALL-JK_f09"
        - TESS: similar patterns
        - RAVDESS: similar patterns

        Args:
            item_name: Unique identifier from CSV

        Returns:
            Full path to audio file
        """
        parts = item_name.split("-")

        # Emotional Speech Dataset (ESD)
        if item_name.startswith("Emotional Speech Dataset (ESD)"):
            # Format: "Emotional Speech Dataset (ESD)-Emotion Speech Dataset-0020-Happy-0020_000928"
            speaker_id = parts[2]  # "0020"
            emotion = parts[3]  # "Happy"
            file_id = parts[4]  # "0020_000928"
            return os.path.join(
                self.audio_root,
                "Emotional Speech Dataset (ESD)",
                "Emotion Speech Dataset",
                speaker_id,
                emotion,
                f"{file_id}.wav",
            )

        # MEAD dataset
        elif item_name.startswith("MEAD"):
            # Format: "MEAD-M007-audio-surprised-level_1-017"
            speaker_id = parts[1]  # "M007"
            emotion = parts[3]  # "surprised"
            level = parts[4]  # "level_1"
            file_id = parts[5]  # "017"
            return os.path.join(
                self.audio_root,
                "MEAD",
                speaker_id,
                "audio",
                emotion,
                level,
                f"{file_id}.wav",
            )

        # SAVEE dataset
        elif item_name.startswith("SAVEE"):
            # Format: "SAVEE-ALL-JK_f09"
            subset = parts[1]  # "ALL"
            file_id = parts[2]  # "JK_f09"
            return os.path.join(self.audio_root, "SAVEE", subset, f"{file_id}.wav")

        # TESS dataset
        elif item_name.startswith("TESS"):
            # Format might vary, handle generically
            # Example: "TESS-OAF-say-angry"
            # Reconstruct path based on parts
            file_name = "-".join(parts[1:]) + ".wav"
            return os.path.join(self.audio_root, "TESS", file_name)

        # RAVDESS dataset
        elif item_name.startswith("RAVDESS"):
            # Format might vary, handle generically
            file_name = "-".join(parts[1:]) + ".wav"
            return os.path.join(self.audio_root, "RAVDESS", file_name)

        # MESS compressed dataset
        elif item_name.startswith("MESS"):
            # Format might vary, handle generically
            file_name = "-".join(parts[1:]) + ".wav"
            return os.path.join(self.audio_root, "MESS compressed", file_name)

        else:
            raise ValueError(f"Unknown dataset format for item_name: {item_name}")

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
            - text: Text to synthesize (str)
            - style_prompt: Style description (str)
            - audio_path: Path to audio file (str)
            - waveform: Audio waveform tensor (1, T)
            - sample_rate: Audio sample rate (int)
            - emotion: Emotion label (str)
            - speaker: Speaker ID (str)
            - gender: Gender (str)
            - duration: Duration category (str)
            - pitch: Pitch category (str)
            - energy: Energy category (str)
            - item_name: Unique identifier (str)
        """
        row = self.data[idx]

        # Parse audio path
        audio_path = self._parse_audio_path(row["item_name"])

        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            sr = self.sample_rate

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return {
            "text": row["txt"],
            "style_prompt": row["style_prompt"],
            "audio_path": audio_path,
            "waveform": waveform,  # (1, T)
            "sample_rate": sr,
            "emotion": row["emotion"],
            "speaker": row["spk"],
            "gender": row["gender"],
            "duration": row["dur"],
            "pitch": row["pitch"],
            "energy": row["energy"],
            "item_name": row["item_name"],
        }

    def get_emotion_counts(self) -> Dict[str, int]:
        """Get count of samples per emotion."""
        counts = {}
        for row in self.data:
            emotion = row["emotion"]
            counts[emotion] = counts.get(emotion, 0) + 1
        return counts

    def get_speaker_counts(self) -> Dict[str, int]:
        """Get count of samples per speaker."""
        counts = {}
        for row in self.data:
            speaker = row["spk"]
            counts[speaker] = counts.get(speaker, 0) + 1
        return counts


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for DataLoader.

    Pads waveforms to the same length and stacks tensors.
    Text and style prompts are kept as lists (to be encoded later).

    Args:
        batch: List of samples from __getitem__

    Returns:
        Batched dictionary with:
        - text: List of text strings (B,)
        - style_prompt: List of style strings (B,)
        - waveform: Padded waveform tensor (B, 1, max_T)
        - waveform_lengths: Original lengths (B,)
        - audio_path: List of audio paths (B,)
        - emotion: List of emotions (B,)
        - speaker: List of speaker IDs (B,)
        - ... other metadata fields
    """
    # Find max waveform length
    max_len = max(item["waveform"].shape[1] for item in batch)

    # Pad waveforms
    waveforms = []
    waveform_lengths = []
    for item in batch:
        waveform = item["waveform"]
        length = waveform.shape[1]
        waveform_lengths.append(length)

        # Pad to max length
        if length < max_len:
            padding = torch.zeros(1, max_len - length)
            waveform = torch.cat([waveform, padding], dim=1)

        waveforms.append(waveform)

    # Stack waveforms
    waveforms = torch.stack(waveforms, dim=0)  # (B, 1, max_T)
    waveform_lengths = torch.tensor(waveform_lengths)  # (B,)

    return {
        "text": [item["text"] for item in batch],
        "style_prompt": [item["style_prompt"] for item in batch],
        "waveform": waveforms,
        "waveform_lengths": waveform_lengths,
        "audio_path": [item["audio_path"] for item in batch],
        "sample_rate": batch[0]["sample_rate"],  # Assuming all same
        "emotion": [item["emotion"] for item in batch],
        "speaker": [item["speaker"] for item in batch],
        "gender": [item["gender"] for item in batch],
        "duration": [item["duration"] for item in batch],
        "pitch": [item["pitch"] for item in batch],
        "energy": [item["energy"] for item in batch],
        "item_name": [item["item_name"] for item in batch],
    }


if __name__ == "__main__":
    # Example usage
    print("Loading VccmTTSDataset...")
    dataset = VccmTTSDataset(limit=100)  # Load only 100 samples for testing

    print(f"\nDataset size: {len(dataset)}")
    print(f"\nEmotion distribution: {dataset.get_emotion_counts()}")
    print(
        f"\nSpeaker distribution (first 10): {dict(list(dataset.get_speaker_counts().items())[:10])}"
    )

    # Get a sample
    print("\n--- Sample 0 ---")
    sample = dataset[0]
    print(f"Text: {sample['text']}")
    print(f"Style prompt: {sample['style_prompt']}")
    print(f"Audio path: {sample['audio_path']}")
    print(f"Waveform shape: {sample['waveform'].shape}")
    print(f"Sample rate: {sample['sample_rate']}")
    print(f"Emotion: {sample['emotion']}")
    print(f"Speaker: {sample['speaker']}")
    print(f"Gender: {sample['gender']}")
    print(
        f"Duration/Pitch/Energy: {sample['duration']}/{sample['pitch']}/{sample['energy']}"
    )

    # Test DataLoader
    from torch.utils.data import DataLoader

    print("\n--- Testing DataLoader ---")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(dataloader))

    print(f"Batch waveform shape: {batch['waveform'].shape}")
    print(f"Batch waveform_lengths: {batch['waveform_lengths']}")
    print(f"Batch text (first 2): {batch['text'][:2]}")
    print(f"Batch emotions: {batch['emotion']}")
