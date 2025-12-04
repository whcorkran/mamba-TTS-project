"""
Text and audio preprocessing module.
Based on ControlSpeech/baseline/promptTTS preprocessing logic.
"""
from .preprocess import DatasetPreprocessor, preprocess_dataset
from .preprocess_parallel import (
    ParallelDatasetPreprocessor,
    preprocess_dataset_parallel,
    BatchedStyleProcessor,
    BatchedAudioEncoder,
)
from .text_processor import TxtProcessor, BaseTextProcessor, BertModel, StyleProcessor, is_sil_phoneme
from .audio_encoder import FACodecEncoder, BaseAudioPreprocessor
from .phonemes import SPECIAL_TOKENS, build_phoneme_vocabulary, load_phoneme_vocabulary

__all__ = [
    # Preprocessing (sequential)
    'DatasetPreprocessor',
    'preprocess_dataset',
    # Preprocessing (parallel - recommended)
    'ParallelDatasetPreprocessor',
    'preprocess_dataset_parallel',
    'BatchedStyleProcessor',
    'BatchedAudioEncoder',
    # Text processing
    'TxtProcessor',
    'BaseTextProcessor',
    'BertModel',
    'StyleProcessor',
    'is_sil_phoneme',
    # Audio processing
    'FACodecEncoder',
    'BaseAudioPreprocessor',
    # Phoneme vocabulary
    'SPECIAL_TOKENS',
    'build_phoneme_vocabulary',
    'load_phoneme_vocabulary',
]

