"""
Text and audio preprocessing module.
Based on ControlSpeech/baseline/promptTTS preprocessing logic.
"""
from .preprocess import DatasetPreprocessor, preprocess_dataset
from .text_processor import TxtProcessor, BaseTextProcessor, BertModel, StyleProcessor, is_sil_phoneme
from .audio_encoder import FACodecEncoder, BaseAudioPreprocessor
from .phonemes import SPECIAL_TOKENS, build_phoneme_vocabulary, load_phoneme_vocabulary

__all__ = [
    # Preprocessing
    'DatasetPreprocessor',
    'preprocess_dataset',
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

