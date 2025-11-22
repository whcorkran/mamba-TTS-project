"""
Text to phoneme preprocessing for ML model data loading.
Recreated from ControlSpeech/baseline/promptTTS/data_gen/tts preprocessing logic.
"""
from .text_processor import TxtProcessor


def text_to_phoneme(text: str) -> dict:
    """
    Convert text to phoneme representation.
    
    Args:
        text: Raw input text string
        
    Returns:
        Dictionary containing:
        - 'ph': Space-separated phonemes
        - 'txt': Cleaned text
        - 'word': Space-separated words
        - 'ph2word': Mapping from phoneme index to word index (list)
        - 'ph_gb_word': Phonemes grouped by word (underscore-separated)
    """
    processor = TxtProcessor()
    ph, txt, word, ph2word, ph_gb_word = processor.txt_to_ph(text)
    
    return {
        'ph': ph,
        'txt': txt,
        'word': word,
        'ph2word': ph2word,
        'ph_gb_word': ph_gb_word
    }

def preprocess_text(text: str) -> str:
    """
    Preprocess text (normalize, clean) before phoneme conversion.
    
    Args:
        text: Raw input text
        
    Returns:
        Preprocessed text string
    """
    return TxtProcessor.preprocess_text(text)




if __name__ == '__main__':
    # Example usage
    test_text = "Hello, world! This is a test."
    result = text_to_phoneme(test_text)
    
    print("Original text:", test_text)
    print("Cleaned text:", result['txt'])
    print("Words:", result['word'])
    print("Phonemes:", result['ph'])
    print("Phonemes grouped by word:", result['ph_gb_word'])
    print("Phoneme to word mapping (first 10):", result['ph2word'][:10])