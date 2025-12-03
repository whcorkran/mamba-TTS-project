"""
Text to phoneme preprocessing logic recreated from ControlSpeech/baseline/promptTTS/data_gen/tts/txt_processors
"""
import re
import unicodedata
from typing import List, Tuple

from g2p_en import G2p
from g2p_en.expand import normalize_numbers
from nltk import pos_tag
from nltk.tokenize import TweetTokenizer

from transformers import AutoTokenizer, AutoModel

# Phoneme Processing

# Constants
PUNCS = '!,.?;:'


def is_sil_phoneme(p: str) -> bool:
    """Check if a phoneme is a silence phoneme (empty or non-alphabetic)."""
    return p == '' or (len(p) > 0 and not p[0].isalpha())


class BaseTextProcessor:
    """Base class for text processors."""
    
    @staticmethod
    def sp_phonemes():
        """Return special phonemes."""
        return ['|']

    @classmethod
    def process(cls, txt: str) -> Tuple[List, str]:
        """Process text to phonemes. Must be implemented by subclasses."""
        raise NotImplementedError

    @classmethod
    def postprocess(cls, txt_struct: List) -> List:
        """
        Postprocess text structure:
        - Remove silence phonemes from head and tail
        - Add boundaries between words
        - Add BOS and EOS tokens
        """
        # Remove sil phoneme in head and tail
        while len(txt_struct) > 0 and is_sil_phoneme(txt_struct[0][0]):
            txt_struct = txt_struct[1:]
        while len(txt_struct) > 0 and is_sil_phoneme(txt_struct[-1][0]):
            txt_struct = txt_struct[:-1]
        
        # Add boundaries between words
        txt_struct = cls.add_bdr(txt_struct)
        
        # Add BOS and EOS tokens
        txt_struct = [["<BOS>", ["<BOS>"]]] + txt_struct + [["<EOS>", ["<EOS>"]]]
        
        return txt_struct

    @classmethod
    def add_bdr(cls, txt_struct: List) -> List:
        """Add boundary markers (|) between words."""
        txt_struct_ = []
        for i, ts in enumerate(txt_struct):
            txt_struct_.append(ts)
            if i != len(txt_struct) - 1 and \
                    not is_sil_phoneme(txt_struct[i][0]) and not is_sil_phoneme(txt_struct[i + 1][0]):
                txt_struct_.append(['|', ['|']])
        return txt_struct_


class EnG2p(G2p):
    """Extended G2P processor with word tokenization."""
    word_tokenize = TweetTokenizer().tokenize

    def __call__(self, text: str) -> List[str]:
        """
        Convert text to phonemes.
        
        Args:
            text: Input text string
            
        Returns:
            List of phonemes with spaces between words
        """
        # Preprocessing
        words = EnG2p.word_tokenize(text)
        tokens = pos_tag(words)  # tuples of (word, tag)

        # Convert to phonemes
        prons = []
        for word, pos in tokens:
            if re.search("[a-z]", word) is None:
                # Non-alphabetic word (punctuation, numbers, etc.)
                pron = [word]
            elif word in self.homograph2features:
                # Check homograph (words with multiple pronunciations)
                pron1, pron2, pos1 = self.homograph2features[word]
                if pos.startswith(pos1):
                    pron = pron1
                else:
                    pron = pron2
            elif word in self.cmu:
                # Lookup in CMU dictionary
                pron = self.cmu[word][0]
            else:
                # Predict for out-of-vocabulary words
                pron = self.predict(word)

            prons.extend(pron)
            prons.extend([" "])  # Space between words

        return prons[:-1]  # Remove trailing space


class TxtProcessor(BaseTextProcessor):
    """English text processor with G2P conversion."""
    g2p = EnG2p()

    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Preprocess text before G2P conversion.
        
        Steps:
        1. Normalize numbers
        2. Strip accents (NFD normalization)
        3. Convert to lowercase
        4. Remove quotes and parentheses
        5. Normalize hyphens
        6. Remove invalid characters
        7. Normalize punctuation
        8. Handle special abbreviations
        9. Add spaces around punctuation
        """
        # Normalize numbers
        text = normalize_numbers(text)
        
        # Strip accents
        text = ''.join(char for char in unicodedata.normalize('NFD', text)
                      if unicodedata.category(char) != 'Mn')
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove quotes and parentheses
        text = re.sub("[\'\"()]+", "", text)
        
        # Normalize hyphens
        text = re.sub("[-]+", " ", text)
        
        # Remove invalid characters (keep only a-z, spaces, and punctuation)
        text = re.sub(f"[^ a-z{PUNCS}]", "", text)
        
        # Normalize punctuation spacing
        text = re.sub(f" ?([{PUNCS}]) ?", r"\1", text)  # Remove spaces around punctuation
        text = re.sub(f"([{PUNCS}])+", r"\1", text)  # Collapse multiple punctuation
        
        # Handle special abbreviations
        text = text.replace("i.e.", "that is")
        text = text.replace("etc.", "etc")
        
        # Add spaces around punctuation
        text = re.sub(f"([{PUNCS}])", r" \1 ", text)
        
        # Normalize whitespace
        text = re.sub(r"\s+", r" ", text)
        
        return text

    @classmethod
    def process(cls, txt: str) -> Tuple[List, str]:
        """
        Process text to phoneme structure.
        
        Args:
            txt: Raw input text
            
        Returns:
            Tuple of (txt_struct, cleaned_txt) where:
            - txt_struct: List of [word, [phonemes]] pairs
            - cleaned_txt: Preprocessed text string
        """
        # Preprocess text
        txt = cls.preprocess_text(txt).strip()
        
        # Convert to phonemes
        phs = cls.g2p(txt)
        
        # Build text structure: [[word, [phonemes]], ...]
        txt_struct = [[w, []] for w in txt.split(" ")]
        i_word = 0
        for p in phs:
            if p == ' ':
                i_word += 1
            else:
                txt_struct[i_word][1].append(p)
        
        # Postprocess (add boundaries, BOS/EOS)
        txt_struct = cls.postprocess(txt_struct)
        
        return txt_struct, txt

    @classmethod
    def txt_to_ph(cls, txt_raw: str) -> Tuple[str, str, str, List[int], str]:
        """
        Convert raw text to phoneme representation.
        
        Args:
            txt_raw: Raw input text
            
        Returns:
            Tuple of:
            - ph: Space-separated phonemes
            - txt: Cleaned text
            - word: Space-separated words
            - ph2word: Mapping from phoneme index to word index
            - ph_gb_word: Phonemes grouped by word (underscore-separated)
        """
        txt_struct, txt = cls.process(txt_raw)
        
        # Extract phonemes
        ph = [p for w in txt_struct for p in w[1]]
        
        # Group phonemes by word
        ph_gb_word = ["_".join(w[1]) for w in txt_struct]
        
        # Extract words
        words = [w[0] for w in txt_struct]
        
        # Create phoneme-to-word mapping (word_id=0 is reserved for padding)
        ph2word = [w_id + 1 for w_id, w in enumerate(txt_struct) for _ in range(len(w[1]))]
        
        return " ".join(ph), txt, " ".join(words), ph2word, " ".join(ph_gb_word)
    
    @staticmethod
    def txt_to_ph_static(txt_processor, txt_raw: str) -> Tuple[str, str, str, List[int], str]:
        """
        Convert raw text to phoneme representation (static method).
        Compatible with the txt_to_ph method signature from libri_emo_preprocess.py.
        
        This static method allows using TxtProcessor with the same interface pattern
        as BasePreprocessor.txt_to_ph in promptTTS preprocessing pipelines.
        
        Args:
            txt_processor: TxtProcessor instance (or compatible processor)
            txt_raw: Raw input text string
            
        Returns:
            Tuple of:
            - ph: Space-separated phonemes
            - txt: Cleaned text
            - word: Space-separated words
            - ph2word: Mapping from phoneme index to word index (list)
            - ph_gb_word: Phonemes grouped by word (underscore-separated)
        """
        txt_struct, txt = txt_processor.process(txt_raw)
        ph = [p for w in txt_struct for p in w[1]]
        ph_gb_word = ["_".join(w[1]) for w in txt_struct]
        words = [w[0] for w in txt_struct]
        # word_id=0 is reserved for padding
        ph2word = [w_id + 1 for w_id, w in enumerate(txt_struct) for _ in range(len(w[1]))]
        return " ".join(ph), txt, " ".join(words), ph2word, " ".join(ph_gb_word)


# Style Embeddings

class BertModel:
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased")
        for p in model.parameters():
            p.requires_grad = False
        self.tokenizer = tokenizer
        self.model = model

    def forward(self, x):
        tok = self.tokenizer(x, padding=True, truncation=True, return_tensors="pt").to(
            self.model.device
        )

        out = self.model(**tok)
        style = out.last_hidden_state[:, 0]

        return style  # shape is bert embedding (B, 768)

class StyleProcessor(TxtProcessor):
    def __init__(self, model: BertModel):
        super().__init__()
        self.model = model

    def embed(self, x):
        x = self.preprocess_text(x)
        return self.model.forward(x)