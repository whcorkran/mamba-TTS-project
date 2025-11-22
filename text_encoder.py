"""
This module provides:
1. TextEncoder: FFT blocks with self-attention and position-wise feed-forward networks
2. DurationPredictor: Predicts phoneme durations with MSE loss, uses MFA alignments as ground truth
3. TextProcessor: Phoneme conversion, phoneme embeddings, and positional encoding

All implementations use FastSpeech2 components directly from lib.FastSpeech2.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import FastSpeech2 components directly
from lib.FastSpeech2.transformer.Models import get_sinusoid_encoding_table
from lib.FastSpeech2.transformer.Layers import FFTBlock
from lib.FastSpeech2.model.modules import VariancePredictor


class TextEncoder(nn.Module):
    """
    Text Encoder with FFT (Feed-Forward Transformer) blocks.
    
    Uses FastSpeech2's Encoder architecture exactly, but adapted for phoneme vocabulary
    instead of text symbols. Implements:
    - Phoneme embeddings
    - Positional encoding (sinusoidal)
    - Stack of FFT blocks (self-attention + position-wise feed-forward)
    """
    
    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_layers=4,
        n_head=2,
        d_k=64,
        d_v=64,
        d_inner=1024,
        kernel_size=(9, 1),
        dropout=0.1,
        max_seq_len=3000,
        padding_idx=0,
    ):
        """
        Args:
            vocab_size: Size of phoneme vocabulary
            d_model: Dimension of model (hidden size)
            n_layers: Number of FFT blocks
            n_head: Number of attention heads
            d_k: Dimension of key
            d_v: Dimension of value
            d_inner: Dimension of inner feed-forward layer
            kernel_size: Kernel size for position-wise feed-forward (conv1d)
            dropout: Dropout rate
            max_seq_len: Maximum sequence length for positional encoding
            padding_idx: Index for padding token
        """
        super(TextEncoder, self).__init__()
        
        n_position = max_seq_len + 1
        d_word_vec = d_model
        
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        # Phoneme embeddings (equivalent to src_word_emb in FastSpeech2 Encoder)
        self.phoneme_emb = nn.Embedding(
            vocab_size, d_word_vec, padding_idx=padding_idx
        )
        
        # Positional encoding (exactly as in FastSpeech2 Encoder)
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx).unsqueeze(0),
            requires_grad=False,
        )
        
        # Stack of FFT blocks (exactly as in FastSpeech2 Encoder)
        self.layer_stack = nn.ModuleList([
            FFTBlock(
                d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
            )
            for _ in range(n_layers)
        ])
    
    def forward(self, phoneme_ids, mask=None, return_attns=False):
        """
        Forward pass through text encoder (exactly as FastSpeech2 Encoder).
        
        Args:
            phoneme_ids: Tensor of shape (batch_size, seq_len) with phoneme token IDs
            mask: Boolean mask of shape (batch_size, seq_len), True for padding positions
            return_attns: If True, return attention weights
        
        Returns:
            enc_output: Encoded phoneme representations of shape (batch_size, seq_len, d_model)
            If return_attns=True, also returns attn_list
        """
        enc_slf_attn_list = []
        batch_size, max_len = phoneme_ids.shape[0], phoneme_ids.shape[1]
        
        # Prepare masks (exactly as FastSpeech2 Encoder)
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1) if mask is not None else None
        
        # Forward (exactly as FastSpeech2 Encoder)
        if not self.training and phoneme_ids.shape[1] > self.max_seq_len:
            enc_output = self.phoneme_emb(phoneme_ids) + get_sinusoid_encoding_table(
                phoneme_ids.shape[1], self.d_model
            )[:phoneme_ids.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                phoneme_ids.device
            )
        else:
            enc_output = self.phoneme_emb(phoneme_ids) + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
        
        # Pass through FFT blocks (exactly as FastSpeech2 Encoder)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        
        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


class DurationPredictor(nn.Module):
    """
    Duration Predictor for phoneme durations.
    
    Uses FastSpeech2's VariancePredictor exactly. Predicts the duration (in frames) 
    for each phoneme. Uses MSE loss for training with MFA alignments as ground truth.
    """
    
    def __init__(
        self,
        d_model=256,
        filter_size=256,
        kernel_size=3,
        dropout=0.1,
    ):
        """
        Args:
            d_model: Input dimension (encoder hidden size)
            filter_size: Hidden dimension of conv layers
            kernel_size: Kernel size for conv layers
            dropout: Dropout rate
        """
        super(DurationPredictor, self).__init__()
        
        # Create model_config dict to pass to VariancePredictor (exactly as FastSpeech2 expects)
        model_config = {
            "transformer": {
                "encoder_hidden": d_model
            },
            "variance_predictor": {
                "filter_size": filter_size,
                "kernel_size": kernel_size,
                "dropout": dropout
            }
        }
        
        # Use FastSpeech2's VariancePredictor directly
        self.predictor = VariancePredictor(model_config)
    
    def forward(self, encoder_output, mask=None):
        """
        Forward pass through duration predictor (exactly as VariancePredictor).
        
        Args:
            encoder_output: Encoder output of shape (batch_size, seq_len, d_model)
            mask: Boolean mask of shape (batch_size, seq_len), True for padding positions
        
        Returns:
            log_duration: Predicted log duration of shape (batch_size, seq_len)
        """
        return self.predictor(encoder_output, mask)
    
    def compute_loss(self, log_duration_pred, duration_target, mask=None):
        """
        Compute MSE loss for duration prediction.
        
        Args:
            log_duration_pred: Predicted log duration (batch_size, seq_len)
            duration_target: Ground truth duration from MFA alignments (batch_size, seq_len)
            mask: Boolean mask for padding positions (batch_size, seq_len)
        
        Returns:
            loss: MSE loss (scalar)
        """
        # Convert target to log space (add small epsilon to avoid log(0))
        log_duration_target = torch.log(duration_target.float() + 1e-8)
        
        # Compute MSE loss
        loss = F.mse_loss(log_duration_pred, log_duration_target, reduction='none')
        
        # Mask out padding positions
        if mask is not None:
            loss = loss.masked_fill(mask, 0.0)
            # Average over non-padding positions
            loss = loss.sum() / (~mask).sum().float()
        else:
            loss = loss.mean()
        
        return loss


class TextProcessor:
    """
    Text Processing for phoneme conversion, embeddings, and positional encoding.
    
    Handles:
    - Text to phoneme conversion
    - Phoneme vocabulary management
    - Phoneme tokenization
    
    Uses FastSpeech2's positional encoding function.
    """
    
    def __init__(
        self,
        vocab_path=None,
        vocab_list=None,
        padding_token="<PAD>",
        unk_token="<UNK>",
    ):
        """
        Initialize text processor.
        
        Args:
            vocab_path: Path to phoneme vocabulary JSON file (list format)
            vocab_list: Phoneme vocabulary as a list (alternative to vocab_path)
            padding_token: Token used for padding
            unk_token: Token used for unknown phonemes
        """
        if vocab_path is not None:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.vocab_list = json.load(f)
        elif vocab_list is not None:
            self.vocab_list = vocab_list
        else:
            raise ValueError("Either vocab_path or vocab_list must be provided")
        
        # Create phoneme to ID mapping
        self.phoneme_to_id = {phoneme: idx for idx, phoneme in enumerate(self.vocab_list)}
        self.id_to_phoneme = {idx: phoneme for phoneme, idx in self.phoneme_to_id.items()}
        
        self.vocab_size = len(self.vocab_list)
        self.padding_token = padding_token
        self.unk_token = unk_token
        
        # Get padding ID (default to 0 if not found)
        self.padding_id = self.phoneme_to_id.get(padding_token, 0)
        
        # Get UNK ID (default to padding_id if UNK token not in vocab)
        # This handles vocabularies that don't have an explicit UNK token
        if unk_token in self.phoneme_to_id:
            self.unk_id = self.phoneme_to_id[unk_token]
        else:
            # If UNK not in vocab, use padding ID as fallback
            # This is common when vocab only has <PAD> at index 0
            self.unk_id = self.padding_id
    
    def text_to_phonemes(self, text, g2p_processor=None):
        """
        Convert text to phonemes.
        
        Args:
            text: Input text string
            g2p_processor: Optional G2P processor (if None, expects pre-phonemized text)
        
        Returns:
            phonemes: List of phoneme strings
        """
        if g2p_processor is not None:
            # Use provided G2P processor
            result = g2p_processor(text)
            if isinstance(result, dict):
                phonemes = result.get('ph', '').split()
            else:
                phonemes = result.split() if isinstance(result, str) else result
        else:
            # Assume text is already space-separated phonemes
            phonemes = text.split()
        
        return phonemes
    
    def phonemes_to_ids(self, phonemes):
        """
        Convert phoneme strings to token IDs.
        
        Args:
            phonemes: List of phoneme strings
        
        Returns:
            ids: List of token IDs
        """
        ids = [
            self.phoneme_to_id.get(ph, self.unk_id)
            for ph in phonemes
        ]
        return ids
    
    def ids_to_phonemes(self, ids):
        """
        Convert token IDs to phoneme strings.
        
        Args:
            ids: List of token IDs
        
        Returns:
            phonemes: List of phoneme strings
        """
        phonemes = [
            self.id_to_phoneme.get(id, self.unk_token)
            for id in ids
        ]
        return phonemes
    
    def process_text(self, text, g2p_processor=None, max_length=None):
        """
        Process text: convert to phonemes and then to token IDs.
        
        Args:
            text: Input text string
            g2p_processor: Optional G2P processor
            max_length: Optional maximum length (truncate if longer)
        
        Returns:
            phoneme_ids: List of phoneme token IDs
            phonemes: List of phoneme strings
        """
        # Convert text to phonemes
        phonemes = self.text_to_phonemes(text, g2p_processor)
        
        # Truncate if necessary
        if max_length is not None and len(phonemes) > max_length:
            phonemes = phonemes[:max_length]
        
        # Convert to IDs
        phoneme_ids = self.phonemes_to_ids(phonemes)
        
        return phoneme_ids, phonemes
    
    def create_phoneme_embedding(self, embedding_dim, padding_idx=None):
        """
        Create a phoneme embedding layer.
        
        Args:
            embedding_dim: Dimension of embeddings
            padding_idx: Index for padding token (default: self.padding_id)
        
        Returns:
            embedding: nn.Embedding layer
        """
        if padding_idx is None:
            padding_idx = self.padding_id
        
        return nn.Embedding(
            self.vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )
    
    def create_positional_encoding(self, max_length, embedding_dim, padding_idx=None):
        """
        Create positional encoding table using FastSpeech2's function.
        
        Args:
            max_length: Maximum sequence length
            embedding_dim: Dimension of embeddings
            padding_idx: Index for padding token (default: self.padding_id)
        
        Returns:
            pos_enc: Positional encoding tensor of shape (max_length, embedding_dim)
        """
        if padding_idx is None:
            padding_idx = self.padding_id
        
        return get_sinusoid_encoding_table(max_length, embedding_dim, padding_idx)
    
    def batch_process(self, texts, g2p_processor=None, max_length=None, pad_to_max=True):
        """
        Process a batch of texts.
        
        Args:
            texts: List of text strings
            g2p_processor: Optional G2P processor
            max_length: Optional maximum length
            pad_to_max: If True, pad all sequences to the same length
        
        Returns:
            phoneme_ids_batch: Padded tensor of shape (batch_size, max_seq_len)
            lengths: List of actual sequence lengths
            masks: Boolean mask tensor of shape (batch_size, max_seq_len)
        """
        # Process each text
        phoneme_ids_list = []
        lengths = []
        
        for text in texts:
            phoneme_ids, _ = self.process_text(text, g2p_processor, max_length)
            phoneme_ids_list.append(phoneme_ids)
            lengths.append(len(phoneme_ids))
        
        # Pad to same length if requested
        if pad_to_max:
            max_len = max(lengths) if lengths else 0
            phoneme_ids_batch = []
            for ids in phoneme_ids_list:
                padded = ids + [self.padding_id] * (max_len - len(ids))
                phoneme_ids_batch.append(padded)
            
            phoneme_ids_batch = torch.LongTensor(phoneme_ids_batch)
            
            # Create mask (True for padding positions)
            masks = torch.zeros(len(texts), max_len, dtype=torch.bool)
            for i, length in enumerate(lengths):
                masks[i, length:] = True
        else:
            phoneme_ids_batch = [torch.LongTensor(ids) for ids in phoneme_ids_list]
            masks = None
        
        return phoneme_ids_batch, lengths, masks
