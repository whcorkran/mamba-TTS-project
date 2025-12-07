"""
Style Cross-Attention Modules for ControlSpeech Architecture

Implements:
1. Style K,V projections from SMSD output
2. Cross-Attention #1: Text ⊗ Style (before duration predictor)
3. Cross-Attention #2: Upsampled ⊗ Style (before codec generator)
4. Length Regulator (phoneme-level → frame-level upsampling)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleProjection(nn.Module):
    """
    Project SMSD style embedding to Key and Value for cross-attention
    
    The SMSD module outputs a sampled style vector (B, d_style).
    This module projects it to K, V tensors for cross-attention with text/decoder.
    """
    def __init__(self, d_style, d_model, dropout=0.1):
        """
        Args:
            d_style: Dimension of SMSD output (256 typically)
            d_model: Model hidden dimension (for K, V projection)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_style = d_style
        self.d_model = d_model
        
        # Project style to key space
        self.key_proj = nn.Sequential(
            nn.Linear(d_style, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        
        # Project style to value space
        self.value_proj = nn.Sequential(
            nn.Linear(d_style, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, style_emb):
        """
        Args:
            style_emb: (B, d_style) from SMSD
        
        Returns:
            K: (B, 1, d_model) - Key for cross-attention
            V: (B, 1, d_model) - Value for cross-attention
        """
        # Project to K, V
        K = self.key_proj(style_emb)      # (B, d_model)
        V = self.value_proj(style_emb)    # (B, d_model)
        
        # Add sequence dimension (single token)
        K = K.unsqueeze(1)  # (B, 1, d_model)
        V = V.unsqueeze(1)  # (B, 1, d_model)
        
        return K, V


class StyleTextCrossAttention(nn.Module):
    """
    Cross-Attention #1: Text ⊗ Style (before Duration Predictor)
    
    Query: Text encoder output (phoneme-level)
    Key, Value: SMSD style output (projected)
    
    This conditions the text representation on the style before duration prediction.
    """
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        """
        Args:
            d_model: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Post-attention layers
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
    
    def forward(self, text_hidden, style_K, style_V, text_mask=None):
        """
        Args:
            text_hidden: (B, T_text, d_model) - from text encoder
            style_K: (B, 1, d_model) - style key
            style_V: (B, 1, d_model) - style value
            text_mask: (B, T_text) - True for padding positions
        
        Returns:
            styled_text: (B, T_text, d_model) - style-conditioned text features
        """
        # Cross-attention: text attends to style
        # No key_padding_mask needed since style is single token
        attn_out, attn_weights = self.cross_attn(
            query=text_hidden,     # (B, T_text, d_model)
            key=style_K,           # (B, 1, d_model)
            value=style_V,         # (B, 1, d_model)
            need_weights=False,
        )
        
        # Residual + norm
        text_hidden = text_hidden + self.dropout(attn_out)
        text_hidden = self.norm(text_hidden)
        
        # Feed-forward
        ffn_out = self.ffn(text_hidden)
        text_hidden = text_hidden + ffn_out
        styled_text = self.ffn_norm(text_hidden)
        
        return styled_text


class LengthRegulator(nn.Module):
    """
    Length Regulator: Upsample phoneme-level features to frame-level
    
    Takes phoneme-level features and durations, expands each phoneme
    by repeating it according to its predicted duration.
    
    This is the component represented by the dashed boxes in the architecture diagram.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, hidden, durations, max_len=None):
        """
        Expand phoneme-level features to frame-level using durations
        
        Args:
            hidden: (B, T_text, d_model) - phoneme-level features
            durations: (B, T_text) - predicted durations (frames per phoneme)
            max_len: Optional maximum output length (for batching)
        
        Returns:
            expanded: (B, T_frame, d_model) - frame-level features
            output_lengths: (B,) - actual lengths before padding
        """
        B, T, D = hidden.shape
        device = hidden.device
        
        # Round durations to positive integers
        durations = torch.clamp(torch.round(durations), min=0).long()
        
        # Calculate output lengths for each sample
        output_lengths = durations.sum(dim=1)  # (B,)
        
        # Determine max length
        if max_len is None:
            max_len = output_lengths.max().item()
        
        # Expand each sample in the batch
        expanded = torch.zeros(B, max_len, D, device=device, dtype=hidden.dtype)
        
        for b in range(B):
            pos = 0
            for t in range(T):
                dur = durations[b, t].item()
                if dur > 0 and pos < max_len:
                    # Repeat phoneme hidden state dur times
                    end_pos = min(pos + dur, max_len)
                    expanded[b, pos:end_pos] = hidden[b, t].unsqueeze(0).repeat(end_pos - pos, 1)
                    pos = end_pos
                
                if pos >= max_len:
                    break
        
        return expanded, output_lengths
    
    def forward_with_target(self, hidden, target_durations):
        """
        Training mode: use ground truth durations from forced alignment
        
        Args:
            hidden: (B, T_text, d_model)
            target_durations: (B, T_text) - ground truth durations from MFA
        
        Returns:
            expanded: (B, T_frame, d_model)
            output_lengths: (B,)
        """
        return self.forward(hidden, target_durations)


class StyleDecoderCrossAttention(nn.Module):
    """
    Cross-Attention #2: Upsampled ⊗ Style (before Codec Generator)
    
    Query: Upsampled features (frame-level, after length regulation)
    Key, Value: SMSD style output (projected, REUSED from Cross-Attention #1)
    
    This conditions the frame-level features on style before codec generation.
    """
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        """
        Args:
            d_model: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Post-attention layers
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
    
    def forward(self, upsampled_hidden, style_K, style_V, frame_mask=None):
        """
        Args:
            upsampled_hidden: (B, T_frame, d_model) - from length regulator
            style_K: (B, 1, d_model) - style key (REUSED from cross-attn #1)
            style_V: (B, 1, d_model) - style value (REUSED from cross-attn #1)
            frame_mask: (B, T_frame) - True for padding positions
        
        Returns:
            styled_frames: (B, T_frame, d_model) - style-conditioned frame features
        """
        # Cross-attention: upsampled features attend to style
        attn_out, attn_weights = self.cross_attn(
            query=upsampled_hidden,  # (B, T_frame, d_model)
            key=style_K,             # (B, 1, d_model)
            value=style_V,           # (B, 1, d_model)
            need_weights=False,
        )
        
        # Residual + norm
        upsampled_hidden = upsampled_hidden + self.dropout(attn_out)
        upsampled_hidden = self.norm(upsampled_hidden)
        
        # Feed-forward
        ffn_out = self.ffn(upsampled_hidden)
        upsampled_hidden = upsampled_hidden + ffn_out
        styled_frames = self.ffn_norm(upsampled_hidden)
        
        return styled_frames


class StyleConditioningPipeline(nn.Module):
    """
    Complete style conditioning pipeline combining all components
    
    This wraps all the cross-attention layers and length regulation
    into a single module for easy integration.
    """
    def __init__(self, d_style=512, d_model=512, num_heads=8, dropout=0.1):
        """
        Args:
            d_style: Dimension of SMSD output (512 as per ControlSpeech paper)
            d_model: Model hidden dimension (512 as per ControlSpeech paper)
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        # Style projection
        self.style_proj = StyleProjection(d_style, d_model, dropout)
        
        # Cross-attention layers
        self.cross_attn_1 = StyleTextCrossAttention(d_model, num_heads, dropout)
        self.cross_attn_2 = StyleDecoderCrossAttention(d_model, num_heads, dropout)
        
        # Length regulator
        self.length_regulator = LengthRegulator()
    
    def forward(
        self,
        text_hidden,
        style_emb,
        durations,
        text_mask=None,
        max_frame_len=None,
    ):
        """
        Full forward pass through style conditioning pipeline
        
        Args:
            text_hidden: (B, T_text, d_model) - from text encoder
            style_emb: (B, d_style) - from SMSD
            durations: (B, T_text) - predicted durations
            text_mask: (B, T_text) - text padding mask
            max_frame_len: Optional max length for frame-level features
        
        Returns:
            styled_frames: (B, T_frame, d_model) - ready for codec generator
            output_lengths: (B,) - actual frame lengths
            style_K: (B, 1, d_model) - for inspection/visualization
            style_V: (B, 1, d_model) - for inspection/visualization
        """
        # 1. Project style to K, V
        style_K, style_V = self.style_proj(style_emb)
        
        # 2. Cross-Attention #1: Text ⊗ Style
        styled_text = self.cross_attn_1(text_hidden, style_K, style_V, text_mask)
        
        # 3. Length Regulation (upsampling)
        upsampled, output_lengths = self.length_regulator(
            styled_text, durations, max_len=max_frame_len
        )
        
        # 4. Cross-Attention #2: Upsampled ⊗ Style
        styled_frames = self.cross_attn_2(upsampled, style_K, style_V)
        
        return styled_frames, output_lengths, style_K, style_V


def test_style_cross_attention():
    """Test the style conditioning pipeline"""
    print("Testing Style Cross-Attention Modules...")
    
    # Parameters
    batch_size = 4
    T_text = 20      # phoneme sequence length
    d_style = 512    # SMSD output dimension (ControlSpeech paper)
    d_model = 512    # model hidden dimension (ControlSpeech paper)
    
    # Initialize pipeline
    pipeline = StyleConditioningPipeline(
        d_style=d_style,
        d_model=d_model,
        num_heads=8,
        dropout=0.1,
    )
    
    # Create dummy inputs
    text_hidden = torch.randn(batch_size, T_text, d_model)
    style_emb = torch.randn(batch_size, d_style)
    durations = torch.randint(1, 5, (batch_size, T_text)).float()  # 1-4 frames per phoneme
    
    print(f"\nInput shapes:")
    print(f"  text_hidden: {text_hidden.shape}")
    print(f"  style_emb: {style_emb.shape}")
    print(f"  durations: {durations.shape}")
    
    # Forward pass
    styled_frames, output_lengths, style_K, style_V = pipeline(
        text_hidden, style_emb, durations
    )
    
    print(f"\nOutput shapes:")
    print(f"  styled_frames: {styled_frames.shape}")
    print(f"  output_lengths: {output_lengths.shape} = {output_lengths}")
    print(f"  style_K: {style_K.shape}")
    print(f"  style_V: {style_V.shape}")
    
    # Verify dimensions
    assert styled_frames.shape[0] == batch_size
    assert styled_frames.shape[2] == d_model
    assert output_lengths.shape[0] == batch_size
    
    print("\nAll tests passed!")
    
    # Test individual components
    print("\nTesting individual components...")
    
    # 1. Style Projection
    style_proj = StyleProjection(d_style, d_model)
    K, V = style_proj(style_emb)
    print(f"  StyleProjection: {style_emb.shape} → K:{K.shape}, V:{V.shape}")
    
    # 2. Cross-Attention #1
    cross_attn_1 = StyleTextCrossAttention(d_model)
    styled_text = cross_attn_1(text_hidden, K, V)
    print(f"  CrossAttention #1: {text_hidden.shape} → {styled_text.shape}")
    
    # 3. Length Regulator
    length_reg = LengthRegulator()
    upsampled, lengths = length_reg(styled_text, durations)
    print(f"  LengthRegulator: {styled_text.shape} → {upsampled.shape}")
    
    # 4. Cross-Attention #2
    cross_attn_2 = StyleDecoderCrossAttention(d_model)
    styled_frames_2 = cross_attn_2(upsampled, K, V)
    print(f"  CrossAttention #2: {upsampled.shape} → {styled_frames_2.shape}")
    
    print("\n✅ Individual component tests passed!")


if __name__ == "__main__":
    test_style_cross_attention()


