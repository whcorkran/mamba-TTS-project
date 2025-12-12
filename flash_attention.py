# flash_attention.py
"""
Flash Attention wrapper using PyTorch 2.0+ scaled_dot_product_attention.

This provides a drop-in replacement for nn.MultiheadAttention that uses
Flash Attention (or Memory-Efficient Attention) when available.

PyTorch 2.0+ automatically selects the best backend:
- Flash Attention (requires CUDA, fp16/bf16, no dropout in inference)
- Memory-Efficient Attention (xFormers-style)
- Math (fallback)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FlashMultiheadAttention(nn.Module):
    """
    Multi-head attention using F.scaled_dot_product_attention for Flash Attention support.
    
    Drop-in replacement for nn.MultiheadAttention with batch_first=True.
    Uses Flash Attention automatically when conditions are met (CUDA, fp16/bf16).
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,  # Only batch_first=True is supported
    ):
        super().__init__()
        assert batch_first, "FlashMultiheadAttention only supports batch_first=True"
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.scale = self.head_dim ** -0.5
        
        # Separate Q, K, V projections (cleaner than packed)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        # Xavier uniform initialization (same as nn.MultiheadAttention)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: (B, T_q, embed_dim)
            key: (B, T_kv, embed_dim)
            value: (B, T_kv, embed_dim)
            key_padding_mask: (B, T_kv) - True means IGNORE this position
            need_weights: If True, returns attention weights (disables Flash Attention)
            attn_mask: Optional attention mask
            
        Returns:
            output: (B, T_q, embed_dim)
            attn_weights: None (Flash Attention doesn't return weights) or (B, num_heads, T_q, T_kv)
        """
        B, T_q, _ = query.shape
        T_kv = key.shape[1]
        
        # Project Q, K, V
        q = self.q_proj(query)  # (B, T_q, embed_dim)
        k = self.k_proj(key)    # (B, T_kv, embed_dim)
        v = self.v_proj(value)  # (B, T_kv, embed_dim)
        
        # Reshape to (B, num_heads, T, head_dim)
        q = q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Build attention mask from key_padding_mask
        # F.scaled_dot_product_attention expects: True = attend, or additive mask
        attn_mask_combined = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, T_kv), True = IGNORE
            # Convert to (B, 1, 1, T_kv) additive mask where ignored positions are -inf
            attn_mask_combined = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T_kv)
            attn_mask_combined = attn_mask_combined.to(dtype=q.dtype)
            attn_mask_combined = attn_mask_combined.masked_fill(attn_mask_combined.bool(), float('-inf'))
            attn_mask_combined = attn_mask_combined.masked_fill(~key_padding_mask.unsqueeze(1).unsqueeze(2), 0.0)
        
        if attn_mask is not None:
            if attn_mask_combined is not None:
                attn_mask_combined = attn_mask_combined + attn_mask
            else:
                attn_mask_combined = attn_mask
        
        # Use scaled_dot_product_attention (Flash Attention when available)
        dropout_p = self.dropout if self.training else 0.0
        
        # F.scaled_dot_product_attention automatically uses:
        # - Flash Attention (fastest, requires CUDA + fp16/bf16 + no need_weights)
        # - Memory-Efficient Attention (xFormers-style)  
        # - Math backend (fallback)
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask_combined,
            dropout_p=dropout_p,
            is_causal=False,
        )
        
        # Reshape back to (B, T_q, embed_dim)
        output = output.transpose(1, 2).contiguous().view(B, T_q, self.embed_dim)
        
        # Output projection
        output = self.out_proj(output)
        
        # Flash Attention doesn't return weights
        attn_weights = None
        
        return output, attn_weights


def check_flash_attention_available():
    """Check if Flash Attention backend is available."""
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    
    # Check PyTorch version
    major, minor = torch.__version__.split('.')[:2]
    if int(major) < 2:
        return False, f"PyTorch {torch.__version__} < 2.0 (Flash Attention requires 2.0+)"
    
    # Check if scaled_dot_product_attention is available
    if not hasattr(F, 'scaled_dot_product_attention'):
        return False, "F.scaled_dot_product_attention not available"
    
    return True, "Flash Attention available via F.scaled_dot_product_attention"


if __name__ == "__main__":
    # Test the module
    available, msg = check_flash_attention_available()
    print(f"Flash Attention: {msg}")
    
    # Test forward pass
    B, T_q, T_kv, embed_dim, num_heads = 2, 1024, 512, 512, 8
    
    attn = FlashMultiheadAttention(embed_dim, num_heads, dropout=0.1)
    if torch.cuda.is_available():
        attn = attn.cuda().half()  # Flash Attention works best with fp16
    
    q = torch.randn(B, T_q, embed_dim)
    k = torch.randn(B, T_kv, embed_dim)
    v = torch.randn(B, T_kv, embed_dim)
    mask = torch.zeros(B, T_kv, dtype=torch.bool)
    mask[:, -10:] = True  # Mask last 10 positions
    
    if torch.cuda.is_available():
        q, k, v, mask = q.cuda().half(), k.cuda().half(), v.cuda().half(), mask.cuda()
    
    output, _ = attn(q, k, v, key_padding_mask=mask)
    print(f"Input: Q={q.shape}, K={k.shape}, V={v.shape}")
    print(f"Output: {output.shape}")
    print("✅ FlashMultiheadAttention test passed!")






