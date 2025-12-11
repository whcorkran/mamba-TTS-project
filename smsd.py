# some of the dependencies have deprecated apis and are not maintained, suppress warnings to not interrupt training
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import math


class SMSD(nn.Module):
    """
    Style Mixture Semantic Density (SMSD) Module
    
    Implements a Gaussian Mixture Density Network over BERT style embeddings
    to handle the many-to-many mapping between style text descriptions and 
    style realizations.
    
    From ControlSpeech paper (arXiv:2406.01205) Section 3.3
    """
    def __init__(
        self,
        bert_model="bert-base-uncased",
        bert_dim=768,
        style_dim=512,  # ControlSpeech paper: 512-dimensional global ground truth style vector from FACodec
        num_mixtures=5,  # K: number of Gaussian components (from paper Appendix H)
        hidden_dim=512,  # MLP hidden dimension
        dropout=0.1,
        variance_mode="isotropic_across_clusters",  # "isotropic_across_clusters", "isotropic", "diagonal", "fixed"
        freeze_bert=True,
    ):
        super().__init__()
        
        self.style_dim = style_dim
        self.num_mixtures = num_mixtures
        self.variance_mode = variance_mode
        
        # 1. BERT Style Semantic Encoder
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.bert = AutoModel.from_pretrained(bert_model)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # 2. Mixture Density Network (MDN) Head
        self.mdn_head = MDNHead(
            input_dim=bert_dim,
            style_dim=style_dim,
            num_mixtures=num_mixtures,
            hidden_dim=hidden_dim,
            dropout=dropout,
            variance_mode=variance_mode,
        )
    
    def encode_style_text(self, texts):
        """
        Encode style text using BERT [CLS] token
        
        Args:
            texts: List[str] or str - style descriptions
        Returns:
            X'_s: (B, bert_dim) BERT [CLS] embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128,
        )
        
        # Move to same device as model
        encoded = {k: v.to(self.bert.device) for k, v in encoded.items()}
        
        # Get BERT embeddings
        with torch.set_grad_enabled(self.bert.training):
            outputs = self.bert(**encoded)
        
        # Extract [CLS] token embedding as global semantic representation
        x_s_prime = outputs.last_hidden_state[:, 0, :]  # (B, bert_dim)
        
        return x_s_prime
    
    def forward(self, style_texts, y_true=None, return_params=False):
        """
        Forward pass for training or inference
        
        Args:
            style_texts: List[str] or str - style text descriptions
            y_true: (B, style_dim) - ground truth style vectors from audio (for training)
            return_params: bool - whether to return mixture parameters
        
        Returns:
            If training (y_true provided):
                loss: scalar - negative log-likelihood loss
            If inference (y_true is None):
                y_sampled: (B, style_dim) - sampled style vector
                [optional] mixture parameters if return_params=True
        """
        # Encode style text with BERT
        x_s_prime = self.encode_style_text(style_texts)  # (B, bert_dim)
        
        # Get mixture parameters from MDN head
        pi, mu, sigma = self.mdn_head(x_s_prime)
        # pi: (B, K)
        # mu: (B, K, style_dim)
        # sigma: (B,) or (B, K) or (B, K, style_dim) depending on variance_mode
        
        if y_true is not None:
            # Training mode: compute mixture NLL loss
            loss = mixture_nll_loss(y_true, pi, mu, sigma, self.variance_mode)
            return loss
        else:
            # Inference mode: sample from mixture
            y_sampled = self.sample(pi, mu, sigma)
            
            if return_params:
                return y_sampled, (pi, mu, sigma)
            return y_sampled
    
    def sample(self, pi, mu, sigma):
        """
        Sample style vector from predicted Gaussian mixture
        
        Args:
            pi: (B, K) mixture weights
            mu: (B, K, style_dim) means
            sigma: variance (shape depends on variance_mode)
        
        Returns:
            y_sampled: (B, style_dim)
        """
        B, K, d = mu.shape
        
        # Sample component index k ~ Categorical(pi)
        k_indices = torch.multinomial(pi, num_samples=1).squeeze(-1)  # (B,)
        
        # Select the chosen component's mean
        mu_selected = mu[torch.arange(B), k_indices]  # (B, style_dim)
        
        # Sample from N(mu_k, sigma_k^2)
        if self.variance_mode == "isotropic_across_clusters":
            # Single scalar variance for all components and dimensions
            std = sigma.unsqueeze(-1)  # (B, 1)
            noise = torch.randn_like(mu_selected) * std
        elif self.variance_mode == "isotropic":
            # One variance per component, shared across dimensions
            std_selected = sigma[torch.arange(B), k_indices].unsqueeze(-1)  # (B, 1)
            noise = torch.randn_like(mu_selected) * std_selected
        elif self.variance_mode == "diagonal":
            # Full diagonal covariance per component
            std_selected = sigma[torch.arange(B), k_indices]  # (B, style_dim)
            noise = torch.randn_like(mu_selected) * std_selected
        else:  # fixed
            noise = torch.randn_like(mu_selected) * 0.1  # fixed std
        
        y_sampled = mu_selected + noise
        return y_sampled


class MDNHead(nn.Module):
    """
    Mixture Density Network Head
    
    Takes BERT embedding and outputs parameters for K-component Gaussian mixture
    """
    def __init__(
        self,
        input_dim,
        style_dim,
        num_mixtures,
        hidden_dim,
        dropout=0.1,
        variance_mode="isotropic_across_clusters",
    ):
        super().__init__()
        
        self.num_mixtures = num_mixtures
        self.style_dim = style_dim
        self.variance_mode = variance_mode
        
        # MLP backbone
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Three output heads for mixture parameters
        
        # 1. Mixture weights: K logits
        self.pi_head = nn.Linear(hidden_dim, num_mixtures)
        
        # 2. Means: K * style_dim
        self.mu_head = nn.Linear(hidden_dim, num_mixtures * style_dim)
        
        # 3. Variances (depends on mode)
        if variance_mode == "isotropic_across_clusters":
            # Single variance shared across all K and all dimensions
            self.sigma_head = nn.Linear(hidden_dim, 1)
        elif variance_mode == "isotropic":
            # One variance per component
            self.sigma_head = nn.Linear(hidden_dim, num_mixtures)
        elif variance_mode == "diagonal":
            # Full diagonal per component
            self.sigma_head = nn.Linear(hidden_dim, num_mixtures * style_dim)
        else:  # fixed
            self.sigma_head = None
        
        # Noise perturbation module
        if self.sigma_head is not None:
            self.noise_net = NoiseNet()
    
    def forward(self, x):
        """
        Args:
            x: (B, input_dim) - BERT [CLS] embeddings
        
        Returns:
            pi: (B, K) - mixture weights (softmax over K)
            mu: (B, K, style_dim) - means
            sigma: variance (shape depends on variance_mode)
        """
        B = x.shape[0]
        
        # Apply MLP backbone
        h = self.mlp(x)  # (B, hidden_dim)
        
        # 1. Mixture weights
        pi_logits = self.pi_head(h)  # (B, K)
        pi = F.softmax(pi_logits, dim=-1)  # (B, K)
        
        # 2. Means
        mu_flat = self.mu_head(h)  # (B, K * style_dim)
        mu = mu_flat.view(B, self.num_mixtures, self.style_dim)  # (B, K, style_dim)
        
        # 3. Variances with noise perturbation
        if self.sigma_head is not None:
            sigma_raw = self.sigma_head(h)  # (B, 1) or (B, K) or (B, K*style_dim)
            
            # Add noise perturbation
            sigma_raw = self.noise_net(sigma_raw)
            
            # Ensure positive variance using softplus
            if self.variance_mode == "isotropic_across_clusters":
                sigma = F.softplus(sigma_raw).squeeze(-1)  # (B,)
            elif self.variance_mode == "isotropic":
                sigma = F.softplus(sigma_raw)  # (B, K)
            else:  # diagonal
                sigma = F.softplus(sigma_raw).view(B, self.num_mixtures, self.style_dim)  # (B, K, style_dim)
        else:  # fixed variance
            sigma = torch.ones(B, device=x.device) * 0.1
        
        return pi, mu, sigma


class NoiseNet(nn.Module):
    """
    Noise Perturbation Module
    
    Adds learnable noise to variance predictions to enhance style diversity
    """
    def __init__(self, noise_scale=0.1):
        super().__init__()
        self.noise_scale = nn.Parameter(torch.tensor(noise_scale))
    
    def forward(self, x, epsilon=None):
        """
        Args:
            x: variance predictions
            epsilon: optional noise (if None, sample during training)
        
        Returns:
            x + noise
        """
        if self.training:
            if epsilon is None:
                epsilon = torch.randn_like(x)
            return x + self.noise_scale * epsilon
        else:
            # During inference, optionally add small noise for diversity
            return x


def mixture_nll_loss(y_true, pi, mu, sigma, variance_mode="isotropic_across_clusters"):
    """
    Negative Log-Likelihood loss for Gaussian Mixture Model
    
    Args:
        y_true: (B, d) ground truth style vectors
        pi: (B, K) mixture weights
        mu: (B, K, d) means
        sigma: variances (shape depends on variance_mode)
        variance_mode: str
    
    Returns:
        loss: scalar
    """
    B, K, d = mu.shape
    
    # Expand y_true for broadcasting
    y_true_expanded = y_true.unsqueeze(1)  # (B, 1, d)
    
    # Compute log probabilities for each component
    # log N(y | mu_k, sigma_k^2)
    
    if variance_mode == "isotropic_across_clusters":
        # Single variance for all
        variance = sigma.unsqueeze(-1).unsqueeze(-1) ** 2  # (B, 1, 1)
        diff = y_true_expanded - mu  # (B, K, d)
        
        # log N(y | mu, sigma^2 I) = -d/2 log(2π) - d/2 log(sigma^2) - 1/(2sigma^2) ||y - mu||^2
        log_prob_components = (
            -0.5 * d * math.log(2 * math.pi)
            - 0.5 * d * torch.log(sigma).unsqueeze(-1)  # (B, 1) -> broadcasts to (B, K)
            - 0.5 * (diff ** 2).sum(dim=-1) / variance.squeeze(-1)  # (B, K)
        )
    
    elif variance_mode == "isotropic":
        # One variance per component
        variance = sigma.unsqueeze(-1) ** 2  # (B, K, 1)
        diff = y_true_expanded - mu  # (B, K, d)
        
        log_prob_components = (
            -0.5 * d * math.log(2 * math.pi)
            - 0.5 * d * torch.log(variance.squeeze(-1))
            - 0.5 * (diff ** 2).sum(dim=-1) / variance.squeeze(-1)  # (B, K)
        )
    
    elif variance_mode == "diagonal":
        # Full diagonal covariance
        variance = sigma ** 2  # (B, K, d)
        diff = y_true_expanded - mu  # (B, K, d)
        
        log_prob_components = (
            -0.5 * d * math.log(2 * math.pi)
            - 0.5 * torch.log(variance).sum(dim=-1)
            - 0.5 * ((diff ** 2) / variance).sum(dim=-1)  # (B, K)
        )
    
    else:  # fixed
        variance = 0.01
        diff = y_true_expanded - mu  # (B, K, d)
        
        log_prob_components = (
            -0.5 * d * math.log(2 * math.pi)
            - 0.5 * d * math.log(variance)
            - 0.5 * (diff ** 2).sum(dim=-1) / variance  # (B, K)
        )
    
    # Weight by mixture coefficients: log(sum_k pi_k * N(...))
    # = logsumexp(log(pi_k) + log(N(...)))
    log_pi = torch.log(pi + 1e-8)  # (B, K)
    log_weighted_probs = log_pi + log_prob_components  # (B, K)
    
    # LogSumExp for numerical stability
    log_prob_mixture = torch.logsumexp(log_weighted_probs, dim=1)  # (B,)
    
    # Negative log-likelihood
    nll = -log_prob_mixture.mean()
    
    return nll


def test_smsd():
    """Test SMSD module"""
    print("Testing SMSD Module...")
    
    # Initialize SMSD
    smsd = SMSD(
        bert_dim=768,
        style_dim=256,  # Match FACodec output
        num_mixtures=5,
        hidden_dim=512,
        variance_mode="isotropic_across_clusters"
    )
    
    # Test style text descriptions
    style_texts = [
        "speak in a fast and energetic voice",
        "use a slow and calm speaking style",
        "sound cheerful and happy",
    ]
    
    # Simulate ground truth style vectors from FACodec
    B = len(style_texts)
    y_true = torch.randn(B, 256)  # (B, style_dim)
    
    print("\n1. Testing Training Mode:")
    smsd.train()
    loss = smsd(style_texts, y_true=y_true)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n2. Testing Inference Mode:")
    smsd.eval()
    with torch.no_grad():
        y_sampled = smsd(style_texts)
    print(f"   Sampled style shape: {y_sampled.shape}")
    print(f"   Expected: torch.Size([{B}, 256])")
    
    print("\n3. Testing with mixture parameters:")
    with torch.no_grad():
        y_sampled, (pi, mu, sigma) = smsd(style_texts, return_params=True)
    print(f"   Mixture weights shape: {pi.shape}")
    print(f"   Means shape: {mu.shape}")
    print(f"   Sigma shape: {sigma.shape}")
    print(f"   Mixture weights sum to 1: {pi.sum(dim=1)}")
    
    print("\nSMSD module test passed!")


if __name__ == "__main__":
    test_smsd()


