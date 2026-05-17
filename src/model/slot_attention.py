# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
# Slot Attention module adapted from VQ-VFM-OCL project
# https://github.com/Genera1Z/VQ-VFM-OCL

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Simple MLP with GELU activation"""
    def __init__(self, in_dim, hidden_dims, out_dim=None, dropout=0.0):
        super().__init__()
        dims = [in_dim] + hidden_dims
        if out_dim is not None:
            dims.append(out_dim)

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class SlotAttention(nn.Module):
    """
    Slot Attention module from "Object-Centric Learning with Slot Attention"
    Adapted from VQ-VFM-OCL project
    """
    def __init__(
        self,
        num_iter,
        embed_dim,
        ffn_dim,
        dropout=0.0,
        kv_dim=None,
        trunc_bp=None
    ):
        """
        Args:
            num_iter: Number of slot attention iterations
            embed_dim: Dimension of slot embeddings
            ffn_dim: Hidden dimension of FFN
            dropout: Dropout rate (only in FFN)
            kv_dim: Dimension of key/value (input features), defaults to embed_dim
            trunc_bp: Truncated backprop strategy, "bi-level" or None
        """
        super().__init__()
        kv_dim = kv_dim or embed_dim
        assert trunc_bp in ["bi-level", None]

        self.num_iter = num_iter
        self.trunc_bp = trunc_bp

        # Query projection (for slots)
        self.norm1q = nn.LayerNorm(embed_dim)
        self.proj_q = nn.Linear(embed_dim, embed_dim, bias=False)

        # Key/Value projection (for input features)
        self.norm1kv = nn.LayerNorm(kv_dim)
        self.proj_k = nn.Linear(kv_dim, embed_dim, bias=False)
        self.proj_v = nn.Linear(kv_dim, embed_dim, bias=False)

        # GRU for slot update
        self.rnn = nn.GRUCell(embed_dim, embed_dim)

        # FFN for slot refinement
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = MLP(embed_dim, [ffn_dim, embed_dim], None, dropout)

    def forward(self, input, query, smask=None, input_mask=None,num_iter=None):
        """
        Args:
            input: Input features, shape (B, H*W, C)
            query: Initial slot queries, shape (B, num_slots, C)
            smask: Slot mask, shape (B, num_slots), dtype=bool. True means valid slot.
            num_iter: Override number of iterations

        Returns:
            slots: Updated slots, shape (B, num_slots, C)
            attn: Attention weights, shape (B, num_slots, H*W)
        """
        B, num_slots, C = query.shape
        self_num_iter = num_iter or self.num_iter

        # Project input to key/value
        kv = self.norm1kv(input)
        k = self.proj_k(kv)  # (B, H*W, C)
        v = self.proj_v(kv)  # (B, H*W, C)
        if input_mask is not None:
            v = v*input_mask.unsqueeze(-1)# 前面不是已经unsqueeze了吗？
            k = k*input_mask.unsqueeze(-1)

        # Iterative slot attention
        slots = query
        for iter_idx in range(self_num_iter):
            # Bi-level optimization trick (optional)
            if iter_idx + 1 == self_num_iter and self.trunc_bp == "bi-level":
                slots = slots.detach() + query - query.detach()

            slots_prev = slots

            # Normalize and project slots to queries
            slots_norm = self.norm1q(slots)
            q = self.proj_q(slots_norm)  # (B, num_slots, C)

            # Inverted dot-product attention
            updates, attn = self.inverted_scaled_dot_product_attention(
                q, k, v, smask
            )

            # Update slots with GRU
            slots = self.rnn(
                updates.flatten(0, 1),#    更新门
                slots_prev.flatten(0, 1)#    遗忘门
            ).view(B, num_slots, -1)

            # Refine with FFN
            slots = slots + self.ffn(self.norm2(slots))

        return slots, attn

    @staticmethod
    def inverted_scaled_dot_product_attention(q, k, v, smask=None, eps=1e-5):
        """
        Inverted attention: softmax over slots (queries) instead of keys

        Args:
            q: Queries (slots), shape (B, num_slots, C)
            k: Keys (features), shape (B, H*W, C)
            v: Values (features), shape (B, H*W, C)
            smask: Slot mask, shape (B, num_slots, 1)

        Returns:
            o: Aggregated values, shape (B, num_slots, C)
            a0: Attention weights, shape (B, num_slots, H*W)
        """
        scale = q.size(2) ** -0.5  # Temperature

        # Compute attention logits: (B, num_slots, H*W)
        logit = torch.einsum("bqc,bkc->bqk", q * scale, k)

        # Apply slot mask if provided
        if smask is not None:
            logit = logit.masked_fill(~smask[:, :, None], float('-inf'))

        # Softmax over slots (inverted attention)
        a0 = logit.softmax(dim=1)  # (B, num_slots, H*W)

        # Re-normalize over spatial dimension
        a = a0 / (a0.sum(dim=2, keepdim=True) + eps)

        # Aggregate values
        o = torch.einsum("bqv,bvc->bqc", a, v)

        return o, a0

class SlotInitializer(nn.Module):
    """
    Initialize slot queries from input features
    """
    def __init__(self, num_slots, slot_dim, input_dim):
        """
        Args:
            num_slots: Number of slots
            slot_dim: Dimension of each slot
            input_dim: Dimension of input features
        """
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        # Learnable slot parameters (Gaussian distribution)
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, num_slots, slot_dim))

        # Project input features to slot dimension (for conditioning)
        self.input_proj = nn.Linear(input_dim, slot_dim)

        # Layer norm for stability
        self.norm = nn.LayerNorm(slot_dim)

        # Initialize parameters
        nn.init.xavier_uniform_(self.slots_mu)
        nn.init.zeros_(self.slots_log_sigma)

    def forward(self, feat, use_mean=False):
        """
        Args:
            feat: Input features, shape (B, H*W, C)
            use_mean: If True, use mean instead of sampling (for inference)

        Returns:
            slots: Initial slot queries, shape (B, num_slots, slot_dim)
        """
        B = feat.shape[0]

        # Expand learnable parameters
        mu = self.slots_mu.expand(B, -1, -1)
        sigma = self.slots_log_sigma.exp().expand(B, -1, -1)

        # Sample from Gaussian (or use mean for inference)
        if use_mean or not self.training:
            slots = mu
        else:
            slots = mu + sigma * torch.randn_like(mu)

        # Optional: condition on input features (simple version)
        # You can make this more sophisticated
        feat_pooled = feat.mean(dim=1, keepdim=True)  # (B, 1, C)
        feat_proj = self.input_proj(feat_pooled)  # (B, 1, slot_dim)

        # Add conditioning to slots
        slots = slots + 0.1 * feat_proj  # Small weight to preserve learned init

        # Normalize
        slots = self.norm(slots)

        return slots

class SlotAggregator(nn.Module):
    """
    Complete slot aggregation module combining initialization and attention
    """
    def __init__(
        self,
        num_slots=7,
        slot_dim=768,
        input_dim=516,  # 512 (VAE mid features) + 4 (rgb_latent)
        num_iter=3,
        ffn_dim=2048,
        dropout=0.0,
    ):
        """
        Args:
            num_slots: Number of object slots
            slot_dim: Dimension of slot embeddings (should match text_embed dim)
            input_dim: Dimension of input features from UNet
            num_iter: Number of slot attention iterations
            ffn_dim: Hidden dimension of FFN in slot attention
            dropout: Dropout rate
        """
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        self.slot_init = SlotInitializer(num_slots, slot_dim, input_dim)
        self.slot_attn = SlotAttention(
            num_iter=num_iter,
            embed_dim=slot_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
            kv_dim=input_dim,
            trunc_bp=None,  # Can use "bi-level" for faster training
        )

    def forward(self, feat,valid_mask=None):
        """
        Args:
            feat: Input features, shape (B, C, H, W) or (B, H*W, C)

        Returns:
            slots: Object slots, shape (B, num_slots, slot_dim)
            attn: Attention maps, shape (B, num_slots, H*W)
        """
        # Handle different input formats
        if feat.dim() == 4:  # (B, C, H, W)
            B, C, H, W = feat.shape
            feat = feat.flatten(2).permute(0, 2, 1)  # (B, H*W, C)

        if valid_mask is not None:
            valid_flat = valid_mask.flatten(1) # (B,H*W)
            feat = feat * valid_flat.unsqueeze(-1) # mask padding -> zero vector
        else:
            valid_flat = None
        # Initialize slots
        queries = self.slot_init(feat)  # (B, num_slots, slot_dim)

        # Slot attention
        slots, attn = self.slot_attn(feat, queries,input_mask=valid_flat)  # (B, num_slots, slot_dim)

        return slots, attn

    @property
    def dtype(self):
        """
        Returns the dtype of the first parameter, compatible with diffusers Pipeline
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32

    @property
    def device(self):
        """
        Returns the device of the first parameter, compatible with diffusers Pipeline
        """
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cpu')
