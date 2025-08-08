import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ff_dim)
        self.fc2 = nn.Linear(ff_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.norm(residual + x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,                          # (B, n, d_model)
        attn_bias: Optional[torch.Tensor] = None  # (B,H,n,n) or (B,n,n)
    ) -> torch.Tensor:
        B, n, _ = x.shape
        q = self.W_q(x).view(B, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, n, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** 0.5
        scores = torch.matmul(q, k.transpose(-1, -2)) / scale        # (B,H,n,n)

        if attn_bias is not None:
            if attn_bias.dim() == 3:                                 # (B,n,n) → (B,1,n,n)
                attn_bias = attn_bias.unsqueeze(1)
            scores = scores + attn_bias

        alpha = F.softmax(scores, dim=-1)
        out = torch.matmul(alpha, v)                                # (B,H,n,hd)

        out = out.transpose(1, 2).contiguous().view(B, n, self.d_model)
        out = self.out_proj(out)
        out = self.dropout(out)
        return self.norm(x + out)



class FactoredTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        bias_embed_dim: int,
        hidden_c: int
    ):
        super().__init__()
        self.num_heads = num_heads
        self.bias_embed_dim = bias_embed_dim

        self.temporal_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.temporal_ffn  = FeedForwardBlock(d_model, ff_dim, dropout)

        self.asset_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.asset_ffn  = FeedForwardBlock(d_model, ff_dim, dropout)

        self.query_token_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_c),
            nn.ReLU(),
            nn.Linear(hidden_c, num_heads * bias_embed_dim, bias=False)
        )

    def forward(
        self,
        x: torch.Tensor,                   # (B, H, D, d_model)
        beta_embeds: torch.Tensor,         # (B, H, D, D, num_heads*bias_E)
        gamma: torch.Tensor,               # scalar
        temporal_mask: torch.Tensor        # (H, H)
    ) -> torch.Tensor:

        B, H, D, M = x.shape

        # Step 11) time-axis attention (Asset ind)
        x_tmp = x.permute(0, 2, 1, 3).contiguous().view(B * D, H, M)
        x_tmp = self.temporal_attn(x_tmp, attn_bias=temporal_mask)
        x_tmp = self.temporal_ffn(x_tmp)
        x_after_temporal = x_tmp.view(B, D, H, M).permute(0, 2, 1, 3)

        # Step 2) Query‑dynamic asset-axis attention
        q_vec = self.query_token_mlp(x_after_temporal)             # (B,H,D,H*E)
        q_vec = q_vec.view(B, H, D, self.num_heads, self.bias_embed_dim)
        beta  = beta_embeds.view(B, H, D, D, self.num_heads, self.bias_embed_dim)
        
        dyn_bias = torch.einsum('btjhe,btjlhe->bthjl', q_vec, beta)
        asset_bias = gamma * dyn_bias                               # (B,H,heads,D,D) # gamma * dyn_bias     
        asset_bias = asset_bias.view(B * H, self.num_heads, D, D)

        x_ast = x_after_temporal.reshape(B * H, D, M)
        x_ast = self.asset_attn(x_ast, attn_bias=asset_bias)
        x_ast = self.asset_ffn(x_ast)
        return x_ast.view(B, H, D, M)



class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.d_asset        = configs.data_pool
        self.d_model        = configs.d_model
        self.dropout        = configs.dropout
        self.ff_dim         = configs.ff_dim
        self.num_heads      = configs.n_heads
        self.num_layers     = configs.num_layers
        self.sig_input_dim  = configs.sig_input_dim
        self.cross_sig_dim  = configs.cross_sig_dim
        self.hidden_c       = configs.hidden_c
        self.max_position   = configs.max_position
        self.bias_embed_dim = 16
        self.time_feat_dim  = configs.time_feat_dim      # 3 (default)

        self.sig_embed      = nn.Linear(self.sig_input_dim, self.d_model)
        self.asset_id_embed = nn.Embedding(self.d_asset, self.d_model)

        self.date_proj      = nn.Linear(self.time_feat_dim, self.d_model)

        self.concat_proj    = nn.Linear(3 * self.d_model, self.d_model)

        self.transformer_layers = nn.ModuleList([
            FactoredTransformerBlock(
                d_model       = self.d_model,
                num_heads     = self.num_heads,
                ff_dim        = self.ff_dim,
                dropout       = self.dropout,
                bias_embed_dim= self.bias_embed_dim,
                hidden_c      = self.hidden_c
            )
            for _ in range(self.num_layers)
        ])
        self.final_norm = nn.LayerNorm(self.d_model)

        self.beta_mlp = nn.Sequential(
            nn.Linear(self.cross_sig_dim, self.hidden_c),
            nn.ReLU(),
            nn.Linear(self.hidden_c, self.num_heads * self.bias_embed_dim, bias=False)
        )
        self._cross_gate_raw = nn.Parameter(torch.tensor(0.0))

        self.projection        = nn.Linear(self.d_model, 1, bias=True)
        self.return_projection = nn.Linear(self.d_model, 1, bias=True)

    def forward(
        self,
        x_sigs: torch.Tensor,          # (B, H, D, sig_in)
        cross_sigs: torch.Tensor,      # (B, H, D, D, cross_in)
        date_feats: torch.Tensor       # (B, H, F)
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, H, D, _ = x_sigs.shape
        device = x_sigs.device

        # First step 1) Signature embedding
        sig_emb = self.sig_embed(x_sigs)                          # (B,H,D,M)

        # Step 2) Calendar feature embedding -------------------------------
        date_emb = self.date_proj(date_feats)                     # (B,H,M)
        date_emb = date_emb.unsqueeze(2).expand(-1, -1, D, -1)    # (B,H,D,M)

        # Step 3) Asset embedding
        asset_ids = torch.arange(D, device=device)
        asset_emb = self.asset_id_embed(asset_ids).view(1, 1, D, -1) \
                                                  .expand(B, H, -1, -1)  # (B,H,D,M)

        # Step 4) Token concat & projection
        x = self.concat_proj(torch.cat([sig_emb, date_emb, asset_emb], dim=-1))

        # Step 5) gate embedding
        beta_embeds = self.beta_mlp(cross_sigs)                   # (B,H,D,D,H*E)
        gamma = F.softplus(self._cross_gate_raw)                 

        # Step 6) Causal mask 
        temporal_mask = torch.triu(
            torch.full((H, H), float('-inf'), device=device), diagonal=1)

        # Step 7. Factorized Transformer layers
        for layer in self.transformer_layers:
            x = layer(x, beta_embeds, gamma, temporal_mask)
        enc_out = self.final_norm(x)                              # (B,H,D,M)

        # Step 8) Outputs Note that (Do not updated delta)
        logits = self.projection(enc_out).squeeze(-1)             # (B,H,D)
        delta  = self.max_position * torch.tanh(logits)
        ret    = self.return_projection(enc_out).squeeze(-1)      # (B,H,D)
        return delta, ret
