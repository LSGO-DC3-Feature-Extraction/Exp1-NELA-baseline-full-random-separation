import torch
import torch.nn.functional as F
from torch import nn
import math


    
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        i = torch.arange(0, dim, 2).float()
        pe[:, 0::2] = torch.sin(pos / (10000 ** (i / dim)))
        pe[:, 1::2] = torch.cos(pos / (10000 ** (i / dim)))
        self.register_buffer('pe', pe)

    def forward(self, seq_len):
        return self.pe[:seq_len]

class MultiHeadEncoder(nn.Module):
    def __init__(self, n_heads, embed_dim, ff_hidden):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x

class EmbeddingNet(nn.Module):
    def __init__(self, node_dim, embedding_dim):
        super().__init__()
        self.linear = nn.Linear(node_dim, embedding_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class FeatureExtractor(nn.Module):
    def __init__(self,
                 node_dim=2,
                 hidden_dim=16,
                 max_feat_dim=1000,
                 n_heads=1,
                 ffh=16,
                 n_layers=1,
                 use_pe=True,
                 is_mlp=False):
        super().__init__()
        self.embed = EmbeddingNet(node_dim, hidden_dim)
        self.is_mlp = is_mlp

        if is_mlp:
            self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        else:
            self.use_pe = use_pe
            self.pe = PositionalEncoding(hidden_dim, max_feat_dim) if use_pe else None
            self.dim_encoder = nn.Sequential(
                *[MultiHeadEncoder(n_heads, hidden_dim, ffh) for _ in range(n_layers)]
            )
            self.ind_encoder = nn.Sequential(
                *[MultiHeadEncoder(n_heads, hidden_dim, ffh) for _ in range(n_layers)]
            )

    def forward(self, xs, ys):
        xs = xs.unsqueeze(0)
        ys = (ys - ys.min()) / (ys.max() - ys.min() + 1e-12)
        ys = ys.unsqueeze(0).unsqueeze(-1)
        feature = torch.cat([xs.unsqueeze(-1), ys.repeat(1,1,xs.shape[1],1)], dim=-1)
        feature = feature.permute(0,2,1,3) 
        h = self.embed(feature.float())

        if self.is_mlp:
            out = self.mlp(h).mean(dim=-3)
        else:
            h = self.dim_encoder(h.view(-1, h.shape[2], h.shape[3])).view(*h.shape)
            o = h.permute(0,2,1,3).reshape(-1, h.shape[1], h.shape[3])
            if self.use_pe:
                o = o + self.pe(h.shape[1]) * 0.5
            out = self.ind_encoder(o).view(xs.shape[0], xs.shape[1], xs.shape[2], -1).mean(dim=-2)
        return out

    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.parameters())
        return total_num