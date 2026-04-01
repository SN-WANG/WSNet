import torch
from torch import nn, Tensor
from torch.nn import functional as F
from tqdm.auto import tqdm
from typing import List, Optional


class TimeProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1, 1)

    def forward(self, t_norm: Tensor) -> Tensor:
        t_norm = t_norm.to(dtype=self.proj.weight.dtype)
        return self.proj(t_norm.view(-1, 1))


class Attention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def _reshape_heads(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, query: Tensor, context: Optional[Tensor] = None) -> Tensor:
        if context is None:
            context = query

        batch_size, query_len, _ = query.shape

        query = self._reshape_heads(self.query_proj(query))
        key = self._reshape_heads(self.key_proj(context))
        value = self._reshape_heads(self.value_proj(context))

        output = F.scaled_dot_product_attention(
            query, key, value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, query_len, self.hidden_dim)
        return self.out_proj(output)


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, ffn_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        inner_dim = max(1, int(round(hidden_dim * ffn_ratio)))
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class HyperFlowBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ffn_ratio: float, dropout: float):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = Attention(hidden_dim, num_heads, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, ffn_ratio=ffn_ratio, dropout=dropout)

    def forward(self, latent_tokens: Tensor) -> Tensor:
        latent_tokens = latent_tokens + self.attn(self.attn_norm(latent_tokens))
        latent_tokens = latent_tokens + self.ffn(self.ffn_norm(latent_tokens))
        return latent_tokens


class HyperFlowNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dim: int,
        width: int = 128,
        depth: int = 4,
        num_heads: int = 8,
        num_latents: int = 64,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_dim = spatial_dim
        self.width = width

        self.time_encoder = TimeProjector()
        self.node_stem = nn.Linear(in_channels + spatial_dim + 1, width)
        self.node_dropout = nn.Dropout(dropout)

        self.latents = nn.Parameter(torch.empty(num_latents, width))
        nn.init.normal_(self.latents, mean=0.0, std=width ** -0.5)

        self.encode_latent_norm = nn.LayerNorm(width)
        self.encode_node_norm = nn.LayerNorm(width)
        self.encode_attention = Attention(width, num_heads, dropout=dropout)

        self.blocks = nn.ModuleList([
            HyperFlowBlock(width, num_heads, ffn_ratio=ffn_ratio, dropout=dropout)
            for _ in range(depth)
        ])

        self.decode_node_norm = nn.LayerNorm(width)
        self.decode_latent_norm = nn.LayerNorm(width)
        self.decode_attention = Attention(width, num_heads, dropout=dropout)

        self.node_ffn_norm = nn.LayerNorm(width)
        self.node_ffn = FeedForward(width, ffn_ratio=ffn_ratio, dropout=dropout)

        self.output_norm = nn.LayerNorm(width)
        self.output_head = nn.Linear(width, out_channels)

    def forward(
        self,
        input_features: Tensor,
        physical_coords: Tensor,
        t_norm: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        del kwargs

        batch_size, num_nodes, _ = physical_coords.shape
        node_dtype = self.node_stem.weight.dtype

        if t_norm is None:
            t_norm = physical_coords.new_zeros(batch_size)
        else:
            t_norm = t_norm.to(device=physical_coords.device)

        input_features = input_features.to(dtype=node_dtype)
        physical_coords = physical_coords.to(dtype=node_dtype)
        t_norm = t_norm.to(dtype=node_dtype)

        time_tokens = self.time_encoder(t_norm).unsqueeze(1).expand(-1, num_nodes, -1)
        node_inputs = torch.cat([input_features, physical_coords, time_tokens], dim=-1)
        node_tokens = self.node_dropout(self.node_stem(node_inputs))

        latent_tokens = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        latent_tokens = latent_tokens + self.encode_attention(
            self.encode_latent_norm(latent_tokens),
            self.encode_node_norm(node_tokens),
        )

        for block in self.blocks:
            latent_tokens = block(latent_tokens)

        node_tokens = node_tokens + self.decode_attention(
            self.decode_node_norm(node_tokens),
            self.decode_latent_norm(latent_tokens),
        )
        node_tokens = node_tokens + self.node_ffn(self.node_ffn_norm(node_tokens))

        return self.output_head(self.output_norm(node_tokens))

    def predict(self, initial_state: Tensor, coords: Tensor, steps: int) -> Tensor:
        device = next(self.parameters()).device
        batch_size = initial_state.shape[0]
        current_state = initial_state.to(device)
        coords = coords.to(device)

        sequence: List[Tensor] = [current_state.cpu()]

        with torch.no_grad():
            for step in tqdm(range(steps), desc="Predicting", leave=False, dynamic_ncols=True):
                t_norm = torch.full(
                    (batch_size,),
                    step / max(steps, 1),
                    device=device,
                    dtype=coords.dtype,
                )
                next_state = self.forward(current_state, coords, t_norm=t_norm)
                sequence.append(next_state.cpu())
                current_state = next_state

        return torch.stack(sequence, dim=1)
