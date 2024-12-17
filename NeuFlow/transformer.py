import torch
import torch.nn as nn

class LinformerAttention(nn.Module):
    def __init__(self, dim, proj_dim):
        super().__init__()
        self.proj_dim = proj_dim
        self.dim = dim
        self.E = None  # Will be initialized based on input sequence length
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        batch_size, seq_len, _ = key.size()
        device = key.device

        # Initialize self.E if not already done or if sequence length changes
        if self.E is None or self.E.size(0) != seq_len:
            # It's important to register self.E as a buffer or parameter if it requires gradients
            self.E = nn.Parameter(torch.randn(seq_len, self.proj_dim, device=device), requires_grad=True)

        # Project key and value using self.E
        # Key projection
        key_proj = torch.einsum('bld,le->bed', key, self.E)  # key: (b, l, d), E: (l, e) -> key_proj: (b, e, d)
        # Value projection
        value_proj = torch.einsum('bld,le->bed', value, self.E)  # value: (b, l, d)

        # Compute attention scores
        scores = torch.einsum('bqd,bed->bqe', query, key_proj) / (self.dim ** 0.5)  # query: (b, q, d), key_proj: (b, e, d)
        attention_weights = self.softmax(scores)

        # Compute the output
        output = torch.einsum('bqe,bed->bqd', attention_weights, value_proj)  # output: (b, q, d)
        return output

class TransformerLayer(nn.Module):
    def __init__(self, feature_dim, proj_dim, ffn=True, ffn_dim_expansion=1):
        super().__init__()
        self.attention = LinformerAttention(feature_dim, proj_dim)
        self.norm1 = nn.LayerNorm(feature_dim)

        self.ffn = ffn
        if self.ffn:
            in_channels = feature_dim
            hidden_dim = in_channels * ffn_dim_expansion
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, in_channels),
            )
            self.norm2 = nn.LayerNorm(feature_dim)

    def forward(self, source, target):
        # source and target shapes: (batch_size, seq_len, feature_dim)
        message = self.attention(source, target, target)
        message = self.norm1(message + source)  # Residual connection

        if self.ffn:
            ffn_output = self.mlp(message)
            message = self.norm2(ffn_output + message)  # Residual connection

        return message

class FeatureAttention(nn.Module):
    def __init__(self, feature_dim, num_layers, ffn=True, ffn_dim_expansion=1, post_norm=True):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(
                feature_dim=feature_dim,
                proj_dim=feature_dim // 4,
                ffn=ffn,
                ffn_dim_expansion=ffn_dim_expansion
            )
            for _ in range(num_layers)
        ])

    def forward(self, concat_features0):

        b, c, h, w = concat_features0.shape

        concat_features0 = concat_features0.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]
        concat_features1 = torch.cat(concat_features0.chunk(chunks=2, dim=0)[::-1], dim=0)

        for layer in self.layers:
            concat_features0 = layer(concat_features0, concat_features1)
            concat_features1 = torch.cat(concat_features0.chunk(chunks=2, dim=0)[::-1], dim=0)

        # reshape back
        # ???
        concat_features0 = concat_features0.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        
        # if self.post_norm:
        #     concat_features0 = self.norm(concat_features0)

        return concat_features0


class FlowAttention(torch.nn.Module):
    def __init__(self, feature_dim, seq_len, k):
        super(FlowAttention, self).__init__()
        self.attention = LinformerAttention(feature_dim, seq_len, k)

    def forward(self, feature, flow):
        b, _, h, w = feature.size()

        feature = feature.flatten(-2).permute(0, 2, 1)
        flow = flow.flatten(-2).permute(0, 2, 1)

        flow = self.attention(feature, feature, flow)
        flow = flow.view(b, h * w, 2).permute(0, 2, 1).contiguous().view(b, 2, h, w)

        return flow

# Example test code
# batch_size = 4
# seq_len = 192  # Example sequence length
# feature_dim = 256
# proj_dim = seq_len // 4  # Or any suitable projection dimension

# source = torch.randn(batch_size, seq_len, feature_dim)
# target = torch.randn(batch_size, seq_len, feature_dim)

# # Initialize the layer
# transformer_layer = TransformerLayer(feature_dim=feature_dim, proj_dim=proj_dim)

# # Forward pass
# output = transformer_layer(source, target)

# print(output.shape)  # Should be (batch_size, seq_len, feature_dim)
