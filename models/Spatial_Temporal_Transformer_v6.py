import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

'''
    Classical Transformer with cos PE
    Spatial & Temporal Multi-Head Attention
'''


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, d_model=512, num_heads=8):
        super(MultiheadAttention, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"

        self.linear_keys = nn.Linear(input_dim, d_model)
        self.linear_values = nn.Linear(input_dim, d_model)
        self.linear_queries = nn.Linear(input_dim, d_model)
        self.final_linear = nn.Linear(d_model, d_model)

    def forward(self, key, value, query, mask=None):
        # Time, Batch_size, Channel
        T, B, C = query.shape

        # 1) Do all the linear projections in batch from d_model => num_heads * d_k
        key = self.linear_keys(key).view(B, -1, self.num_heads, self.head_dim)
        value = self.linear_values(value).view(B, -1, self.num_heads, self.head_dim)
        query = self.linear_queries(query).view(B, -1, self.num_heads, self.head_dim)

        # 2) Transpose for attention dot product: b x num_heads x seq_len x d_model
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query = query.transpose(1, 2)

        # 3) Calculate and scale scores.
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)

        # 4) Apply attention to values, concatenate and apply final linear
        x = torch.matmul(attention, value)
        x = x.transpose(1, 2).contiguous().view(T, B, self.d_model)
        return self.final_linear(x)


class EncoderLayer(nn.Module):
    def __init__(self, input_dim, d_model=512, num_heads=8):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(input_dim, d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x2 = self.layer_norm1(x)
        x2 = self.self_attn(x2, x2, x2)
        x = x + self.dropout(x2)
        x2 = self.layer_norm2(x)
        x = x + self.dropout(self.feed_forward(x2))
        return x


class Spatial_Temporal_Transformer_v6(nn.Module):
    def __init__(self, input_dim=129, max_len=500, output_dim=2, d_model=512, num_heads=8, num_encoder_layers=6,
                 num_decoder_layers=6):
        super(Spatial_Temporal_Transformer_v6, self).__init__()

        # Linear layer to transform input dimension
        self.temporal_linear = nn.Linear(input_dim, d_model)

        # Temporal Transformer Encoder
        self.temporal_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.temporal_encoder = nn.TransformerEncoder(self.temporal_encoder_layer, num_layers=num_encoder_layers)

        # Linear layer to transform input dimension
        self.spatial_linear = nn.Linear(max_len, d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=d_model)

        # Spatial Transformer Encoder
        self.spatial_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.spatial_encoder = nn.TransformerEncoder(self.spatial_encoder_layer, num_layers=num_encoder_layers)

        # # Linear layer to transform input dimension
        # self.input_linear = nn.Linear(input_dim, d_model)

        # # Encoder part of the Transformer
        # self.encoder = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads),
        #     num_layers=num_encoder_layers
        # )

        # Linear layer to transform spatial_encoder output back to input dimension
        self.spatial_output_linear = nn.Linear(d_model, max_len)

        # Decoder part of the Transformer
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads),
            num_layers=num_decoder_layers
        )

        # Predictor for eye position
        self.predictor = torch.nn.Sequential(torch.nn.Linear(d_model, d_model // 2, bias=True),
                                             torch.nn.Dropout(p=0.3),
                                             torch.nn.Linear(d_model // 2, d_model // 4, bias=True),
                                             torch.nn.Dropout(p=0.3),
                                             torch.nn.Linear(d_model // 4, output_dim, bias=True))

        # Linear layer to transform decoder output back to input dimension
        self.output_linear = nn.Linear(d_model, input_dim)

    def forward(self, x):
        # Preprocess input
        # Shape: [seq_len, batch_size, num_electrodes]
        x = x.squeeze(1).transpose(0, 2).transpose(1, 2)
        # Shape: [seq_len, batch_size, 512]
        x = self.temporal_linear(x)

        # Temporal Encoder
        # Shape: [seq_len, batch_size, 512]
        x = self.temporal_encoder(x)

        # Transpose for spatial dimension encoding
        x = x.transpose(0, 2)

        # Apply linear transformation for spatial encoding
        x = self.spatial_linear(x)

        # Apply Positional Encoding
        x = self.pos_encoder(x)

        # Apply Transformer Encoder to space (electrodes) dimension
        x = self.spatial_encoder(x)

        # Transpose back to original shape for decoder
        # Shape: [seq_len, batch_size, num_electrodes]
        x = self.spatial_output_linear(x)
        x = x.transpose(0, 2)

        # Branch 1: Predicting eye position
        position_pred = self.predictor(x[-1, :, :])

        # Branch 2: Reconstructing input
        reconstructed_x = self.decoder(x, x)
        reconstructed_x = self.output_linear(reconstructed_x)
        reconstructed_x = reconstructed_x.transpose(1, 2).transpose(0, 2).unsqueeze(1)

        return position_pred, reconstructed_x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        device = x.device
        encoding = self.encoding.transpose(0, 1).repeat(1, x.size(1), 1).to(device)
        return x + encoding


if __name__ == '__main__':
    # Instantiate the model
    model = Spatial_Temporal_Transformer_v6()

    # Create a dummy input tensor
    batch_size = 1
    input_tensor = torch.randn(batch_size, 1, 129, 500)  # Using random values

    # Forward pass
    positions, x_reconstructed = model(input_tensor)

    # Print output shapes to verify
    print("Positions Shape:", positions.shape)
    print("Reconstructed Shape:", x_reconstructed.shape)
