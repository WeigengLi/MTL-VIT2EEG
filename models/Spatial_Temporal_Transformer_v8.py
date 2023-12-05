import torch
import torch.nn as nn
import numpy as np

'''
    Classical Transformer with cos PE
'''


class Spatial_Temporal_Transformer_v8(nn.Module):
    def __init__(self, input_dim=129, max_len=500, output_dim=2, d_model=512, num_heads=8, num_encoder_layers=6,
                 num_decoder_layers=6):
        super(Spatial_Temporal_Transformer_v8, self).__init__()

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        # Linear layer to transform input dimension
        self.input_linear = nn.Linear(input_dim, d_model)

        # Encoder part of the Transformer
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads),
            num_layers=num_encoder_layers
        )

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
        x = x.squeeze(1).transpose(0, 2).transpose(1, 2)
        x = self.input_linear(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Encoder output
        x = self.encoder(x)

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


class Discriminator(nn.Module):
    def __init__(self, input_dim=129, d_model=512):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * d_model, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        return self.net(x)


if __name__ == '__main__':
    # Instantiate the model
    model = Spatial_Temporal_Transformer_v8()

    # Create a dummy input tensor
    batch_size = 1
    input_tensor = torch.randn(batch_size, 1, 129, 500)  # Using random values

    # Forward pass
    positions, x_reconstructed = model(input_tensor)

    # Print output shapes to verify
    print("Positions Shape:", positions.shape)
    print("Reconstructed Shape:", x_reconstructed.shape)
