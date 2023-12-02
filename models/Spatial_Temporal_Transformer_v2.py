import torch
import torch.nn as nn
import numpy as np


class Spatial_Temporal_Transformer_v2(nn.Module):
    def __init__(self, input_dim=129, output_dim=2, d_model=512, num_heads=8, num_encoder_layers=6,
                 num_decoder_layers=6):
        super(Spatial_Temporal_Transformer_v2, self).__init__()

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim))

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
        cls_token = self.cls_token.repeat(x.size(0), 1, 1)
        x = x.squeeze(1).transpose(0, 2).transpose(1, 2)
        x = torch.cat([cls_token, x], dim=0)
        x = self.input_linear(x)

        # Encoder output
        x = self.encoder(x)

        # Branch 1: Predicting eye position
        cls_output = x[0, :, :]
        position_pred = self.predictor(cls_output)

        # Branch 2: Reconstructing input
        reconstructed_x = self.decoder(x, x)
        reconstructed_x = self.output_linear(reconstructed_x)
        reconstructed_x = reconstructed_x[1:, :, :].transpose(1, 2).transpose(0, 2).unsqueeze(1)

        return position_pred, reconstructed_x


if __name__ == '__main__':
    # Instantiate the model
    model = Spatial_Temporal_Transformer_v2()

    # Create a dummy input tensor
    batch_size = 1
    input_tensor = torch.randn(batch_size, 1, 129, 500)  # Using random values

    # Forward pass
    positions, x_reconstructed = model(input_tensor)

    # Print output shapes to verify
    print("Positions Shape:", positions.shape)
    print("Reconstructed Shape:", x_reconstructed.shape)
