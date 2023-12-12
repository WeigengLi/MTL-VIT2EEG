import torch
from transformers import ViTConfig, ViTForImageClassification
from torch import nn

'''
changes:
ViT encoder + Transformer Decoder
'''


class MTLT_v14(nn.Module):
    def __init__(self, num_layers=6):
        super().__init__()

        # Using a pre-trained ViT model as the encoder
        config = ViTConfig.from_pretrained("google/vit-base-patch16-224")
        config.num_channels = 1
        config.image_size = (129, 500)
        config.patch_size = (129, 1)
        config.num_hidden_layers = num_layers

        self.ViT = ViTForImageClassification(config=config)
        self.ViT.classifier = torch.nn.Sequential(torch.nn.Linear(768, 1000, bias=True),
                                                  torch.nn.Dropout(p=0.1),
                                                  torch.nn.Linear(1000, 2, bias=True))

        # Decoder part of the Transformer
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=12, batch_first=True), num_layers=num_layers)

        # Linear layer to transform decoder output back to input dimension
        self.output_linear = nn.Linear(config.hidden_size, 129)

    def forward(self, x):
        # Encoder output using ViT
        output = self.ViT.forward(x, output_hidden_states=True)

        # Extracting the shared features
        shared_features = output.hidden_states[-1][:, 1:, :]

        # Branch 1: Predicting eye position
        positions = output.logits

        # Branch 2: Reconstructing input
        x_reconstructed = self.decoder(shared_features, shared_features)
        x_reconstructed = self.output_linear(x_reconstructed)
        x_reconstructed = x_reconstructed.transpose(1, 2).unsqueeze(1)

        return positions, x_reconstructed, shared_features[:, 0]


if __name__ == '__main__':
    # Instantiate the model
    model = MTLT_v14()

    # Create a dummy input tensor
    batch_size = 1
    input_tensor = torch.randn(batch_size, 1, 129, 500)  # Using random values

    # Forward pass
    positions, x_reconstructed, _ = model(input_tensor)

    # Print output shapes to verify
    print("Positions Shape:", positions.shape)
    print("Reconstructed Shape:", x_reconstructed.shape)
