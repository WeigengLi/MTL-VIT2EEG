import torch
from transformers import ViTConfig, ViTForImageClassification
from torch import nn

'''
ViT encoder + Transformer Decoder
'''


class MTLT_v16(nn.Module):
    def __init__(self, num_decoder_layers=12):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=256,
            kernel_size=(1, 36),
            stride=(1, 36),
            padding=(0, 2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)

        # Using a pre-trained ViT model as the encoder
        config = ViTConfig.from_pretrained("google/vit-base-patch16-224")
        config.num_channels = 256
        config.image_size = (129, 14)
        config.patch_size = (8, 1)

        self.ViT = ViTForImageClassification(config=config)
        self.ViT.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(8, 1),
                                                                              stride=(8, 1), padding=(0, 0), groups=256)
        self.ViT.classifier = torch.nn.Sequential(torch.nn.Linear(768, 1000, bias=True),
                                                  torch.nn.Dropout(p=0.1),
                                                  torch.nn.Linear(1000, 2, bias=True))

        # Decoder part of the Transformer
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=12, batch_first=True),
            num_layers=num_decoder_layers)

        # Reduce channels
        self.output_conv = nn.Conv2d(config.hidden_size, 1, kernel_size=(1, 1))

        # Upsampling
        self.up = nn.UpsamplingBilinear2d((129, 500))  # fixed output size
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder output using ViT
        x = self.conv1(x)
        x = self.batchnorm1(x)
        output = self.ViT.forward(x, output_hidden_states=True)

        # Extracting the shared features
        shared_features = output.hidden_states[-1][:, 1:, :]

        # Branch 1: Predicting eye position
        positions = output.logits

        # Branch 2: Reconstructing input
        x_reconstructed = self.decoder(shared_features, shared_features)
        x_reconstructed = x_reconstructed.transpose(1, 2).reshape(x_reconstructed.shape[0], -1, 16, 14)
        x_reconstructed = self.output_conv(x_reconstructed)
        x_reconstructed = self.up(x_reconstructed)
        x_reconstructed = self.tanh(x_reconstructed)

        return positions, x_reconstructed, shared_features[:, 0]


if __name__ == '__main__':
    # Instantiate the model
    model = MTLT_v16()

    # Create a dummy input tensor
    batch_size = 1
    input_tensor = torch.randn(batch_size, 1, 129, 500)  # Using random values

    # Forward pass
    positions, x_reconstructed, _ = model(input_tensor)

    # Print output shapes to verify
    print("Positions Shape:", positions.shape)
    print("Reconstructed Shape:", x_reconstructed.shape)
