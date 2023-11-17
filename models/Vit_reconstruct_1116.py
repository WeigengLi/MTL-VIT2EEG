import torch
import transformers
from torch import nn


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args
        self.dec_block1 = nn.Sequential(
            nn.ConvTranspose2d(768, 256, (3, 3)),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        )

        self.dec_block2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, (3, 3)),
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        )

        self.dec_block3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (3, 3)),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )

        self.dec_block4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, (3, 3), stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(True)
        )

        self.dec_block5 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, (3, 3), stride=2, padding=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(True)
        )

        self.dec_block6 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, (3, 3), stride=2, padding=1),
            nn.InstanceNorm2d(3),
            nn.ReLU(True)
        )

        self.up = nn.UpsamplingBilinear2d((args.image_size, args.image_size))  # fixed output size
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = x[:, 1:, :]
        out = out.transpose(1, 2)
        out = out.reshape(x.shape[0], -1, 16, 14)
        out = self.dec_block1(out)
        out = self.dec_block2(out)
        out = self.dec_block3(out)
        out = self.dec_block4(out)
        out = self.dec_block5(out)
        out = self.dec_block6(out)
        out = self.up(out)
        out = self.tanh(out)
        return out


class ViT_reconstruct(nn.Module):
    def __init__(self, reconstruct_kernel_size=(3, 3)):
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
        model_name = "google/vit-base-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (129, 14)})
        config.update({'patch_size': (8, 1)})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(8, 1), stride=(8, 1),
                                                                           padding=(0, 0), groups=256)
        model.classifier = torch.nn.Sequential(torch.nn.Linear(768, 1000, bias=True),
                                               torch.nn.Dropout(p=0.1),
                                               torch.nn.Linear(1000, 2, bias=True))
        self.ViT = model

        # Decoder
        self.dec_block1 = nn.Sequential(
            nn.ConvTranspose2d(768, 256, reconstruct_kernel_size),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        )

        self.dec_block2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, reconstruct_kernel_size),
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        )

        self.dec_block3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, reconstruct_kernel_size),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )

        self.dec_block4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, reconstruct_kernel_size, stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(True)
        )

        self.dec_block5 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, reconstruct_kernel_size, stride=2, padding=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(True)
        )

        self.dec_block6 = nn.Sequential(
            nn.ConvTranspose2d(16, 1, reconstruct_kernel_size, stride=2, padding=1),
            nn.InstanceNorm2d(3),
            nn.ReLU(True)
        )

        self.up = nn.UpsamplingBilinear2d((129, 500))  # fixed output size
        self.tanh = nn.Tanh()

        # Reconstruction Branch
        self.spatial_deconv = nn.ConvTranspose2d(768, 256, kernel_size=(8, 1), stride=(8, 1), padding=(0, 0),
                                                 output_padding=(1, 0))
        self.temporal_deconv = nn.ConvTranspose2d(256, 1, kernel_size=(1, 36), stride=(1, 36), padding=(0, 0))

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        output = self.ViT.forward(x, output_hidden_states=True)
        positions = output.logits

        # Extracting the shared features
        shared_features = output.hidden_states[-1]

        # # Discard the [CLS] token and reshape
        # reshaped_features = shared_features[:, 1:].view(x.size(0), 768, 16, 14)

        # Decoder
        reshaped_features = shared_features[:, 1:, :].transpose(1, 2).reshape(shared_features.shape[0], -1, 16, 14)
        x_reconstructed = self.dec_block1(reshaped_features)
        x_reconstructed = self.dec_block2(x_reconstructed)
        x_reconstructed = self.dec_block3(x_reconstructed)
        x_reconstructed = self.dec_block4(x_reconstructed)
        x_reconstructed = self.dec_block5(x_reconstructed)
        x_reconstructed = self.dec_block6(x_reconstructed)
        x_reconstructed = self.up(x_reconstructed)
        x_reconstructed = self.tanh(x_reconstructed)

        return positions, x_reconstructed


if __name__ == '__main__':
    # Instantiate the model
    model = ViT_reconstruct((3, 3))

    # Create a dummy input tensor
    batch_size = 1
    input_tensor = torch.randn(batch_size, 1, 129, 500)  # Using random values

    # Forward pass
    positions, x_reconstructed = model(input_tensor)

    # Print output shapes to verify
    print("Positions Shape:", positions.shape)
    print("Reconstructed Shape:", x_reconstructed.shape)
