import torch
import transformers
from torch import nn

'''
Inspired by:

@article{lee2022anovit,
  title={AnoViT: Unsupervised anomaly detection and localization with vision transformer-based encoder-decoder},
  author={Lee, Yunseung and Kang, Pilsung},
  journal={IEEE Access},
  volume={10},
  pages={46717--46724},
  year={2022},
  publisher={IEEE}
}

changes:
nothing changed compared to EEGViT
add decoder blocks
'''


class ViT_reconstruct_v6(nn.Module):
    def __init__(self):
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
            nn.ConvTranspose2d(768, 256, (1, 36)),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        )

        self.dec_block2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, (8, 1)),
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        )

        self.dec_block3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (1, 36)),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )

        self.dec_block4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, (1, 36), stride=(1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(32),
            nn.ReLU(True)
        )

        self.dec_block5 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, (8, 1), stride=(2, 1), padding=(2, 0)),
            nn.InstanceNorm2d(16),
            nn.ReLU(True)
        )

        self.dec_block6 = nn.Sequential(
            nn.ConvTranspose2d(16, 1, (1, 36), stride=(1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(1),
            nn.ReLU(True)
        )

        self.up = nn.UpsamplingBilinear2d((129, 500))  # fixed output size
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        output = self.ViT.forward(x, output_hidden_states=True)
        positions = output.logits

        # Extracting the shared features
        shared_features = output.hidden_states[-1]

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
    model = ViT_reconstruct_v6()

    # Create a dummy input tensor
    batch_size = 1
    input_tensor = torch.randn(batch_size, 1, 129, 500)  # Using random values

    # Forward pass
    positions, x_reconstructed = model(input_tensor)

    # Print output shapes to verify
    print("Positions Shape:", positions.shape)
    print("Reconstructed Shape:", x_reconstructed.shape)
