import torch
import transformers
from transformers import ViTModel
import torch
from torch import nn
import transformers


class MTLDeiT_pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=256,
            kernel_size=(1, 32),
            stride=(1, 32),
            padding=(0, 6),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)

        model_name = "facebook/deit-base-distilled-patch16-224"
        config = transformers.DeiTConfig()
        config.update({'num_channels': 256})
        config.update({'image_size': (129, 16)})
        config.update({'patch_size': (8, 1)})
        config.update({'encoder_stride': 1})

        model = transformers.DeiTForMaskedImageModeling.from_pretrained(model_name, config=config,
                                                                        ignore_mismatched_sizes=True)
        model.deit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(8, 1), stride=(8, 1),
                                                                            padding=(0, 0), groups=256)

        self.classifier = torch.nn.Sequential(torch.nn.Linear(768, 1000, bias=True),
                                              torch.nn.Dropout(p=0.1),
                                              torch.nn.Linear(1000, 2, bias=True))
        self.DeiT = model

        # Reconstruction Branch
        self.spatial_deconv = nn.ConvTranspose2d(256, 256, kernel_size=(8, 1), stride=(8, 1), padding=(0, 0),
                                                 output_padding=(1, 0))
        self.temporal_deconv = nn.ConvTranspose2d(256, 1, kernel_size=(1, 32), stride=(1, 32), padding=(0, 0))

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        outputs = self.DeiT(x)

        # Position
        sequence_output = self.DeiT.deit(x)[0]
        position = self.classifier(sequence_output[:, 0, :])

        # Reconstruction
        reconstruction = self.spatial_deconv(outputs.reconstruction)
        reconstruction = self.temporal_deconv(reconstruction)

        return position, reconstruction[:, :, :, 6:-6]


# Instantiate the model
model = MTLDeiT_pretrained()

# Create a dummy input tensor
batch_size = 1
input_tensor = torch.randn(batch_size, 1, 129, 500)  # Using random values

# Forward pass
positions, reconstruction = model(input_tensor)

# Print output shapes to verify
print("Positions Shape:", positions.shape)
print("Reconstructed Shape:", reconstruction.shape)

# 1, 256, 16, 16 -> 1, 256, 256, 256 ????
