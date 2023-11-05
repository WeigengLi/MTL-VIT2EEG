import torch
import transformers
from torch import nn


class MTLViT_pretrained(nn.Module):
    def __init__(self):
        super().__init__()

        # Shared Features Extraction
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
        model.classifier=torch.nn.Sequential(torch.nn.Linear(768,1000,bias=True),
                                     torch.nn.Dropout(p=0.1),
                                     torch.nn.Linear(1000,2,bias=True))
        self.ViT_model = model
        self.ViT = model.vit  # Only take the ViT part without the classification head


        # Reconstruction Branch
        self.spatial_deconv = nn.ConvTranspose2d(768, 128, kernel_size=(8, 1), stride=(8, 1), padding=(0, 0),
                                                 output_padding=(1, 0))
        self.temporal_deconv = nn.ConvTranspose2d(128, 1, kernel_size=(1, 36), stride=(1, 36), padding=(0, 0))

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        positions=self.ViT_model.forward(x).logits
        shared_features = self.ViT(x).last_hidden_state  # Extracting the shared features
        # Position Prediction

        # Discard the [CLS] token and reshape
        reshaped_features = shared_features[:, 1:].view(x.size(0), 768, 16, 14)

        # Reconstruction
        x_reconstructed = self.spatial_deconv(reshaped_features)
        x_reconstructed = self.temporal_deconv(x_reconstructed)

        return positions, x_reconstructed[:, :, :, 2:-2]



model = MTLViT_pretrained()