import torch
import transformers
from transformers import ViTModel
import torch
from torch import nn
import transformers


class EEGViT_pretrained_hierachical(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(1, 18), stride=(1, 18), padding=(0, 2), bias=False)
        self.conv2 = nn.Conv2d(1, 256, kernel_size=(1, 36), stride=(1, 18), padding=(0, 11), bias=False)
        self.conv3 = nn.Conv2d(1, 256, kernel_size=(1, 72), stride=(1, 18), padding=(0, 29), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(768, False)
        model_name = "google/vit-base-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 768})
        config.update({'image_size': (129, 28)})
        config.update({'patch_size': (129, 1)})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(768, 768, kernel_size=(129, 1),
                                                                           stride=(129, 1),
                                                                           padding=(0, 0), groups=768)
        model.classifier = torch.nn.Sequential(torch.nn.Linear(768, 1000, bias=True),
                                               torch.nn.Dropout(p=0.1),
                                               torch.nn.Linear(1000, 2, bias=True))
        self.ViT = model

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.batchnorm1(x)
        x = self.ViT.forward(x).logits

        return x


# Instantiate the model
model = EEGViT_pretrained_hierachical()

# Create a dummy input tensor
batch_size = 1
input_tensor = torch.randn(batch_size, 1, 129, 500)  # Using random values

# Forward pass
positions = model(input_tensor)

# Print output shapes to verify
print("Positions Shape:", positions.shape)
