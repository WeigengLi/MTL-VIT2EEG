import torch
import transformers
from torch import nn
import torch.nn.functional as F

'''
Inspired by:

@article{godoy2022eeg,
  title={Eeg-based epileptic seizure prediction using temporal multi-channel transformers},
  author={Godoy, Ricardo V and Reis, Tharik JS and Polegato, Paulo H and Lahr, Gustavo JG and Saute, Ricardo L and Nakano, Frederico N and Machado, Helio R and Sakamoto, Americo C and Becker, Marcelo and Caurin, Glauco AP},
  journal={arXiv preprint arXiv:2209.11172},
  year={2022}
}
'''


class ViT_conv(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN for Embedding Creation
        self.conv1 = nn.Conv2d(1, 64, (1, 20), bias=False)
        self.conv2 = nn.Conv2d(64, 128, (1, 20), bias=False)
        self.conv3 = nn.Conv2d(128, 256, (1, 10), bias=False)
        self.conv4 = nn.Conv2d(256, 256, (3, 3), bias=False)

        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)

        # Max-Pooling Layers
        self.pool1 = nn.MaxPool2d((1, 3))
        self.pool2 = nn.MaxPool2d((1, 3))
        self.pool3 = nn.MaxPool2d((1, 3))

        # Transformer
        model_name = "google/vit-base-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (127, 10)})
        config.update({'patch_size': (8, 1)})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(8, 1), stride=(8, 1),
                                                                           padding=(0, 0), groups=256)
        model.classifier = torch.nn.Sequential(torch.nn.Linear(768, 1000, bias=True),
                                               torch.nn.Dropout(p=0.1),
                                               torch.nn.Linear(1000, 2, bias=True))
        self.ViT = model

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))

        x = self.ViT.forward(x).logits

        return x


if __name__ == '__main__':
    # Instantiate the model
    model = ViT_conv()

    # Create a dummy input tensor
    batch_size = 1
    input_tensor = torch.randn(batch_size, 1, 129, 500)  # Using random values

    # Forward pass
    positions = model(input_tensor)

    # Print output shapes to verify
    print("Positions Shape:", positions.shape)
