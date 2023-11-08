import torch
import transformers
from transformers import ViTModel
import torch
from torch import nn
import transformers


class DeiT_pretrained(nn.Module):
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

        model_name = "facebook/deit-base-distilled-patch16-224"
        config = transformers.DeiTConfig()
        config.update({'num_channels': 256})
        config.update({'image_size': (129, 14)})
        config.update({'patch_size': (8, 1)})

        model = transformers.DeiTForImageClassification.from_pretrained(model_name, config=config,
                                                                        ignore_mismatched_sizes=True)
        model.deit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(8, 1), stride=(8, 1),
                                                                            padding=(0, 0), groups=256)
        model.classifier = torch.nn.Sequential(torch.nn.Linear(768, 1000, bias=True),
                                               torch.nn.Dropout(p=0.1),
                                               torch.nn.Linear(1000, 2, bias=True))
        self.DeiT = model

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.DeiT(x).logits

        return x


if __name__ == '__main__':
    # Instantiate the model
    model = DeiT_pretrained()

    # Create a dummy input tensor
    batch_size = 1
    input_tensor = torch.randn(batch_size, 1, 129, 500)  # Using random values

    # Forward pass
    positions = model(input_tensor)

    # Print output shapes to verify
    print("Positions Shape:", positions.shape)
