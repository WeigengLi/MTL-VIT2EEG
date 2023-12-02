import torch
import transformers
from torch import nn

'''
Inspired by:

Baseline 55.4

changes:
nothing changed compared to EEGViT
'''


class ViT_STL_v2(nn.Module):
    def __init__(self):
        super().__init__()
        model_name = "google/vit-base-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 1})
        config.update({'image_size': (129, 500)})
        config.update({'patch_size': (1, 36)})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        # model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(1, 768, kernel_size=(1, 36), stride=(1, 36),
        #                                                                    padding=(0, 2), groups=1)
        self.ViT1 = model.vit

        model_name = "google/vit-base-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 768})
        config.update({'image_size': (129, 13)})
        config.update({'patch_size': (8, 1)})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(768, 768, kernel_size=(8, 1), stride=(8, 1),
                                                                           padding=(0, 0), groups=768)
        model.classifier = torch.nn.Sequential(torch.nn.Linear(768, 1000, bias=True),
                                               torch.nn.Dropout(p=0.1),
                                               torch.nn.Linear(1000, 2, bias=True))
        self.ViT2 = model

    def forward(self, x):
        x = self.ViT1(x)['last_hidden_state']
        # 1, 1678, 768
        x = x[:, 1:, :].transpose(1, 2).reshape(x.shape[0], -1, 129, 13)
        x = self.ViT2(x).logits

        return x


if __name__ == '__main__':
    # Instantiate the model
    model = ViT_STL_v2()

    # Create a dummy input tensor
    batch_size = 1
    input_tensor = torch.randn(batch_size, 1, 129, 500)  # Using random values

    # Forward pass
    positions = model(input_tensor)

    # Print output shapes to verify
    print("Positions Shape:", positions.shape)
