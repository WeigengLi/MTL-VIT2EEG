import torch
import transformers
from torch import nn
# from einops.layers.torch import Rearrange

'''
Inspired by:

@article{song2022eeg,
  title={EEG conformer: Convolutional transformer for EEG decoding and visualization},
  author={Song, Yonghao and Zheng, Qingqing and Liu, Bingchuan and Gao, Xiaorong},
  journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering},
  volume={31},
  pages={710--719},
  year={2022},
  publisher={IEEE}
}
'''


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (129, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        # self.projection = nn.Sequential(
        #     nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fitting ability slightly
        #     Rearrange('b e h w -> b h w e'),
        # )

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        # x = self.projection(x)
        return x


class ViT_Conformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.patchEmbedding = PatchEmbedding()

        model_name = "google/vit-base-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 40})
        config.update({'image_size': (1, 27)})
        config.update({'patch_size': (1, 27)})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(40, 768, kernel_size=(1, 27),
                                                                           stride=(1, 27),
                                                                           padding=(0, 0), groups=1)
        model.classifier = torch.nn.Sequential(torch.nn.Linear(768, 1000, bias=True),
                                               torch.nn.Dropout(p=0.1),
                                               torch.nn.Linear(1000, 2, bias=True))
        self.ViT = model

    def forward(self, x):
        x = self.patchEmbedding(x)
        x = self.ViT.forward(x).logits

        return x


# Test Codes
if __name__ == '__main__':
    # Instantiate the model
    model = ViT_Conformer()

    # Create a dummy input tensor
    batch_size = 1
    input_tensor = torch.randn(batch_size, 1, 129, 500)  # Using random values

    # Forward pass
    positions = model(input_tensor)

    # Print output shapes to verify
    print("Positions Shape:", positions.shape)
