import torch
import transformers
from transformers import ViTModel
import torch
from torch import nn
import transformers


class InceptionModule(nn.Module):
    def __init__(self, in_channels):
        super(InceptionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(1, 1), padding='same')
        self.conv2 = nn.Conv2d(in_channels, 64, kernel_size=(1, 1), padding='same')
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 1), padding='same')
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(5, 1), padding='same')
        self.batchnorm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

    def forward(self, x):
        branch1 = self.relu(self.batchnorm(self.conv1(x)))
        branch2 = self.relu(self.batchnorm(self.conv3(self.conv2(x))))
        branch3 = self.relu(self.batchnorm(self.conv4(self.conv2(x))))
        output = torch.cat([branch1, branch2, branch3], 1)
        return output


class InceptionViT_pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.inception1 = InceptionModule(1)
        self.inception2 = InceptionModule(192)
        self.inception3 = InceptionModule(192)

        self.batchnorm1 = nn.BatchNorm2d(192)
        self.batchnorm2 = nn.BatchNorm2d(192)
        self.batchnorm3 = nn.BatchNorm2d(192)

        model_name = "google/vit-base-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 192})  # Update number of channels to match output of inception3
        config.update({'image_size': (129, 500)})
        config.update({'patch_size': (129, 1)})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(192, 768, kernel_size=(129, 1),
                                                                           stride=(129, 1),
                                                                           padding=(0, 0), groups=192)
        model.classifier = torch.nn.Sequential(torch.nn.Linear(768, 1000, bias=True),
                                               torch.nn.Dropout(p=0.1),
                                               torch.nn.Linear(1000, 2, bias=True))
        self.ViT = model

    def forward(self, x):
        x = self.batchnorm1(self.inception1(x))
        x = self.batchnorm2(self.inception2(x))
        x = self.batchnorm3(self.inception3(x))
        x = self.ViT.forward(x).logits

        return x


class EEGViT_pretrained_hierachical(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(1, 24), stride=(1, 24), padding=(0, 2), bias=False)
        self.conv2 = nn.Conv2d(1, 256, kernel_size=(1, 36), stride=(1, 24), padding=(0, 8), bias=False)
        self.conv3 = nn.Conv2d(1, 256, kernel_size=(1, 48), stride=(1, 24), padding=(0, 14), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(768, False)
        model_name = "google/vit-base-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 768})
        config.update({'image_size': (129, 21)})
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


'''
Codes below is origiin form VIT2EEG Paper, See Citation 1 in README
'''


class EEGViT_pretrained(nn.Module):
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.ViT.forward(x).logits

        return x


class EEGViT_raw(nn.Module):
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
        config = transformers.ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            num_channels=256,
            image_size=(129, 14),
            patch_size=(8, 1)
        )
        model = ViTModel(config)
        model.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(8, 1), stride=(8, 1),
                                                                       padding=(0, 0), groups=256)
        model.pooler.activation = torch.nn.Sequential(torch.nn.Dropout(p=0.1),
                                                      torch.nn.Linear(768, 2, bias=True))
        self.ViT = model

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.ViT(x).pooler_output

        return x


class ViTBase_pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        model_name = "google/vit-base-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 1})
        config.update({'image_size': (129, 500)})
        config.update({'patch_size': (8, 35)})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = nn.Sequential(
            torch.nn.Conv2d(1, 768, kernel_size=(8, 36), stride=(8, 36), padding=(0, 2)),
            nn.BatchNorm2d(768))
        model.classifier = torch.nn.Sequential(torch.nn.Linear(768, 1000, bias=True),
                                               torch.nn.Dropout(p=0.1),
                                               torch.nn.Linear(1000, 2, bias=True))
        self.ViT = model

    def forward(self, x):
        x = self.ViT(x).logits
        return x


class ViTBase(nn.Module):
    def __init__(self):
        super().__init__()
        config = transformers.ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            num_channels=1,
            image_size=(129, 500),
            patch_size=(8, 35)
        )
        model = ViTModel(config)
        model.embeddings.patch_embeddings.projection = torch.nn.Conv2d(1, 768, kernel_size=(8, 36), stride=(8, 36),
                                                                       padding=(0, 2))
        model.pooler.activation = torch.nn.Sequential(torch.nn.Dropout(p=0.1),
                                                      torch.nn.Linear(768, 2, bias=True))

    def forward(self, x):
        x = self.model(x).pooler_output
        return x


class EEGViT_pretrained_hierachical(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(1, 24), stride=(1, 24), padding=(0, 2), bias=False)
        self.conv2 = nn.Conv2d(1, 256, kernel_size=(1, 36), stride=(1, 24), padding=(0, 8), bias=False)
        self.conv3 = nn.Conv2d(1, 256, kernel_size=(1, 48), stride=(1, 24), padding=(0, 14), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(768, False)
        model_name = "google/vit-base-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 768})
        config.update({'image_size': (129, 21)})
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


# Test Codes
if __name__ == '__main__':
    # Instantiate the model
    model = InceptionViT_pretrained()

    # Create a dummy input tensor
    batch_size = 1
    input_tensor = torch.randn(batch_size, 1, 129, 500)  # Using random values

    # Forward pass
    positions = model(input_tensor)

    # Print output shapes to verify
    print("Positions Shape:", positions.shape)
