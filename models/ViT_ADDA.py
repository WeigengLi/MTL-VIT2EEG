
import torch
import transformers
from transformers import ViTModel
from torch import nn

from PIL import Image

import numpy as np
import torch
from torch.autograd import Function

import torch.nn.functional as F
from torch_geometric.nn import *


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
        output = self.ViT.forward(x, output_hidden_states=True)
        positions = output.logits
        # Extracting the shared features
        shared_features = output.hidden_states[-1][:, 0]
        return positions, shared_features

class EEGViT_pretrained_129(nn.Module):
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
        config.update({'patch_size': (129, 1)})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(129, 1), stride=(129, 1),
                                                                           padding=(0, 0), groups=256)
        model.classifier = torch.nn.Sequential(torch.nn.Linear(768, 1000, bias=True),
                                               torch.nn.Dropout(p=0.1),
                                               torch.nn.Linear(1000, 2, bias=True))
        self.ViT = model

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        output = self.ViT.forward(x, output_hidden_states=True)
        positions = output.logits
        # Extracting the shared features
        shared_features = output.hidden_states[-1][:, 0]
        return positions, shared_features
    

class ViT_pupil_Cascade(nn.Module):
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
        config.update({'patch_size': (129, 1)})
        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(129, 1), stride=(129, 1),
                                                                           padding=(0, 0), groups=256)
    
        self.model = model.vit  # Only take the ViT part without the classification head
        # Position Prediction Branch
        self.position_predictor = nn.Sequential(
                                nn.Linear(769 , 2048, bias=True),
                                nn.Dropout(p=0.1),
                                nn.Linear(2048 , 1000, bias=True),
                                nn.Dropout(p=0.1),
                                nn.Linear(1000, 2, bias=True))
        self.pupil_size_predictor=torch.nn.Sequential(
                                torch.nn.Linear(768 ,1000,bias=True),
                                torch.nn.Dropout(p=0.1),
                                torch.nn.Linear(1000,1,bias=True))
        

        

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        output = self.model(x,output_hidden_states=True)
        shared_features = output.hidden_states[-1][:, 0]
        pupil_size = self.pupil_size_predictor(shared_features)
        combined_features = torch.cat((shared_features, pupil_size), dim=1)
        positions = self.position_predictor(combined_features)
        return positions, pupil_size, shared_features




class EEGViT_pretrained_with_dis(nn.Module):
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
        self.discriminator = nn.Sequential(
                GradientReversal(),
                nn.Linear(768, 1000),
                nn.ReLU(),
                nn.Linear(1000, 20),
                nn.ReLU(),
                nn.Linear(20, 1)
            )
        self.ViT = model

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        output = self.ViT.forward(x, output_hidden_states=True)
        positions = output.logits
        # Extracting the shared features
        shared_features = output.hidden_states[-1][:, 0]
        domain = self.discriminator(shared_features)
        return positions, domain
    






class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)

    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None

class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    

class discriminator_clean(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = nn.Sequential(
                #GradientReversal(),
                nn.Linear(768, 1000),
                torch.nn.Dropout(p=0.1),
                nn.ReLU(),
                nn.Linear(1000, 1)
            )
    def forward(self, x):
        return self.discriminator(x)
    
    
class discriminator_position(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
            )
    def forward(self, x):
        return self.discriminator(x)


class discriminator_PointNet2(nn.Module):
    def __init__(self, num_points, in_channels=2, num_classes=1):
        super(discriminator_PointNet2, self).__init__()
        self.pointnet2 = PointNet2(num_points=num_points, in_channels=in_channels)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x, pos):
        x = self.pointnet2(x, pos)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class discriminator_regrad(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = nn.Sequential(
                GradientReversal(),
                nn.Linear(768, 1000),
                nn.ReLU(),
                nn.Linear(1000, 20),
                nn.ReLU(),
                nn.Linear(20, 1)
            )
    def forward(self, x):
        return self.discriminator(x)
