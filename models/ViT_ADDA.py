
import torch
import transformers
from transformers import ViTModel
from torch import nn

from PIL import Image
from torch import nn, einsum
import numpy as np
import torch
from torch.autograd import Function

import torch.nn.functional as F
from torch_geometric.nn import *
from einops import repeat, rearrange

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
    

class EEGViT_pretrained_with_point_net(nn.Module):
    def __init__(self, bach_size):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=256,
            kernel_size=(1, 36),
            stride=(1, 36),
            padding=(0, 2),
            bias=False
        )
        self.batch_size = bach_size
        self.feature_dim = 32
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
        self.position_predictor = torch.nn.Sequential(torch.nn.Linear(769, 1000, bias=True),
                                               torch.nn.Dropout(p=0.1),
                                               torch.nn.Linear(1000, 2, bias=True))
        self.pupil_size_predictor=torch.nn.Sequential( torch.nn.Linear(768 ,1000,bias=True),
                                                torch.nn.Dropout(p=0.1),
                                                torch.nn.Linear(1000,1,bias=True))
        self.discriminator = PointT_Discriminator_reg(feature_dim = self.feature_dim, batch_size = self.batch_size)
        self.ViT = model

    def forward(self, x, pre_domain=True):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        output = self.ViT.vit(x,output_hidden_states=True)
        shared_features = output.hidden_states[-1][:, 0]
        pupil_size = self.pupil_size_predictor(shared_features)
        combined_features = torch.cat((shared_features, pupil_size), dim=1)
        positions = self.position_predictor(combined_features)
        
        if pre_domain:
            #TODO: 把positions从batchsize,2变成·batchsize,3
            positions_3d = torch.cat([positions, torch.zeros(positions.shape[0], 1, device=positions.device)], dim=1).unsqueeze(0)
            feats=torch.ones(1, self.batch_size, self.feature_dim).to( device=positions.device)
            mask = torch.ones(1, self.batch_size).bool().to( device=positions.device)
            domain = self.discriminator(x = positions_3d,feats = feats,  mask = mask).squeeze(1)
        else:
            domain = self.discriminator(shared_features)
        return positions, domain,pupil_size
    




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
    
    
class point_transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.pint_T = MultiheadPointTransformerLayer(
                    dim = 1,
                    pos_mlp_hidden_dim = 64,
                    attn_mlp_hidden_mult = 4,
                    num_neighbors = 4
                )
        self.discreminator = nn.Sequential(
                GradientReversal(),
                nn.Linear(768, 1000),
                nn.ReLU(),
                nn.Linear(1000, 20),
                nn.ReLU(),
                nn.Linear(20, 1)
            )
    def forward(self, x):
        return self.discriminator(x)

from einops import repeat, rearrange

# helpers


def exists(val):
    return val is not None


def max_value(t):
    return torch.finfo(t.dtype).max


def batched_index_select(values, indices, dim=1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(
        lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *
                     ((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

# classes


class MultiheadPointTransformerLayer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads=4,
        dim_head=64,
        pos_mlp_hidden_dim=64,
        attn_mlp_hidden_mult=4,
        num_neighbors=None
    ):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads

        self.num_neighbors = num_neighbors

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden_dim, inner_dim)
        )

        attn_inner_dim = inner_dim * attn_mlp_hidden_mult

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(inner_dim, attn_inner_dim, 1, groups=heads),
            nn.ReLU(),
            nn.Conv2d(attn_inner_dim, inner_dim, 1, groups=heads),
        )

    def forward(self, x, pos, mask=None):
        n, h, num_neighbors = x.shape[1], self.heads, self.num_neighbors

        # get queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # split out heads

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # calculate relative positional embeddings

        rel_pos = rearrange(pos, 'b i c -> b i 1 c') - \
            rearrange(pos, 'b j c -> b 1 j c')
        rel_pos_emb = self.pos_mlp(rel_pos)

        # split out heads for rel pos emb

        rel_pos_emb = rearrange(rel_pos_emb, 'b i j (h d) -> b h i j d', h=h)

        # use subtraction of queries to keys. i suppose this is a better inductive bias for point clouds than dot product

        qk_rel = rearrange(q, 'b h i d -> b h i 1 d') - \
            rearrange(k, 'b h j d -> b h 1 j d')

        # prepare mask

        if exists(mask):
            mask = rearrange(mask, 'b i -> b i 1') * \
                rearrange(mask, 'b j -> b 1 j')

        # expand values

        v = repeat(v, 'b h j d -> b h i j d', i=n)

        # determine k nearest neighbors for each point, if specified

        if exists(num_neighbors) and num_neighbors < n:
            rel_dist = rel_pos.norm(dim=-1)

            if exists(mask):
                mask_value = max_value(rel_dist)
                rel_dist.masked_fill_(~mask, mask_value)

            dist, indices = rel_dist.topk(num_neighbors, largest=False)

            indices_with_heads = repeat(indices, 'b i j -> b h i j', h=h)

            v = batched_index_select(v, indices_with_heads, dim=3)
            qk_rel = batched_index_select(qk_rel, indices_with_heads, dim=3)
            rel_pos_emb = batched_index_select(
                rel_pos_emb, indices_with_heads, dim=3)

            if exists(mask):
                mask = batched_index_select(mask, indices, dim=2)

        # add relative positional embeddings to value

        v = v + rel_pos_emb

        # use attention mlp, making sure to add relative positional embedding first

        attn_mlp_input = qk_rel + rel_pos_emb
        attn_mlp_input = rearrange(attn_mlp_input, 'b h i j d -> b (h d) i j')

        sim = self.attn_mlp(attn_mlp_input)

        # masking

        if exists(mask):
            mask_value = -max_value(sim)
            mask = rearrange(mask, 'b i j -> b 1 i j')
            sim.masked_fill_(~mask, mask_value)

        # attention

        attn = sim.softmax(dim=-2)

        # aggregate

        v = rearrange(v, 'b h i j d -> b i j (h d)')
        agg = einsum('b d i j, b i j d -> b i d', attn, v)

        # combine heads

        return self.to_out(agg)


class PointT_Discriminator_reg(nn.Module):
    def __init__(self,feature_dim = 32, batch_size = 64):
        super(PointT_Discriminator_reg,self).__init__()
        self.feature_dim = feature_dim
        self.grad_reversal1 = GradientReversal()
        self.grad_reversal2 = GradientReversal()
        self.grad_reversal3 = GradientReversal()
        self.pointT = MultiheadPointTransformerLayer(
                            dim=self.feature_dim,
                            pos_mlp_hidden_dim=64,
                            attn_mlp_hidden_mult=4,
                            # only the 16 nearest neighbors would be attended to for each point
                            num_neighbors=8)
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim*batch_size, 1024),  # 根据扁平化后的特征大小调整
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid() 
        )

    def forward(self, feats, x, mask):
        x = self.grad_reversal1(x)
        feats = self.grad_reversal2(feats)
        mask = self.grad_reversal3(mask)
        x = self.pointT(feats, x, mask)
        return self.fc_layers(x)


# attn = MultiheadPointTransformerLayer(
#     dim=1,
#     pos_mlp_hidden_dim=64,
#     attn_mlp_hidden_mult=4,
#     # only the 16 nearest neighbors would be attended to for each point
#     num_neighbors=16
# )

# feats = torch.ones(24, 128, 1)
# pos = torch.randn(24, 128, 3)
# mask = torch.ones(24, 128).bool()

# a = attn(feats, pos, mask=mask)  # (1, 16, 128)


# print(a.shape)
