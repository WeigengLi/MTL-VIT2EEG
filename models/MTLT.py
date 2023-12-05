import torch
import transformers
from torch import nn
from transformers import ViTModel
'''
Inspired by:

@article{lee2022anovit,
  title={AnoViT: Unsupervised anomaly detection and localization with vision transformer-based encoder-decoder},
  author={Lee, Yunseung and Kang, Pilsung},
  journal={IEEE Access},
  volume={10},
  pages={46717--46724},
  year={2022},
  publisher={IEEE}
}

changes:
nothing changed compared to EEGViT
add decoder blocks
'''

class Recoonstruct_module(nn.Module):
    def __init__(self):
        super().__init__()
        # Decoder
        self.dec_block1 = nn.Sequential(
            nn.ConvTranspose2d(768, 512, (1, 36),
                               stride=(1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(512),
            nn.ReLU(True)
        )

        self.dec_block2 = nn.Sequential(
            nn.ConvTranspose2d(
                512, 256, (8, 1), stride=(2, 1), padding=(1, 0)),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        )

        self.dec_block3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, (1, 36),
                               stride=(1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        )

        self.dec_block4 = nn.Sequential(
            nn.ConvTranspose2d(128, 1, (8, 1), stride=(2, 1), padding=(1, 0)),
            nn.InstanceNorm2d(1),
            nn.ReLU(True)
        )

        self.up = nn.UpsamplingBilinear2d((129, 500))  # fixed output size
        self.tanh = nn.Tanh()

        
    def forward(self, shared_features):
        reshaped_features = shared_features[:, 1:, :].transpose(
            1, 2).reshape(shared_features.shape[0], -1, 16, 14)
        x_reconstructed = self.dec_block1(reshaped_features)
        x_reconstructed = self.dec_block2(x_reconstructed)
        x_reconstructed = self.dec_block3(x_reconstructed)
        x_reconstructed = self.dec_block4(x_reconstructed)
        x_reconstructed = self.up(x_reconstructed)
        x_reconstructed = self.tanh(x_reconstructed)
        return x_reconstructed

class MTLT(nn.Module):
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
        # model.classifier = torch.nn.Sequential(torch.nn.Linear(768, 512, bias=True),
        #                                        torch.nn.Dropout(p=0.5),
        #                                        torch.nn.Linear( 512, 256, bias=True),
        #                                        torch.nn.Dropout(p=0.5),
        #                                        torch.nn.Linear(256, 2, bias=True))
        model.classifier = torch.nn.Sequential(torch.nn.Linear(768, 1000, bias=True),
                                               torch.nn.Dropout(p=0.1),
                                               torch.nn.Linear(1000, 2, bias=True))
        self.ViT = model

        # Decoder
        self.dec_block1 = nn.Sequential(
            nn.ConvTranspose2d(768, 512, (1, 36),
                               stride=(1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(512),
            nn.ReLU(True)
        )

        self.dec_block2 = nn.Sequential(
            nn.ConvTranspose2d(
                512, 256, (8, 1), stride=(2, 1), padding=(1, 0)),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        )

        self.dec_block3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, (1, 36),
                               stride=(1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        )

        self.dec_block4 = nn.Sequential(
            nn.ConvTranspose2d(128, 1, (8, 1), stride=(2, 1), padding=(1, 0)),
            nn.InstanceNorm2d(1),
            nn.ReLU(True)
        )

        self.up = nn.UpsamplingBilinear2d((129, 500))  # fixed output size
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        output = self.ViT.forward(x, output_hidden_states=True)
        positions = output.logits

        # Extracting the shared features
        shared_features = output.hidden_states[-1]

        # Decoder
        reshaped_features = shared_features[:, 1:, :].transpose(
            1, 2).reshape(shared_features.shape[0], -1, 16, 14)
        x_reconstructed = self.dec_block1(reshaped_features)
        x_reconstructed = self.dec_block2(x_reconstructed)
        x_reconstructed = self.dec_block3(x_reconstructed)
        x_reconstructed = self.dec_block4(x_reconstructed)
        x_reconstructed = self.up(x_reconstructed)
        x_reconstructed = self.tanh(x_reconstructed)

        return positions, x_reconstructed, shared_features[:, 0]

class MTLT_raw(nn.Module):
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
        self.reconstruct_decoder = Recoonstruct_module()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.ViT(x,output_hidden_states=True)
        position = x.pooler_output
        # Extracting the shared features
        shared_features = x.hidden_states[-1]
        x_reconstructed = self.reconstruct_decoder(shared_features)
        return position ,  x_reconstructed , shared_features[:,0]



if __name__ == '__main__':
    # Instantiate the model
    model = MTLT()

    # Create a dummy input tensor
    batch_size = 1
    input_tensor = torch.randn(batch_size, 1, 129, 500)  # Using random values

    # Forward pass
    positions, x_reconstructed = model(input_tensor)

    # Print output shapes to verify
    print("Positions Shape:", positions.shape)
    print("Reconstructed Shape:", x_reconstructed.shape)
