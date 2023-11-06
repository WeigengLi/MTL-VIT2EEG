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

        model_name = "google/vit-large-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (129, 14)})
        config.update({'patch_size': (16, 1)})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 1024 , kernel_size=(16, 1), stride=(16, 1),
                                                                           padding=(0, 0), groups=256)
    
        self.model = model.vit  # Only take the ViT part without the classification head
        # Position Prediction Branch
        self.position_predictor = nn.Sequential(
                                nn.Linear(1025 , 1000, bias=True),
                                nn.Dropout(p=0.1),
                                nn.Linear(1000, 2, bias=True))
        self.pupil_size=torch.nn.Sequential(
                                torch.nn.Linear(1024 ,1000,bias=True),
                                torch.nn.Dropout(p=0.1),
                                torch.nn.Linear(1000,1,bias=True))


    def forward(self, x,pupil_size_prediction):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        output = self.model(x,output_hidden_states=True)
        shared_features = output.hidden_states[-1][:, 0]

        combined_features = torch.cat((shared_features, pupil_size_prediction), dim=1)
        positions = self.position_predictor(combined_features)

        return positions, pupil_size_prediction

