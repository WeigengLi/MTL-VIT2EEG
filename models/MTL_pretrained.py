import torch
import transformers
from transformers import ViTModel
import torch
from torch import nn
import transformers

'''
This is our purposed models and other tested model
Region 1: Reconstruct Subtask
Region 2: Predict Pupil Size Subtask
'''



#region Predict Pupil Size Subtask

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
        return positions, pupil_size


class ViT_pupil(nn.Module):
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
        config.update({'patch_size': (8, 1)})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(8, 1), stride=(8, 1),
                                                                           padding=(0, 0), groups=256)
        model.classifier=torch.nn.Sequential(torch.nn.Linear(768,1000,bias=True),
                                     torch.nn.Dropout(p=0.1),
                                     torch.nn.Linear(1000,2,bias=True))
        self.model = model
        self.pupil_size=torch.nn.Sequential(torch.nn.Linear(768,1000,bias=True),
                                torch.nn.Dropout(p=0.1),
                                torch.nn.Linear(1000,1,bias=True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        output = self.model.forward(x,output_hidden_states=True)
        positions=output.logits
        shared_features = output.hidden_states[-1]
        # Position Prediction
        pupil_size_prediction = self.pupil_size(shared_features[:, 0])

        return positions, pupil_size_prediction,shared_features[:, 0]





#endregion

'''
Test Codes
'''

# # Instantiate the model
# model = MTLViT_pretrained()
#
# # Create a dummy input tensor
# batch_size = 1
# input_tensor = torch.randn(batch_size, 1, 129, 500)  # Using random values
#
# # Forward pass
# positions, x_reconstructed = model(input_tensor)
#
# # Print output shapes to verify
# print("Positions Shape:", positions.shape)
# print("Reconstructed Shape:", x_reconstructed.shape)
