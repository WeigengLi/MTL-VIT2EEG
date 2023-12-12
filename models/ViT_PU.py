import torch
import transformers
from transformers import ViTModel
import torch
from torch import nn
import transformers
from models.ModelTrainer import MTL_PU_Trainer

class get_model(nn.Module):
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



def get_config():
    model = get_model()
    # weight = MTL_WEIGHT.get(model_name, 0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
    trainer = MTL_PU_Trainer
    return {
        'model' : model,
        'optimizer' : optimizer,
        'scheduler' : scheduler,
        'weight' : 120
    }, trainer