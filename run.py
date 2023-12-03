import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import argparse

from dataset.Datasets import EEGEyeNetDataset, MTLPupilDataset
# TODO: ADD COMMIT about possible models and instructions

from models.STL import EEGViT_pretrained, InceptionViT_pretrained,EEGViT_pretrained_hierachical2
from models.ViT_reconstruct_v7 import ViT_reconstruct_v7
from models.MTL_pretrained import ViT_reconstruct

from models.ModelTrainer import STL_Trainer, MTL_RE_Trainer, MTL_PU_Trainer


# region Global Config
SINGLE_TASK  = 'STL'
MULTI_TASK_RECON = 'MTL_RE'
MULTI_TASK_PUPIL='MTL_PU'
MULTI_TASK_ADDA = 'MTL_ADDA'

TASKS_HELP = {
    SINGLE_TASK : 'Single Task Learning: predict fixtion postion ((x,y) value) using EEG data ',
    MULTI_TASK_RECON : 'Multi-Task Learning: STL with a subtask that reconstruct EEG data',
    MULTI_TASK_PUPIL : 'Multi-Task Learning: STL with a subtask that predict pupil size'
}

TASKS_DATA = {
    SINGLE_TASK : EEGEyeNetDataset,
    MULTI_TASK_RECON : EEGEyeNetDataset,
    MULTI_TASK_ADDA : EEGEyeNetDataset,
    MULTI_TASK_PUPIL : MTLPupilDataset
}
TASKS_TRAINER = {
    SINGLE_TASK : STL_Trainer,
    MULTI_TASK_RECON : MTL_RE_Trainer,
    MULTI_TASK_PUPIL : MTL_PU_Trainer,
    MULTI_TASK_ADDA : MTL_ADDA_Trainer3
}
# endregion

# region Task Config

DEFAULT_TASK = MULTI_TASK_RECON

DEFAULT_MODEL = ViT_reconstruct_v7

NEW_DATA_PATH = False
NUM_ITER = 3

# endregion



def main():
    data_path = './dataset/Position_task_with_dots_synchronised_min.npz' if not NEW_DATA_PATH else NEW_DATA_PATH
    Dataset = TASKS_DATA[DEFAULT_TASK](data_path)

    for weight in [100]:
        for i in range(5):
            model = DEFAULT_MODEL()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

            mt = TASKS_TRAINER[DEFAULT_TASK](model, Dataset, optimizer, scheduler, batch_size=32, n_epoch=15, weight = weight,
                                            Trainer_name=f'ViT_reconstruct_v7_{weight}/iter{str(i+1)}')
            mt.run()

if __name__ == '__main__':
    main()
