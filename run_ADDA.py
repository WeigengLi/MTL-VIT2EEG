import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import argparse

from dataset.Datasets import EEGEyeNetDataset, MTLPupilDataset
# TODO: ADD COMMIT about possible models and instructions
from models.STL import  InceptionViT_pretrained,EEGViT_pretrained_hierachical2
from models.Vit_reconstruct_1116 import ViT_reconstruct_modified
from models.ViT_reconstruct_v4 import ViT_reconstruct_v4
from models.MTL_pretrained import ViT_reconstruct
from models.ModelTrainer import *
from models.ViT_ADDA import *



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
DEFAULT_TASK = MULTI_TASK_ADDA
NEW_DATA_PATH = False
NUM_ITER = 3
# endregion

# 从头训练模型
def ADDA_with_dis():
    data_path = './dataset/Position_task_with_dots_synchronised_min.npz' if not NEW_DATA_PATH else NEW_DATA_PATH
    Dataset = TASKS_DATA[DEFAULT_TASK](data_path)
    for weight in [1000]:
        for i in range(5):
            model = EEGViT_pretrained_with_dis()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
            mt = MTL_ADDA_Trainer_with_dis(model, Dataset, optimizer = optimizer, scheduler = scheduler,
                                             batch_size=64, n_epoch=15, weight = weight,
                                            Trainer_name=f'MULTI_TASK_ADDA_weight{weight}/iter{str(i+1)}')
            mt.run()

# 使用训练好的模型测测ADDA有没有用
def ADDA_with_pre():
    data_path = './dataset/Position_task_with_dots_synchronised_min.npz' if not NEW_DATA_PATH else NEW_DATA_PATH
    Dataset = TASKS_DATA[DEFAULT_TASK](data_path)
    # TODO: 能不能先把discriminator训练好，然后再训练整个网络
    # 比如将反转梯度作为一个选项，然后在训练的时候，可以选择是否反转梯度
    for weight in [3000,4000]:
        for i in range(5):
            model = torch.load('EEGViT_pretrained.pth')
            discriminator = discriminator_regrad()
            optimizer = torch.optim.Adam([
                            {'params': model.parameters(), 'lr': 1e-4},
                            {'params': discriminator.parameters(), 'lr': 1e-2}
                        ])
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
            mt = MTL_ADDA_Trainer_with_pre2(model, Dataset, optimizer = optimizer, scheduler = scheduler, discriminator= discriminator,
                                             batch_size=64, n_epoch=15, weight = weight,
                                            Trainer_name=f'MULTI_TASK_ADDA_weight{weight}/iter{str(i+1)}')
            mt.run()
    
    
# 预训练模型和判别器，然后再训练整个网络
def ADDA_position():
    data_path = './dataset/Position_task_with_dots_synchronised_min.npz' if not NEW_DATA_PATH else NEW_DATA_PATH
    Dataset = TASKS_DATA[DEFAULT_TASK](data_path)

    for weight in [3000,4000,6000,8000]:
        for i in range(2):
            model = model=torch.load('EEGViT_pretrained.pth')
            discriminator = discriminator_position()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
            mt = MTL_position_ADDA(model, Dataset, optimizer = optimizer, scheduler = scheduler, discriminator= discriminator,
                                             batch_size=64, n_epoch=10, weight = weight,
                                            Trainer_name=f'MULTI_TASK_ADDA_weight{weight}_test5/iter{str(i+1)}')
            mt.run()

# TODO: 对预测结果进行对抗学习呢

if __name__ == '__main__':
    ADDA_position()
