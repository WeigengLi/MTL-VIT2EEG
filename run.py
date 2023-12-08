import torch

from dataset.Datasets import EEGEyeNetDataset, MTLPupilDataset
# TODO: ADD COMMIT about possible models and instructions

from models.ModelTrainer import STL_Trainer, MTL_RE_Trainer, MTL_PU_Trainer
from models.MTLT import MTLT, MTLT_raw
from models.MTLT_v13 import MTLT_v13
from models.MTLT_v12 import MTLT_v12
from models.MTLT_v10 import MTLT_v10
from models.MTLT_v11 import MTLT_v11
from models.MTLT_v14 import MTLT_v14
from models.MTLT_v15 import MTLT_v15
from models.MTLT_v16 import MTLT_v16
from models.STL import EEGViT_pretrained, EEGViT_raw

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
}

MTL_WEIGHT = {
    MTLT.__name__ : 100,
    MTLT_raw.__name__ : 120
}

# endregion


# region Task Config

DEFAULT_TASK = MULTI_TASK_RECON

DEFAULT_MODEL = MTLT

NEW_DATA_PATH = False
# endregion


def main():
    data_path = './dataset/Position_task_with_dots_synchronised_min.npz' if not NEW_DATA_PATH else NEW_DATA_PATH
    Dataset = TASKS_DATA[DEFAULT_TASK](data_path)
    for weight in [100,120]:
        for i in range(1,6):
            model = DEFAULT_MODEL()
            model_name = DEFAULT_MODEL.__name__
            # weight = MTL_WEIGHT.get(model_name, 0)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

            mt = TASKS_TRAINER[DEFAULT_TASK](model, Dataset, optimizer, scheduler, batch_size=64, n_epoch=15, weight = 100,
                                            Trainer_name=f'{model_name}_weight{weight}_wc_seed1_{i}')
            mt.run()


def main2():
    data_path = './dataset/Position_task_with_dots_synchronised_min.npz' if not NEW_DATA_PATH else NEW_DATA_PATH
    Dataset = TASKS_DATA[DEFAULT_TASK](data_path)
    for weight in [0,20,40]:
        for i in range(1,6):
            model = MTLT_v13()
            model_name = model.__class__.__name__

            # weight = MTL_WEIGHT.get(model_name, 0)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

            mt = TASKS_TRAINER[DEFAULT_TASK](model, Dataset, optimizer, scheduler, batch_size=64, n_epoch=15, weight = weight,
                                            Trainer_name=f'{model_name}_weight{weight}_wc_seed1_{i}')
            mt.run()


def main3():
    data_path = './dataset/Position_task_with_dots_synchronised_min.npz' if not NEW_DATA_PATH else NEW_DATA_PATH
    Dataset = TASKS_DATA[DEFAULT_TASK](data_path)
    for weight in [100]:
        for i in range(4):
            model = MTLT_v16()
            model_name = model.__class__.__name__

            # weight = MTL_WEIGHT.get(model_name, 0)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

            mt = TASKS_TRAINER[DEFAULT_TASK](model, Dataset, optimizer, scheduler, batch_size=32, n_epoch=15, weight = weight,
                                            Trainer_name=f'{model_name}_weight{weight}_wc_seed1_{i}')
            mt.run()

def post_process(data):
    import numpy as np
    
    # 创建一个示例数据集
    data = np.array(data)

    # 计算平均数
    mean_value = np.mean(data)

    # 计算方差
    variance_value = np.var(data)
    
    # 打印平均数和方差，保留5位小数
    std_str = f"{round(mean_value, 5)} +- {round(variance_value, 5)}"
    return std_str


if __name__ == '__main__':
    # main()
    # main2()
    # main3()
    # post_process([54.27, 55.83, 56.11, 54.93, 55.03]) # weight100
    # post_process([54.26, 55.14, 55.23, 55.03, 54.21]) # weight120
    # post_process([56.76, 54.88, 54.57]) # v12_weight100
    # post_process([54.9, 54.36, 54.98]) # v12_weight120

    avg_acc = post_process([55.67,55.8])# v13_weight0
    print(f'v13_weight0:\t{avg_acc}')
    avg_acc =  post_process([54.17, 54.26, 55.16, 55.63,56.21])# v13_weight20
    print(f'v13_weight20:\t{avg_acc}')
    avg_acc = post_process([53.93, 54.6, 54.82, 54.86,55.24])# v13_weight40
    print(f'v13_weight40:\t{avg_acc}')
    avg_acc = post_process([54.58, 55.18, 54.31, 54.93, 54.05, 53.69]) # v13_weight100
    print(f'v13_weight100:\t{avg_acc}')
    avg_acc = post_process([54.27, 54.64, 54.65, 54.19, 53.79, 53.94]) # v13_weight120
    print(f'v13_weight120:\t{avg_acc}')
    