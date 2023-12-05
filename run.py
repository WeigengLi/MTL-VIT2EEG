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

DEFAULT_MODEL = MTLT_v14

NEW_DATA_PATH = False
# endregion


def main():
    data_path = './dataset/Position_task_with_dots_synchronised_min.npz' if not NEW_DATA_PATH else NEW_DATA_PATH
    Dataset = TASKS_DATA[DEFAULT_TASK](data_path)
    model = DEFAULT_MODEL()
    model_name = DEFAULT_MODEL.__name__
    weight = MTL_WEIGHT.get(model_name, 0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

    mt = TASKS_TRAINER[DEFAULT_TASK](model, Dataset, optimizer, scheduler, batch_size=32, n_epoch=15, weight = 100,
                                    Trainer_name=f'{model_name}_num_layer_6_weight_100_wrong_criteria')
    mt.run()

if __name__ == '__main__':
    main()
