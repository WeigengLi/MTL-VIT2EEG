# Enhancing Eye-Tracking Performance through Multi-Task Learning

This repository is the official implementation of Enhancing Eye-Tracking Performance through Multi-Task Learning. 


## Overview
Electroencephalography (EEG) is a critical technology in the domain of human-computer interaction and cognitive neuroscience.   

MLT-ViT is a Multi Task Learning approch to Decode EEG data using Vision Transformer. Our model's main objective is to improve the Vision Transformer's performance on EEG eye-tracking tasks.  

This repository consists of model and our paper proposal.  

## Where to Find?
Our purposed Models and benchmark models are in `models` folder  

We use jupyter notebook to store our experiment results. For reproduction, please run the `notebooks`

We use tensorboard to keep our traning on record, the traning record is in the `log` folder

## Dataset download
TODO: Pupil_size Dataset

Download data for EEGEyeNet absolute position task
```bash
wget -O "./dataset/Position_task_with_dots_synchronised_min.npz" "https://osf.io/download/ge87t/"
```
For more details about EEGEyeNet dataset, please refer to ["EEGEyeNet: a Simultaneous Electroencephalography and Eye-tracking Dataset and Benchmark for Eye Movement Prediction"](https://arxiv.org/abs/2111.05100) and [OSF repository](https://osf.io/ktv7m/)

## Installation

To install requirements
```bash
pip3 install -r requirements.txt 
```

To install Pytorch requirements
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For other installation details and different cuda versions, visit [pytorch.org](https://pytorch.org/get-started/locally/).

## Training & Evaluation

To train and evaluate the model(s) in the paper, run this command:

```train
python run.py
```

### Tensorboard
```bash
tensorboard --bind_all --logdir logs 
```
> You can now easily monitor training and validation progress

## Results

Our model achieves the following performance on :

| Model name         | Absolute Position <br> RMSE (mm) | 
| ------------------ |--------------------------------- |
| CNN                | 70.2                             |
| PyramidalCNN       | 73.6                             |
| EEGNet             | 81.7                             |
| InceptionTime      | 70.8                             |
| Xception           | 78.7                             |
| MTL-ViT(Ours)      | 55.0                             |

## Citation
We reused most codes from 
```bibtex
@article{kastrati2021eegeyenet,
  title={EEGEyeNet: a simultaneous electroencephalography and eye-tracking dataset and benchmark for eye movement prediction},
  author={Kastrati, Ard and P{\l}omecka, Martyna Beata and Pascual, Dami{\'a}n and Wolf, Lukas and Gillioz, Victor and Wattenhofer, Roger and Langer, Nicolas},
  journal={arXiv preprint arXiv:2111.05100},
  year={2021}
}
@article{yang2023vit2eeg,
  title={ViT2EEG: Leveraging Hybrid Pretrained Vision Transformers for EEG Data},
  author={Yang, Ruiqi and Modesitt, Eric},
  journal={arXiv preprint arXiv:2308.00454},
  year={2023}
}
```

Our main contributions are in these files under models folder:
```
MTL_pretrained.py
MTL_raw.py
```