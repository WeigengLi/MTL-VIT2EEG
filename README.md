# MLT-VIT2EEG
## Overview
Electroencephalography (EEG) is a critical technology in the domain of human-computer interaction and cognitive neuroscience.   
MLT-ViT2EEG is a Multi Task Learning approch to Decode EEG data using Vision Transformer. Our model’s main objective is to improve the Vision Transformer's performance on EEG eye-tracking tasks.  
This repository consists of model and our paper proposal.  

## Where to Find?
Our purposed Models and benchmark models are in 'models' folder  
We use jupyter notebook to store our experiment results. For reproduction, please run the notebooks
We use tensorboard to keep our traning on record, the traning record is in the 'log' folder
## Dataset download
Download data for EEGEyeNet absolute position task
```bash
wget -O "./dataset/Position_task_with_dots_synchronised_min.npz" "https://osf.io/download/ge87t/"
```
For more details about EEGEyeNet dataset, please refer to ["EEGEyeNet: a Simultaneous Electroencephalography and Eye-tracking Dataset and Benchmark for Eye Movement Prediction"](https://arxiv.org/abs/2111.05100) and [OSF repository](https://osf.io/ktv7m/)

## Installation

### Requirements

First install the general_requirements.txt

```bash
pip3 install -r general_requirements.txt 
```

### Pytorch Requirements

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For other installation details and different cuda versions, visit [pytorch.org](https://pytorch.org/get-started/locally/).

## How to run?

### Tensorboard

```bash
tensorboard --logdir logs
```