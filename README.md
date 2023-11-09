# Enhancing Eye-Tracking Performance through Multi-Task Learning

This repository is the official implementation of Enhancing Eye-Tracking Performance through Multi-Task Learning (https://arxiv.org/abs/2030.12345). 


## Overview
Electroencephalography (EEG) is a critical technology in the domain of human-computer interaction and cognitive neuroscience.   
MLT-ViT2EEG is a Multi Task Learning approch to Decode EEG data using Vision Transformer. Our model’s main objective is to improve the Vision Transformer's performance on EEG eye-tracking tasks.  
This repository consists of model and our paper proposal.  

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Where to Find?
Our purposed Models and benchmark models are in 'models' folder  
We use jupyter notebook to store our experiment results. For reproduction, please run the notebooks
We use tensorboard to keep our traning on record, the traning record is in the 'log' folder

## Dataset download
TODO: Pupil_size Dataset
Download data for EEGEyeNet absolute position task
```bash
wget -O "./dataset/Position_task_with_dots_synchronised_min.npz" "https://osf.io/download/ge87t/"
```
For more details about EEGEyeNet dataset, please refer to ["EEGEyeNet: a Simultaneous Electroencephalography and Eye-tracking Dataset and Benchmark for Eye Movement Prediction"](https://arxiv.org/abs/2111.05100) and [OSF repository](https://osf.io/ktv7m/)

## Installation

To install requirements:

```bash
pip3 install -r requirements.txt 
```


```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
For other installation details and different cuda versions, visit [pytorch.org](https://pytorch.org/get-started/locally/).

For Conda Enviroment

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```
### Tensorboard
```bash
tensorboard --bind_all --logdir logs 
```
>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 


## Citation
'''
These Code is origiin form VIT2EEG Paper, See Citation 1 in README
'''