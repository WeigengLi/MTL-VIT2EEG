# Enhancing Eye-Tracking Performance through Multi-Task Learning

This repository is the official implementation of Enhancing Eye-Tracking Performance through Multi-Task Learning. 


## Overview
Electroencephalography (EEG) is a critical technology in the domain of human-computer interaction and cognitive neuroscience.   

MLT-ViT is a Multi-Task Learning approch to Decode EEG data using Vision Transformer. Our model's main objective is to improve the Vision Transformer's performance on EEG eye-tracking tasks.  

This repository consists of the model and our paper proposal.  

## Where to Find? 
`dataset`: Code to generate and load the dataset. Data files should also be placed in this folder

`log`: Training logs e.g. loss. We use tensorboard to automatically record training and generate log files.

`models`: Our proposed models, benchmark models, and code to evaluate models.

`run.py` Code to reproduce our experiments.  

## Dataset download
**Data for EEG Single Task and Multi-Task with reconstruction subtask can be acquired by this code**  
```bash
wget -O "./dataset/Position_task_with_dots_synchronised_min.npz" "https://osf.io/download/ge87t/"
```
For more details about EEGEyeNet dataset, please refer to ["EEGEyeNet: a Simultaneous Electroencephalography and Eye-tracking Dataset and Benchmark for Eye Movement Prediction"](https://arxiv.org/abs/2111.05100) and [OSF repository](https://osf.io/ktv7m/)

**Data for Multi-Task with pupil size prediction as subtask can be acquired by:**  
1. First, Download "dots/synchronized_min" folder from [EEGEyeNet dataset OSF repository](https://osf.io/ktv7m/)  
2. Unzip it in "./dataset/synchronized_min"  
3. Run dataset "./preparator/tasks.py"

## Installation
We use Python 3.8, Cuda 12.1

To install Pytorch requirements
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

To install requirements
```bash
pip3 install -r requirement.txt 
```



For other installation details and different cuda versions, visit [pytorch.org](https://pytorch.org/get-started/locally/).

## Training & Evaluation

To reproduce our best result using the reconstruction module in Muti-task Framework, run this command:

```train
python run.py
```

### Tensorboard
After the evaluation is complete, use this command to open tensorboard UI
```bash
tensorboard --bind_all --logdir logs 
```
> You can now easily monitor training and validation progress

## Results

Our model achieves the following performance:

| Model name           | Absolute Position <br> RMSE (mm) | 
| -------------------- |--------------------------------- |
| CNN                  | 70.2 +- 1.1                      |
| EEG2VIT-pretrained   | 55.4 +- 0.2                      |
| MTLT-pretrained(Ours)| 54.1 +- 0.2                      |

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

Our main contributions are in these files under the models folder:
```
MTL_pretrained.py
MTL_raw.py
```