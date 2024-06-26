# OpenGCD: Assisting Open World Recognition with Generalized Category Discovery
[![Bower](https://img.shields.io/bower/l/mi)](https://github.com/Fulin-Gao/OpenGCD/blob/main/LICENSE)
[![](https://img.shields.io/badge/python-v3.9-blue.svg?logo=python)](https://www.python.org)
[![](https://img.shields.io/badge/pytorch-v1.13.1-red.svg?logo=pytorch)](https://pytorch.org)

A desirable open world recognition (OWR) system requires performing three tasks: (1) Open set recognition (OSR), i.e., classifying the known (classes seen during training) and rejecting the unknown (unseen/novel classes) online; (2) Grouping and labeling these unknown as novel known classes; (3) Incremental learning (IL), i.e., incrementally learning these novel classes and retaining the memory of old classes.

![image](https://github.com/Fulin-Gao/OpenGCD/blob/main/methods.png)

## :fire: Preparation
### Dependencies
All dependencies are included in ```environment.yml```. To install, run
```
conda env create -f environment.yml
```

### Data
You can find the required data from the link below and place them according to the path shown in ```OpenGCD/config.py```.
* [CIFAR-10, CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
* [CUB](https://www.vision.caltech.edu/datasets/cub_200_2011/)
* [ImangeNet](https://image-net.org/challenges/LSVRC/2012/index.php)

### Pretrained weights
You can download the ViT weights (dino_vitbase16_pretrain.pth) trained on ImageNet with DINO self-supervision at [ViT-B/16](https://github.com/facebookresearch/dino).

### Features
You can run [extract_features.py](https://github.com/sgvaze/generalized-category-discovery/blob/main/methods/clustering/extract_features.py) to get the feature embedding for each dataset.

## :rocket: Code
We provide code and models for our experiments on CIFAR10, CIFAR100, and CUB in ```OpenGCD```:
* Code for exemplar selection in ```OpenGCD/methods/exemplars_selection```
* Code for closed set recognition in ```OpenGCD/methods/closed_set_recognition```
* Code for open set recognition in ```OpenGCD/methods/open_set_recognition```
* Code for generalized category discovery in ```OpenGCD/methods/novel_category_discover```
* Code for evaluation metrics (Harmonic normalized accuracy and Harmonic clustering accuracy) in ```OpenGCD/project_utils/metrics.py```
* Code for our experiments in ```OpenGCD/exp```

## :bookmark: Procedure
The schematic and pseudocode of the formulated OpenGCD are provided in Appendix.pdf.

## :bulb: Acknowledgement
Our code is partially based on the following repositories:
* [A Tool for Software Defect Prediction](https://github.com/afiore51/CNN_DS3_BugDetection/tree/master)
* [Generalized Category Discovery](https://github.com/sgvaze/generalized-category-discovery)
* [AutoNovel: Automatically Discovering and Learning Novel Visual Categories](https://github.com/k-han/AutoNovel)
