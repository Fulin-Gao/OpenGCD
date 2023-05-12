# OpenGCD: Assisting Open World Recognition with Generalized Category Discovery
A desirable open world recognition (OWR) system requires performing three tasks: (1) Open set recognition (OSR), i.e., classifying the known (classes seen during training) and rejecting the unknown (unseen/novel classes) online; (2) Grouping and labeling these unknown as novel known classes; (3) Incremental learning (IL), i.e., incrementally learning these novel classes and retaining the memory of old classes.

![image](https://github.com/Fulin-Gao/OpenGCD/blob/main/methods.png)

## Code
We provide code and models for our experiments on CIFAR10, CIFAR100, and CUB in ```OpenGCD/exp```:
* Code for exemplar selection in ```OpenGCD/methods/```
* Code for closed set recognition in ```OpenGCD/methods/```
* Code for open set recognition in ```OpenGCD/methods/```
* Code for generalized category discovery in ```OpenGCD/methods/```

You can download the model weights (dino_vitbase16_pretrain.pth) trained on ImageNet with DINO self-supervision at [ViT-B/16](https://github.com/facebookresearch/dino).

You can run [feature extraction](https://github.com/sgvaze/generalized-category-discovery/blob/main/methods/clustering/extract_features.py) to get the feature embedding for each dataset.
