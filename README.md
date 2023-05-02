# OpenGCD: Assisting Open World Recognition with Generalized Category Discovery
A desirable open world recognition (OWR) system requires performing three tasks: (1) Open set recognition (OSR), i.e., classifying the known (classes seen during training) and rejecting the unknown (unseen/novel classes) online; (2) Grouping and labeling these unknown as novel known classes; (3) Incremental learning (IL), i.e., incrementally learning these novel classes and retaining the memory of old classes.

![image](https://github.com/Fulin-Gao/OpenGCD/blob/main/methods.png)

## Code
We provide code and models for our experiments on CIFAR10, CIFAR100, and CUB:
* Code for exemplar selection
* Code for closed set recognition
* Code for open set recognition
* Code for generalized category discovery

You can download the model weights (dino_vitbase16_pretrain.pth) trained on ImageNet with DINO self-supervision at [].
