# OKDDip
**O**nline **K**nowledge **D**istillation with **Di**verse **P**eers (AAAI-2020) https://arxiv.org/abs/1912.00350

This is a PyTorch-1.0 implementation of the OKDDip algorithm together with the compared approaches (such as classic KD and online variants like ONE, CL-ILR, DML). 

This paper attempts to alleviate homogenization problem during training of student models. Specifically, OKDDip performs two-level distillation during training with multiple auxiliary peers and one group leaders. In the first-level distillation, each auxiliary peer holds an individual set of aggregation weights generated with an attention-based mechanism to derive its own targets from predictions of other auxiliary peers. The second-level distillation is performed to transfer the knowledge in the ensemble of auxiliary peers further to the group leader, i.e., the model used for inference.

```
pip install -r requirements.txt
```

## Basic Usage

The default experimental parameter setting is : 

```
--num_epochs 300 --batch_size 128 --lr 0.1 --schedule 150 225 --wd 5e-4 
```

### 1. Baseline 

Train **resnet32** model on **CIFAR10** dataset.

```
python train.py --model resnet32 --dataset CIFAR10 
```
### 2. Classic KD 

Train student **resnet32** model with teacher **resnet110** model on **CIFAR10** dataset.

```
python train_kd.py --model resnet32 --T_model resnet110 --T_model_path ./CIFAR10/resnet110 --dataset CIFAR10
```

### 3. OKDDip 

Train **resnet32** model on **CIFAR10** dataset.

```
python train_GL.py --model resnet32 --dataset CIFAR10
```

### 4. ONE

Train **resnet32** model on **CIFAR10** dataset.

```
python train_one.py --model resnet32 --dataset CIFAR10
```

### 5. CL-ILR

Train **resnet32** model on **CIFAR10** dataset.

```
python train_one.py --model resnet32 --dataset CIFAR10 --avg --bpscale
```

### 6. Individual Branch

Train **resnet32** model on **CIFAR10** dataset.

```
python train_one.py --model resnet32 --dataset CIFAR10 --ind
```



**Notes:** The codes in this repository is merged from different sources, and we have not tested them thoroughly. Hence, if you have any questions, please contact us without hesitation.

The results may slightly vary as the environment changed, just run it again! (Thanks the feedback from [Zheng Li](https://github.com/zhengli427))

Email: defchern **&alpha;** t zju dot edu d **&omicron;** t cn



## Reference
If you find this repository useful, please consider citing the following paper:
```
@article{chen2019online,
  title={Online Knowledge Distillation with Diverse Peers},
  author={Chen, Defang and Mei, Jian-Ping and Wang, Can and Feng, Yan and Chen, Chun},
  journal={arXiv preprint arXiv:1912.00350},
  year={2019}
}

```
