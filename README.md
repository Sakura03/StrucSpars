# Structured Sparsification with Joint Optimization of Group Convolution and Channel Shuffle

Official implementation of paper: **Structured Sparsification with Joint Optimization of Group Convolution and Channel Shuffle** (UAI 2021), by Xin-Yu Zhang, Kai Zhao, Taihong Xiao, Ming-Ming Cheng, and Ming-Hsuan Yang. [[paper](https://arxiv.org/abs/2002.08127), [poster](images/uai-poster.pdf), [video](images/8min-video.mp4), [short slides](images/brief-slides.pdf), [full slides](images/full-slides.pdf)]

## Introduction

This repository contains the official implementation of the structured spasification algorithm for model compression, which converts the vanilla convolutions into GroupConvs and learns a channel shuffle pattern between consecutive GroupConvs.

## Overview

<figure>
  <img src="images/overview.png" alt="Overview of structured sparsification" width="720" height="300">
</figure>

## Reproduce the Experimental Results

### Dependencies

Please make sure the following packages are installed in your environment:

| **Package**    | **Version**              |
|----------------|--------------------------|
| python         |  >=3.6                   |
| pytorch        |  >=1.2                   |
| tensorboardX   |  >=2.0                   |
| thop           |  ==0.0.31.post2005241907 |
| POT            |  ==0.7.0                 |

### Compression Results on CIFAR

One can simply run `train_cifar.py` to reproduce the compression results on CIFAR classification benchmark (Table 1 in our [paper](https://arxiv.org/abs/2002.08127)). An exemplary script is given below:
```
CUDA_VISIBLE_DEVICES='0' python3 train_cifar.py -a "resnet20" --data "./data" --dataset "cifar10" --save-path "results/cifar10-resnet20-prune-percent-0.4" --prune-percent "0.4"
```
Here, one can specify the GPU id with `CUDA_VISIBLE_DEVICES`, and one GPU is sufficient in most cases. `-a` specifies the architecture, which can be chosen from `["resnet20", "resnet56", "resnet110"]`, and `--prune-percent` denotes the percent of parameters to be pruned.

### Compression Results on ImageNet

#### Extra Dependency and Dataset Preparation

Our ImageNet experiment is based on the [NVIDIA-DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/) pipeline. Please use the following script to install DALI:
```
pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali==0.13.0
```

Before running the codes, we need to prepare the ImageNet recorder. You have to download the original ImageNet dataset on your server. Please refer to its official [instructions](http://image-net.org/download). The downloaded files should be originzed in the following structure:
```
/your-download-path
├── train
│   ├── n01440764
│   │   ├── n01440764_10470.JPEG
│   │   ├── n01440764_11151.JPEG
│   │   ├── n01440764_12021.JPEG
│   │   ├── ...
│   ├── n01443537
│   │   ├── n01443537_11513.JPEG
│   │   ├── n01443537_12098.JPEG
│   │   ├── n01443537_12507.JPEG
│   │   ├── ...
│   ├── n01484850
│   │   ├── n01484850_10370.JPEG
│   │   ├── n01484850_1054.JPEG
│   │   ├── n01484850_13243.JPEG
│   │   ├── ...
│   ├── ...
├── val
│   ├── n01440764
│   │   ├── ILSVRC2012_val_00000293.JPEG
│   │   ├── ILSVRC2012_val_00002138.JPEG
│   │   ├── ILSVRC2012_val_00003014.JPEG
│   │   ├── ...
│   ├── ...
```

In order to build the image recorder (see [reference](https://cv.gluon.ai/build/examples_datasets/recordio.html#sphx-glr-download-build-examples-datasets-recordio-py)), run the following commands:
```
# For training data
python3 imagenet/im2rec.py /your-download-path/train /your-download-path/train/ --recursive --list --num-thread 8
python3 imagenet/im2rec.py /your-download-path/train /your-download-path/train/ --recursive --pass-through --pack-label --num-thread 8

# For validation data
python3 imagenet/im2rec.py /your-download-path/val /your-download-path/val/ --recursive --list --num-thread 8
python3 imagenet/im2rec.py /your-download-path/val /your-download-path/val/ --recursive --pass-through --pack-label --no-shuffle --num-thread 8
```

P.S. Please make sure that your disk have enough space available, since this operation will copy the whole ImageNet (>=150G will be enough). Besides, absolute path is preferred in the above arguments. Running these commands may take a while.

Finally, under `/your-download-path`, there should be six files:
```
/your-download-path
├── train.idx
├── train.lst
├── train.rec
├── val.idx
├── val.lst
├── val.rec
```

You only need these six files to run experiments on ImageNet. Other files are at your disposal.

#### Code Usage

Use `train_imagenet.py` file to reproduce our compression results on Imagenet (Table 2 in our [paper](https://arxiv.org/abs/2002.08127)). For example, to compress 35% parameters of ResNet-50, run

```
# ResNet-50 with 35% parameters pruned
CUDA_VISIBLE_DEVICES='0,1,2,3' python3 -m torch.distributed.launch --nproc_per_node=4 train_imagenet.py \
                -a "resnet50" \
                -j "4" \
                --data "/path/to/rec" \
                --save-path "results/imagenet-resnet50-prune-0.35" \
                --batch-size "64" \
                --prune-percent "0.35"
```
Here, the arguments `-a` and `--prune-percent` have the same meaning as in the CIFAR experiments. One can choose an architecture from `["resnet50", "resnet101", "densenet201"]`. Notably. the argument `--batch-size` denotes the number of training examples **per GPU** in each batch. If the number of GPUs used for training is changed, the number passed to `--batch-size` should be adapted accordingly.

## Citation

If you find our work intersting or helpful to your research, please consider citing our paper.

```
@InProceedings{zhang2021structured,
    title = {Structured Sparsification with Joint Optimization of Group Convolution and Channel Shuffle},
    author = {Zhang, Xin-Yu and Zhao, Kai and Xiao, Taihong and Cheng, Ming-Ming and Yang, Ming-Hsuan},
    booktitle = {Proceedings of the 37th Conference on Uncertainty in Artificial Intelligence (UAI)},
    year = {2021}
}
```

