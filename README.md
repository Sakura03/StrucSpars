# Structured Sparsification with Joint Optimization of Group Convolution and Channel Shuffle

Official implementation of paper: **Structured Sparsification with Joint Optimization of Group Convolution and Channel Shuffle** (UAI 2021), by Xin-Yu Zhang, Kai Zhao, Taihong Xiao, Ming-Ming Cheng, and Ming-Hsuan Yang. [[paper](https://arxiv.org/abs/2002.08127), [poster](images/aistats-poster.pdf), [video](images/8min-video.mp4), [short slides](images/brief-slides.pdf), [full slides](images/full-slides.pdf)]

## Introduction

This repository contains the official implementation of the structured spasification algorithm for model compression, which converts the vanilla convolutions into GroupConvs and learns a channel shuffle pattern between consecutive GroupConvs.

## Overview

<figure>
  <img src="images/overview.png" alt="Overview of structured sparsification" width="720" height="240">
</figure>

## Reproduce the Experimental Results

### Prerequisite

Please make sure the following packages are installed in your environment:

| **Package**    | **Version**              |
|----------------|--------------------------|
| python         |  >=3.6                   |
| pytorch        |  >=1.2                   |
| tensorboardX   |  >=2.0                   |
| thop           |  ==0.0.31.post2005241907 |
| POT            |  ==0.7.0                 |

Besides, our ImageNet experiment is based on the [NVIDIA-DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/) pipeline. Please use the following script to install DALI:
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

In order to build the image recorder (See [reference](https://cv.gluon.ai/build/examples_datasets/recordio.html#sphx-glr-download-build-examples-datasets-recordio-py)). Run the following commands:
```
# For training data
python3 im2rec.py /your-download-path/train /your-download-path/train/ --recursive --list --num-thread 8
python3 im2rec.py /your-download-path/train /your-download-path/train/ --recursive --pass-through --pack-label --num-thread 8

# For validation data
python im2rec.py /your-download-path/val /your-download-path/val/ --recursive --list --num-thread 8
python im2rec.py /your-download-path/val /your-download-path/val/ --recursive --pass-through --pack-label --no-shuffle --num-thread 8
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
