# Pyramid Stereo Matching Network

This repository contains the code (in PyTorch) for "[Pyramid Stereo Matching Network](https://arxiv.org/abs/1803.08669)" paper (CVPR 2018) by [Jia-Ren Chang](https://jiarenchang.github.io/) and [Yong-Sheng Chen](https://people.cs.nctu.edu.tw/~yschen/).

#### changelog
2020/12/20: Update PSMNet: now support torch 1.6.0 / torchvision 0.5.0 and python 3.7, Removed inconsistent indentation.

2020/12/20: Our proposed Real-Time Stereo can be found here [Real-time Stereo](https://github.com/JiaRenChang/RealtimeStereo).
### Citation
```
@inproceedings{chang2018pyramid,
  title={Pyramid Stereo Matching Network},
  author={Chang, Jia-Ren and Chen, Yong-Sheng},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5410--5418},
  year={2018}
}
```

## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Results](#results)
4. [Contacts](#contacts)

## Introduction

Recent work has shown that depth estimation from a stereo pair of images can be formulated as a supervised learning task to be resolved with convolutional neural networks (CNNs). However, current architectures rely on patch-based Siamese networks, lacking the means to exploit context information for finding correspondence in illposed regions. To tackle this problem, we propose PSMNet, a pyramid stereo matching network consisting of two main modules: spatial pyramid pooling and 3D CNN. The spatial pyramid pooling module takes advantage of the capacity of global context information by aggregating context in different scales and locations to form a cost volume. The 3D CNN learns to regularize cost volume using stacked multiple hourglass networks in conjunction with intermediate supervision.

<img align="center" src="https://user-images.githubusercontent.com/11732099/43501836-1d32897c-958a-11e8-8083-ad41ec26be17.jpg">

## Usage

### Dependencies

- [Python 3.7](https://www.python.org/downloads/)
- [PyTorch(1.6.0+)](http://pytorch.org)
- torchvision 0.5.0
- [KITTI Stereo](http://www.cvlibs.net/datasets/kitti/eval_stereo.php)
- [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

```
Usage of Scene Flow dataset
Download RGB cleanpass images and its disparity for three subset: FlyingThings3D, Driving, and Monkaa.
Put them in the same folder.
And rename the folder as: "driving_frames_cleanpass", "driving_disparity", "monkaa_frames_cleanpass", "monkaa_disparity", "frames_cleanpass", "frames_disparity".
```
### Notice
1. Warning of upsample function in PyTorch 0.4.1+: add "align_corners=True" to upsample functions.
2. Output disparity may be better with multipling by 1.17. Reported from issues [#135](https://github.com/JiaRenChang/PSMNet/issues/135) and [#113](https://github.com/JiaRenChang/PSMNet/issues/113).

### Train
As an example, use the following command to train a PSMNet on Scene Flow

```
python main.py --maxdisp 192 \
               --model stackhourglass \
               --datapath (your scene flow data folder)\
               --epochs 10 \
               --loadmodel (optional)\
               --savemodel (path for saving model)
```

As another example, use the following command to finetune a PSMNet on KITTI 2015

```
python finetune.py --maxdisp 192 \
                   --model stackhourglass \
                   --datatype 2015 \
                   --datapath (KITTI 2015 training data folder) \
                   --epochs 300 \
                   --loadmodel (pretrained PSMNet) \
                   --savemodel (path for saving model)
```
You can also see those examples in run.sh.

### Evaluation
Use the following command to evaluate the trained PSMNet on KITTI 2015 test data

```
python submission.py --maxdisp 192 \
                     --model stackhourglass \
                     --KITTI 2015 \
                     --datapath (KITTI 2015 test data folder) \
                     --loadmodel (finetuned PSMNet) \
```

### Pretrained Model
※NOTE: The pretrained model were saved in .tar; however, you don't need to untar it. Use torch.load() to load it.

Update: 2018/9/6 We released the pre-trained KITTI 2012 model.

Update: 2021/9/22 a pretrained model using torch 1.8.1 (the previous model weight are trained torch 0.4.1)

| KITTI 2015 |  Scene Flow | KITTI 2012 | Scene Flow (torch 1.8.1)
|---|---|---|---|
|[Google Drive](https://drive.google.com/file/d/1pHWjmhKMG4ffCrpcsp_MTXMJXhgl3kF9/view?usp=sharing)|[Google Drive](https://drive.google.com/file/d/1xoqkQ2NXik1TML_FMUTNZJFAHrhLdKZG/view?usp=sharing)|[Google Drive](https://drive.google.com/file/d/1p4eJ2xDzvQxaqB20A_MmSP9-KORBX1pZ/view?usp=sharing)| [Google Drive](https://drive.google.com/file/d/1NDKrWHkwgMKtDwynXVU12emK3G5d5kkp/view?usp=sharing)

### Test on your own stereo pair
```
python Test_img.py --loadmodel (finetuned PSMNet) --leftimg ./left.png --rightimg ./right.png
```

## Results

### Evaluation of PSMNet with different settings
<img align="center" src="https://user-images.githubusercontent.com/11732099/37817886-45a12ece-2eb3-11e8-8254-ae92c723b2f6.png">

※Note that the reported 3-px validation errors were calculated using KITTI's official matlab code, not our code.

### Results on KITTI 2015 leaderboard
[Leaderboard Link](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

| Method | D1-all (All) | D1-all (Noc)| Runtime (s) |
|---|---|---|---|
| PSMNet | 2.32 % | 2.14 % | 0.41 |
| [iResNet-i2](https://arxiv.org/abs/1712.01039) | 2.44 % | 2.19 % | 0.12 |
| [GC-Net](https://arxiv.org/abs/1703.04309) | 2.87 % | 2.61 % | 0.90 |
| [MC-CNN](https://github.com/jzbontar/mc-cnn) | 3.89 % | 3.33 % | 67 |

### Qualitative results
#### Left image
<img align="center" src="http://www.cvlibs.net/datasets/kitti/results/efb9db97938e12a20b9c95ce593f633dd63a2744/image_0/000004_10.png">

#### Predicted disparity
<img align="center" src="http://www.cvlibs.net/datasets/kitti/results/efb9db97938e12a20b9c95ce593f633dd63a2744/result_disp_img_0/000004_10.png">

#### Error
<img align="center" src="http://www.cvlibs.net/datasets/kitti/results/efb9db97938e12a20b9c95ce593f633dd63a2744/errors_disp_img_0/000004_10.png">

### Visualization of Receptive Field
We visualize the receptive fields of different settings of PSMNet, full setting and baseline.

Full setting: dilated conv, SPP, stacked hourglass

Baseline: no dilated conv, no SPP, no stacked hourglass

The receptive fields were calculated for the pixel at image center, indicated by the red cross.

<img align="center" src="https://user-images.githubusercontent.com/11732099/37876179-6d6dd97e-307b-11e8-803e-bcdbec29fb94.png">



## Contacts
followwar@gmail.com

Any discussions or concerns are welcomed!


# 记录
## 安装
```bash
# 创建anaconda环境
conda create -n psmnet python=3.7
# 检查本机cuda版本: CUDA Version: 11.4
# 安装pytorch(1.6.0+)和torchvision(0.5.0)==>原repo要求的版本
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch

pip3 install opencv-python
pip3 install chardet  (current version 5.1.0)
```
[chardet用处](https://www.liaoxuefeng.com/wiki/1016959663602400/1183255880134144)

检查pytorch是否安装成功
```python
import torch
torch.cuda.is_available()  # 检查是否可用cuda
torch.zeros([3,3])    # 检查cpu是否可以创建tensor
torch.zeros([3,3], device='cuda')    # 检查gpu是否可以创建tensor
```
下载Scene Flow数据集上的`预训练模型`并进行测试
```bash
下载地址：https://drive.google.com/file/d/1xoqkQ2NXik1TML_FMUTNZJFAHrhLdKZG/view
在项目文件夹PSMNet下新建文件夹:pretrained_model/scene_flow/
并将下载的模型复制到新建的文件夹中
```
下载Scene Flow数据集的Sample进行测试 \
测试图片[下载地址](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html#downloads)  \
下载[Example pack](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlow/assets/Sampler.tar.gz)  \
将下载的数据解压到`/PSMNet/test_img_data/`中，即在此文件夹中存在`Sampler`文件夹。  \
测试命令：
```bash
python3 Test_img.py  \
--loadmodel ./pretrained_model/scene_flow/pretrained_sceneflow.tar  \
--leftimg ./test_img_data/Sampler/FlyingThings3D/RGB_cleanpass/left/0006.png \
--rightimg ./test_img_data/Sampler/FlyingThings3D/RGB_cleanpass/right/0006.png
```
图像的大小为540*960，用自己的图像测试一下！

## 训练
下载数据集 \
```bash
python3 main.py --maxdisp 192 --model stackhourglass --datapath /data/net/dl_data/ProjectDatasets_bkx/sceneflow --epoch 10 --savemodel ./trained_model/
```
报错：  \
