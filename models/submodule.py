from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
        nn.BatchNorm3d(out_planes))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class disparityregression(nn.Module):
    def __init__(self, maxdisp, device='gpu'):
        super(disparityregression, self).__init__()
        self.device = device
        if self.device == 'cpu':
            self.disp = torch.Tensor(np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1]))
        elif self.device == 'gpu':
            self.disp = torch.Tensor(np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1])).cuda()

    def forward(self, x):
        out = torch.sum(x * self.disp.data, 1, keepdim=True)
        return out


class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)  # 空洞率修改为和论文一致
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 4)  # 空洞率修改为和论文一致

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64, 64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):  # [1,3,544,960]
        output = self.firstconv(x)  # [1,32,272,480]
        output = self.layer1(output)  # [1,32,272,480]
        output_raw = self.layer2(output)  # [1,64,136,240]
        output = self.layer3(output_raw)  # [1,128,136,240]
        output_skip = self.layer4(output)  # [1,128,136,240]

        # 将F.upsample修改为使用scale_factor指定倍率，而不是使用size参数指定大小，防止后续tensorrt报错
        # 用F.interpolate代替F.unsample，因为F.unsample在Pytorch高版本中被弃用；参数scale_factor应为tuple，分别代表高和宽的放大倍率，参数align_corners设置为False以保证兼容性和预期
        # 每个分支都根据自己的尺度进行了独立的上采样。这种方法在进行模型转换（如PyTorch到TensorRT）时更稳定，可以避免因使用size参数而引发的问题。
        output_branch1 = self.branch1(output_skip)  # [1,32,2,3]
        # output_branch1 = F.upsample(output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear') # [1,32,136,240]
        scale_factor_branch1_h = output_skip.size()[2] / output_branch1.size()[2]
        scale_factor_branch1_w = output_skip.size()[3] / output_branch1.size()[3]
        output_branch1 = F.interpolate(output_branch1, scale_factor=(scale_factor_branch1_h, scale_factor_branch1_w),
                                       mode='bilinear', align_corners=False)

        output_branch2 = self.branch2(output_skip)  # [1,32,4,7]
        # output_branch2 = F.upsample(output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')  # [1,32,136,240]
        scale_factor_branch2_h = output_skip.size()[2] / output_branch2.size()[2]
        scale_factor_branch2_w = output_skip.size()[3] / output_branch2.size()[3]
        output_branch2 = F.interpolate(output_branch2, scale_factor=(scale_factor_branch2_h, scale_factor_branch2_w),
                                       mode='bilinear', align_corners=False)

        output_branch3 = self.branch3(output_skip)  # [1,32,8,15]
        # output_branch3 = F.upsample(output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')  # [1,32,136,240]
        scale_factor_branch3_h = output_skip.size()[2] / output_branch3.size()[2]
        scale_factor_branch3_w = output_skip.size()[3] / output_branch3.size()[3]
        output_branch3 = F.interpolate(output_branch3, scale_factor=(scale_factor_branch3_h, scale_factor_branch3_w),
                                       mode='bilinear', align_corners=False)

        output_branch4 = self.branch4(output_skip)  # [1,32,17,30]
        # output_branch4 = F.upsample(output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')  # [1,32,136,240]
        scale_factor_branch4_h = output_skip.size()[2] / output_branch4.size()[2]
        scale_factor_branch4_w = output_skip.size()[3] / output_branch4.size()[3]
        output_branch4 = F.interpolate(output_branch4, scale_factor=(scale_factor_branch4_h, scale_factor_branch4_w),
                                       mode='bilinear', align_corners=False)

        output_feature = torch.cat(
            (output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)  # [1,320,136,240], 这里的320=64+128+32+32+32+32
        output_feature = self.lastconv(output_feature)  # [1,32,136,240]

        return output_feature
