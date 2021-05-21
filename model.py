import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# import pytorch_colors as colors
import numpy as np
import cv2 as cv
import torchvision

output_channels_1 = 24
output_channels_2 = 16

class Block(nn.Module):
    """ implement conv+ReLU two times """
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        conv_relu = []
        conv_relu.append(nn.Conv2d(in_channels=in_channels, out_channels=middle_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.ReLU())
        conv_relu.append(nn.Conv2d(in_channels=middle_channels, out_channels=out_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.ReLU())
        self.conv_ReLU = nn.Sequential(*conv_relu)
    def forward(self, x):
        out = self.conv_ReLU(x)
        return out

#四层U_Net
class U_Net_1(nn.Module):
    def __init__(self):
        super().__init__()

        # 首先定义左半部分网络
        # left_conv_1 表示连续的两个（卷积+激活）
        # 随后进行最大池化
        self.ReLU = nn.ReLU()
        self.left_conv_1 = Block(in_channels=3, middle_channels=output_channels_1, out_channels=output_channels_1)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_2 = Block(in_channels=output_channels_1, middle_channels=2*output_channels_1, out_channels=2*output_channels_1)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_3 = Block(in_channels=2*output_channels_1, middle_channels=4*output_channels_1, out_channels=4*output_channels_1)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_4 = Block(in_channels=4*output_channels_1, middle_channels=8*output_channels_1, out_channels=8*output_channels_1)

        # 定义右半部分网络
        self.deconv_1 = nn.ConvTranspose2d(in_channels=8*output_channels_1, out_channels=4*output_channels_1, kernel_size=2, stride=2)
        self.right_conv_1 = Block(in_channels=8*output_channels_1, middle_channels=4*output_channels_1, out_channels=4*output_channels_1)

        self.deconv_2 = nn.ConvTranspose2d(in_channels=4*output_channels_1, out_channels=2*output_channels_1, kernel_size=2, stride=2)
        self.right_conv_2 = Block(in_channels=4*output_channels_1, middle_channels=2*output_channels_1, out_channels=2*output_channels_1)

        self.deconv_3 = nn.ConvTranspose2d(in_channels=2*output_channels_1, out_channels=output_channels_1, kernel_size=2, stride=2)
        self.right_conv_3 = Block(in_channels=2*output_channels_1, middle_channels=output_channels_1, out_channels=output_channels_1)
        # 最后是1x1的卷积，用于将通道数化为3
        self.right_conv_4 = nn.Conv2d(in_channels=output_channels_1, out_channels=3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        # 1：进行编码过程
        feature_1 = self.left_conv_1(x)
        feature_1_pool = self.pool_1(feature_1)

        feature_2 = self.left_conv_2(feature_1_pool)
        feature_2_pool = self.pool_2(feature_2)

        feature_3 = self.left_conv_3(feature_2_pool)
        feature_3_pool = self.pool_3(feature_3)

        feature_4 = self.left_conv_4(feature_3_pool)

        # 2：进行解码过程
        de_feature_1 = self.deconv_1(feature_4)
        de_feature_1 = self.ReLU(de_feature_1)

        # 特征拼接
        temp = torch.cat((feature_3, de_feature_1), dim=1)
        de_feature_1_conv = self.right_conv_1(temp)

        de_feature_2 = self.deconv_2(de_feature_1_conv)
        de_feature_2 = self.ReLU(de_feature_2)
        temp = torch.cat((feature_2, de_feature_2), dim=1)
        de_feature_2_conv = self.right_conv_2(temp)

        de_feature_3 = self.deconv_3(de_feature_2_conv)
        de_feature_3 = self.ReLU(de_feature_3)
        temp = torch.cat((feature_1, de_feature_3), dim=1)
        de_feature_3_conv = self.right_conv_3(temp)

        out = self.right_conv_4(de_feature_3_conv )

        return out

#三层U_Net
class U_Net_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.ReLU = nn.ReLU()

        # 首先定义左半部分网络
        # left_conv_1 表示连续的两个（卷积+激活）
        # 随后进行最大池化
        self.left_conv_1 = Block(in_channels=3, middle_channels=output_channels_1, out_channels=output_channels_1)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_2 = Block(in_channels=output_channels_1, middle_channels=2*output_channels_1, out_channels=2*output_channels_1)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_3 = Block(in_channels=2*output_channels_1, middle_channels=4*output_channels_1, out_channels=4*output_channels_1)

        # 定义右半部分网络
        self.deconv_1 = nn.ConvTranspose2d(in_channels=4*output_channels_1, out_channels=2*output_channels_1, kernel_size=2, stride=2)
        self.right_conv_1 = Block(in_channels=4*output_channels_1, middle_channels=2*output_channels_1, out_channels=2*output_channels_1)

        self.deconv_2 = nn.ConvTranspose2d(in_channels=2*output_channels_1, out_channels=output_channels_1, kernel_size=2, stride=2)
        self.right_conv_2 = Block(in_channels=2*output_channels_1, middle_channels=output_channels_1, out_channels=output_channels_1)
        # 最后是1x1的卷积，用于将通道数化为3
        self.right_conv_3 = nn.Conv2d(in_channels=output_channels_1, out_channels=3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        # 1：进行编码过程
        feature_1 = self.left_conv_1(x)
        feature_1_pool = self.pool_1(feature_1)

        feature_2 = self.left_conv_2(feature_1_pool)
        feature_2_pool = self.pool_2(feature_2)

        feature_3 = self.left_conv_3(feature_2_pool)

        # 2：进行解码过程
        de_feature_1 = self.deconv_1(feature_3)
        de_feature_1 = self.ReLU(de_feature_1)
        # 特征拼接
        temp = torch.cat((feature_2, de_feature_1), dim=1)
        de_feature_1_conv = self.right_conv_1(temp)

        de_feature_2 = self.deconv_2(de_feature_1_conv)
        de_feature_2 = self.ReLU(de_feature_2)
        temp = torch.cat((feature_1, de_feature_2), dim=1)
        de_feature_2_conv = self.right_conv_2(temp)

        out = self.right_conv_3(de_feature_2_conv)

        return out

#三层U_Net
class U_Net_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.ReLU = nn.ReLU()

        # 首先定义左半部分网络
        # left_conv_1 表示连续的两个（卷积+激活）
        # 随后进行最大池化
        self.left_conv_1 = Block(in_channels=3, middle_channels=output_channels_2, out_channels=output_channels_2)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_2 = Block(in_channels=output_channels_2, middle_channels=2*output_channels_2, out_channels=2*output_channels_2)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_3 = Block(in_channels=2*output_channels_2, middle_channels=4*output_channels_2, out_channels=4*output_channels_2)

        # 定义右半部分网络
        self.deconv_1 = nn.ConvTranspose2d(in_channels=4*output_channels_2, out_channels=2*output_channels_2, kernel_size=2, stride=2)
        self.right_conv_1 = Block(in_channels=4*output_channels_2, middle_channels=2*output_channels_2, out_channels=2*output_channels_2)

        self.deconv_2 = nn.ConvTranspose2d(in_channels=2*output_channels_2, out_channels=output_channels_2, kernel_size=2, stride=2)
        self.right_conv_2 = Block(in_channels=2*output_channels_2, middle_channels=output_channels_2, out_channels=output_channels_2)
        # 最后是1x1的卷积，用于将通道数化为3
        self.right_conv_3 = nn.Conv2d(in_channels=output_channels_2, out_channels=3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        # 1：进行编码过程
        feature_1 = self.left_conv_1(x)
        feature_1_pool = self.pool_1(feature_1)

        feature_2 = self.left_conv_2(feature_1_pool)
        feature_2_pool = self.pool_2(feature_2)

        feature_3 = self.left_conv_3(feature_2_pool)

        # 2：进行解码过程
        de_feature_1 = self.deconv_1(feature_3)
        de_feature_1 = self.ReLU(de_feature_1)
        # 特征拼接
        temp = torch.cat((feature_2, de_feature_1), dim=1)
        de_feature_1_conv = self.right_conv_1(temp)

        de_feature_2 = self.deconv_2(de_feature_1_conv)
        de_feature_2 = self.ReLU(de_feature_2)
        temp = torch.cat((feature_1, de_feature_2), dim=1)
        de_feature_2_conv = self.right_conv_2(temp)

        out = self.right_conv_3(de_feature_2_conv)

        return out


#Multi_Scale_model
class PEC_model(nn.Module):
    """ implement conv+ReLU two times """
    def __init__(self):
        super().__init__()
        self.conv1 = U_Net_1()
        self.conv2 = U_Net_2()
        self.conv3 = U_Net_3()
        self.decov = usample()

    def forward(self, x_1, x_2, x_3, x_4):
        #进行第一层U-Net
        x1 = self.conv1(x_1)
        out_1 = self.decov(x1)
        y1 = out_1 + x_2

        #进行第二层U-Net
        x2 = self.conv2(y1)
        z2 = 0.6 * x2 + y1
        out_2 = self.decov(z2)
        y2 = out_2 + x_3

        #进行第三层U-Net
        x3 = self.conv2(y2)
        z3 = 0.4 * x3 + y2
        out_3 = self.decov(z3)
        y3 = out_3 + x_4

        #进行第四层U-Net
        x4 = self.conv3(y3)
        out_4 = 0.2 * x4 + y3

        return out_1, out_2, out_3, out_4

#Multi_Scale_model
class usample(nn.Module):
    def __init__(self):
        super().__init__()
        self.decov = nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=2, stride=2)

    def forward(self, x):
        out_1 = self.decov(x)
        out_1 = out_1.squeeze(1)
        out_2 = self.decov(x)
        out_2 = out_2.squeeze(1)
        out_3 = self.decov(x)
        out_3 = out_3.squeeze(1)
        out = torch.stack((out_1, out_2, out_3), dim=1)
        return out



"""
if __name__ == "__main__":
    x_1 = torch.rand(size=(1, 3, 64, 64))
    x_2 = torch.rand(size=(1, 3, 128, 128))
    x_3 = torch.rand(size=(1, 3, 256, 256))
    x_4 = torch.rand(size=(1, 3, 512, 512))
    net = PEC_model()
    out = net(x_1, x_2, x_3, x_4)
    print(out[0].size())
    print("ok")
"""