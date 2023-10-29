import torch.nn as nn
import torch
'''
定义基本卷积，注意力机制和一些基本组件
#define basic convolution, attention, and components
'''

'''
#定义CBR组合
define CBR combination
'''
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

'''
定义转置卷积CBR
Define transposed convolution CBR component
'''
class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1,output_padding=0):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, output_padding=output_padding,bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

'''
定义通道注意力机制
Define channelattention 
'''
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
                            nn.Conv2d(in_planes,in_planes//ratio,1,bias=False),nn.ReLU(),
                            nn.Conv2d(in_planes//ratio,in_planes,1,bias=False))
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout+maxout)

'''
定义空间注意力机制
Define sptialattention 
'''
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x,dim=1,keepdim=True)
        maxout,_ = torch.max(x,dim=1,keepdim=True)
        x = torch.cat([avgout,maxout],dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

'''
定义CBAM组件
Define CBAM component
'''
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):  # x: [b,c,h,w]
        out = x * self.ca(x)
        result = out * self.sa(out)
        # print(result.size())
        return result  # [b,c,h,w]


