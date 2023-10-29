import torch
import torch.nn as nn
from torch.nn import init
from models.Align import BasicConv2d, ChannelAttention, SpatialAttention


'''
定义文中描述的多尺度增强融合模块 (MEF Module)，这里写的是早期命名HAIM
Define the Multi-Scale Enhancement Fusion Module (MEF Module) described in our paper
'''
class HAIM(nn.Module):
    def __init__(self, in_channel, rate=[1,3,5,7]):
        super(HAIM, self).__init__()

        self.rgb_b1 = nn.Sequential(BasicConv2d(in_channel, in_channel//4, 1,1,0),
                                    BasicConv2d(in_channel//4, in_channel//4, 3, padding=rate[0], dilation=rate[0]))
        self.rgb_b2 = nn.Sequential(BasicConv2d(in_channel, in_channel//4, 1,1,0),
                                    BasicConv2d(in_channel//4, in_channel//4, 3, padding=rate[1], dilation=rate[1]))
        self.rgb_b3 = nn.Sequential(BasicConv2d(in_channel, in_channel//4, 1,1,0),
                                    BasicConv2d(in_channel//4, in_channel//4, 3, padding=rate[2], dilation=rate[2]))
        self.rgb_b4 = nn.Sequential(BasicConv2d(in_channel, in_channel//4, 1,1,0),
                                    BasicConv2d(in_channel//4, in_channel//4, 3, padding=rate[3], dilation=rate[3]))

        self.rgb_b1_sa = SpatialAttention()
        self.rgb_b2_sa = SpatialAttention()
        self.rgb_b3_sa = SpatialAttention()
        self.rgb_b4_sa = SpatialAttention()

        self.rgb_b1_ca = ChannelAttention(in_channel//4)
        self.rgb_b2_ca = ChannelAttention(in_channel//4)
        self.rgb_b3_ca = ChannelAttention(in_channel//4)
        self.rgb_b4_ca = ChannelAttention(in_channel//4)

        self.ca = ChannelAttention(in_channel)
        self.identity = BasicConv2d(in_channel,in_channel,kernel_size=1)
        self.out_conv = nn.Conv2d(in_channel, in_channel, kernel_size=1,padding=0,stride=1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, RGB):
        rgb = RGB
        x1_rgb = self.rgb_b1(rgb)
        x2_rgb = self.rgb_b2(rgb)
        x3_rgb = self.rgb_b3(rgb)
        x4_rgb = self.rgb_b4(rgb)

        '''
        下面四行代码是实现文中描述的增强融合组件 (FE component)
        Define the feature enhancement (FE) component described in our paper
        '''
        x1_rgb_f = x1_rgb + \
                   x1_rgb*\
                   self.rgb_b1_sa(x1_rgb*self.rgb_b1_ca(x1_rgb))

        x2_rgb_f = (x2_rgb+x1_rgb_f) + \
                   (x2_rgb+x1_rgb_f) * self.rgb_b2_sa((x2_rgb+x1_rgb_f) * self.rgb_b2_ca(x2_rgb+x1_rgb_f))

        x3_rgb_f = (x3_rgb+x2_rgb_f) + \
                   (x3_rgb+x2_rgb_f) * self.rgb_b3_sa((x3_rgb+x2_rgb_f) * self.rgb_b3_ca(x3_rgb+x2_rgb_f))

        x4_rgb_f = (x4_rgb+x3_rgb_f) + \
                   (x4_rgb+x3_rgb_f) * self.rgb_b4_sa((x4_rgb+x3_rgb_f) * self.rgb_b4_ca(x4_rgb+x3_rgb_f))

        y = torch.cat((x1_rgb_f,x2_rgb_f,x3_rgb_f,x4_rgb_f),1)
        y_ca = y.mul(self.ca(y))
        z = self.out_conv(y_ca + rgb)

        return z


if __name__ == '__main__':
    a = torch.randn((2, 256,12,12))
    b = torch.Tensor(a).cuda()
    res = HAIM(256).cuda()
    out = res(b)
    print(out.size())