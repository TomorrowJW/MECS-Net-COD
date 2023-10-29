import torch
import torch.nn.functional as F
import torch.nn as nn
from .Res2Net import res2net50_v1b_26w_4s
from .Align import BasicConv2d, TransBasicConv2d, ChannelAttention, SpatialAttention
from .HM import HAIM

'''
定义文中描述的掩码和边缘共同引导的可分离注意力模块，这里写的是早期命名Edge_Fore_Back
Define the Mask-and-Edge Co-Guided Separable Attention (MECSA) Module described in our paper
'''
class Edge_Fore_Back(nn.Module):
    def __init__(self,inchannel):
        super().__init__()

        self.edge_stream = nn.Sequential(BasicConv2d(inchannel,inchannel,1,1,0),BasicConv2d(inchannel,inchannel,3,1,1))
        self.fore_stream = nn.ModuleList([BasicConv2d(inchannel, inchannel, 1, 1, 0), BasicConv2d(inchannel, inchannel//2, 3, 1, 1)])
        self.back_stream = nn.ModuleList([BasicConv2d(inchannel, inchannel, 1, 1, 0), BasicConv2d(inchannel, inchannel//2, 3, 1, 1)])

        self.edge_head = nn.Conv2d(inchannel, 1, 1, 1, 0)

    def forward(self,Feature_Map,Prediction):

        Pre = Prediction.detach()
        pre = torch.sigmoid(Pre)

        edge = self.edge_stream(Feature_Map) + Feature_Map
        fore = self.fore_stream[1](self.fore_stream[0](Feature_Map) * edge + Feature_Map)
        back = self.back_stream[1](self.back_stream[0](Feature_Map) * edge + Feature_Map)

        edge = self.edge_head(edge)
        fore = fore * pre
        back = back * (1-pre)

        return edge, fore, back

class fusion_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.conv = BasicConv2d(768,512,3,1,1)

    def forward(self,o2,o3,o4):
        o4 = self.upsample4(o4)
        o3 = self.upsample2(o3)
        o = torch.cat((o2,o3,o4),1)
        o = self.conv(o)

        return o

'''
定义decoder
'''
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        self.HM1 = HAIM(512)
        self.HM2 = HAIM(512)
        self.HM3 = HAIM(512)
        self.HM4 = HAIM(512)
        self.HM5 = HAIM(512)

        self.fusion = fusion_layer()

        self.decoder5_0 = nn.ModuleList([BasicConv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                        BasicConv2d(512, 256, kernel_size=1, stride=1, padding=0)])
        self.d_out5 = nn.Sequential(nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0))
        self.e_f_b5 = Edge_Fore_Back(256)

        self.decoder4_0 = nn.ModuleList([BasicConv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                        BasicConv2d(512, 256, kernel_size=1, stride=1, padding=0)])
        self.d_out4 = nn.Sequential(nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0))
        self.e_f_b4 = Edge_Fore_Back(256)

        self.decoder3_0 = nn.ModuleList([BasicConv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                        BasicConv2d(512, 256, kernel_size=1, stride=1, padding=0)])
        self.d_out3 = nn.Sequential(nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0))
        self.e_f_b3 = Edge_Fore_Back(256)

        self.decoder2_0 =  nn.ModuleList([BasicConv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                        BasicConv2d(512, 256, kernel_size=1, stride=1, padding=0)])
        self.d_out2 = nn.Sequential(nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0))
        self.e_f_b2 = Edge_Fore_Back(256)

        self.decoder1_0 = nn.ModuleList([BasicConv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                        BasicConv2d(512, 256, kernel_size=1, stride=1, padding=0)])
        self.d_out1 = nn.Sequential(nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0))
        self.e_f_b1 = Edge_Fore_Back(256)


    def forward(self,  o1, o2, o3, o4): #[b,64,96,96],[b,128,48,48],[b,256,24,24],[b,256,12,12]
        o = self.fusion(o2,o3,o4) #融合层用于生成文中所描述的O5(The fusion layer is used to generate O5 as described in our paper)

        s5_up = self.decoder5_0[0](o)

        s5_up = self.HM5(s5_up)
        s5_up = self.decoder5_0[1](s5_up)
        s5_out = self.d_out5(s5_up)
        e5,f5,b5 = self.e_f_b5(s5_up,s5_out)
        s5 = self.upsample8(s5_out)
        e5 = self.upsample8(e5)

        s4_up = self.decoder4_0[0](torch.cat((torch.cat((f5,b5),1)*self.upsample4(o4) + self.upsample4(o4),s5_up),1))
        s4_up = self.HM4(s4_up)
        s4_up = self.decoder4_0[1](s4_up)
        s4_out = self.d_out4(s4_up)
        e4,f4,b4 = self.e_f_b4(s4_up,s4_out)
        s4 = self.upsample8(s4_out)
        e4 = self.upsample8(e4)

        s3_up = self.decoder3_0[0](torch.cat((torch.cat((f4,b4),1)*self.upsample2(o3) + self.upsample2(o3),s5_up),1))
        s3_up = self.HM3(s3_up)
        s3_up = self.decoder3_0[1](s3_up)
        s3_out = self.d_out3(s3_up)
        e3,f3,b3 = self.e_f_b3(s3_up,s3_out)
        s3 = self.upsample8(s3_out)
        e3 = self.upsample8(e3)

        s2_up = self.decoder2_0[0](torch.cat((torch.cat((f3,b3),1)*o2 + o2,s5_up),1))
        s2_up = self.HM2(s2_up)
        s2_up = self.decoder2_0[1](s2_up)
        s2_out = self.d_out2(s2_up)
        e2,f2,b2 = self.e_f_b2(s2_up,s2_out)
        s2 = self.upsample8(s2_out)
        e2 = self.upsample8(e2)

        s1_up = self.decoder1_0[0](torch.cat((torch.cat((self.upsample2(f2),self.upsample2(b2)),1)*o1 + o1,self.upsample2(s5_up)),1))
        s1_up = self.HM1(s1_up)
        s1_up = self.decoder1_0[1](s1_up)
        s1_out = self.d_out1(s1_up)
        e1,f1,b1 = self.e_f_b1(s1_up,s1_out)
        s1 = self.upsample4(s1_out)
        e1 = self.upsample4(e1)

        return s1,s2,s3,s4,s5,e1,e2,e3,e4,e5

'''
定义我们的模型，这里写的是早期命名
Define the Proposed model(MECS-Net)
'''

class CL_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.res2net = res2net50_v1b_26w_4s(pretrained=False)

        self.sigmoid = nn.Sigmoid()

        self.conv1 = BasicConv2d(256, 256, 1,stride=1,padding=0)
        self.conv2 = BasicConv2d(512, 256, 1, stride=1, padding=0)
        self.conv3 = BasicConv2d(1024, 256, 1, stride=1, padding=0)
        self.conv4 = BasicConv2d(2048, 256, 1, stride=1, padding=0)

        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

        self.decoder = Decoder()

        if self.training:
            self.initialize_weights()


    def forward(self, rgb):
        stage_rgb = self.res2net(rgb)

        s1 = stage_rgb[1] #[b,256,96,96]
        s2 = stage_rgb[2] #[b,512,48,48]
        s3 = stage_rgb[3] #[b,1024,24,24]
        s4 = stage_rgb[4] #[b,2048,12,12]

        c1 = self.conv1(s1)#[b,64,96,96]
        c2 = self.conv2(s2)#[b,128,48,48]
        c3 = self.conv3(s3)#[b,256,24,24]
        c4 = self.conv4(s4)#[b,256,12,12]

        f1 = c1 #[b,64,96,96]
        f2 = c2 #[b,128,48,48]
        f3 = c3 #[b,256,24,24]
        f4 = c4#[b,256,12,12]

        s1,s2,s3,s4,s5,e1,e2,e3,e4,e5 = self.decoder(f1,f2,f3,f4)

        return s1,s2,s3,s4,s5,e1,e2,e3,e4,e5

    def initialize_weights(self):  # 加载预训练模型权重，做初始化
        self.res2net.load_state_dict(torch.load('./pre_train/res2net50_v1b_26w_4s-3cf99910.pth'))
