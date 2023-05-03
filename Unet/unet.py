import torch
import torch.nn as nn
from torch.nn import functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")  # GPU 可用
else:
    device = torch.device("cpu")  # 只能使用 CPU

# device = torch.device("cpu")  # 只能使用 CPU


class Unet(nn.Module):
    def __init__(self, KERNEL=3,PADDING=1):
        super(Unet, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)   # 池化层
        self.convt1=nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)   # 上采样层，1014->512
        self.convt2=nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)    # 对输入特征图进行2倍上采样操作，将每个像素点扩大为2x2的区域
        self.convt3=nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)    #  并使用反卷积操作来填充像素
        self.convt4=nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)


        self.conv_seq1 = nn.Sequential( nn.Conv2d(1,64,kernel_size=KERNEL,padding=PADDING),   # 卷积层，1->64
                                        nn.InstanceNorm2d(64),  # 例归一化层，对神经网络中的特征图进行归一化处理。输入特征图的通道数
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(64,64,kernel_size=KERNEL,padding=PADDING),
                                        nn.InstanceNorm2d(64),
                                        nn.ReLU(inplace=True))
        self.conv_seq2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(128),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(128, 128, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(128),
                                       nn.ReLU(inplace=True))
        self.conv_seq3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(256),
                                       nn.ReLU(inplace=True))
        self.conv_seq4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(512),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(512, 512, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(512),
                                       nn.ReLU(inplace=True))
        self.conv_seq5 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(1024),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(1024, 1024, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(1024),
                                       nn.ReLU(inplace=True))

        self.deconv_seq1 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(512),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout2d(p=0.5),  # 训练过程中随机删除输入的一部分神经元，以减少过拟合的风险
                                       nn.Conv2d(512, 512, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(512),
                                       nn.ReLU(inplace=True))
        self.deconv_seq2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(256),
                                       nn.ReLU(inplace=True),
                                        )
        self.deconv_seq3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=KERNEL, padding=PADDING),
                                         nn.InstanceNorm2d(128),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout2d(p=0.5),
                                         nn.Conv2d(128, 128, kernel_size=KERNEL, padding=PADDING),
                                         nn.InstanceNorm2d(128),
                                         nn.ReLU(inplace=True))
        self.deconv_seq4 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=KERNEL, padding=PADDING),
                                         nn.InstanceNorm2d(64),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout2d(p=0.5),
                                         nn.Conv2d(64, 64, kernel_size=KERNEL, padding=PADDING),
                                         nn.InstanceNorm2d(64),
                                         nn.ReLU(inplace=True), )
        self.deconv_seq5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=KERNEL, padding=PADDING),
                                         nn.InstanceNorm2d(64),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(64, 4, kernel_size=KERNEL, padding=PADDING))


        self.segdown2_seq = nn.Sequential(nn.Conv2d(256, 4, kernel_size=KERNEL, padding=PADDING),)
        self.segdown3_seq = nn.Sequential(nn.Conv2d(128, 4, kernel_size=KERNEL, padding=PADDING), )



        self.fc1 = nn.Sequential(
                                 nn.InstanceNorm2d(64),
                                 nn.Tanh())
        self.fc2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=KERNEL, padding=PADDING),
                                 nn.InstanceNorm2d(64),
                                 nn.Tanh())


        self.soft = nn.Softmax2d()

        self.upsample2 = nn.Upsample(scale_factor=2,mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=4,mode='bilinear')
        # 分割融合
        self.segfusion = nn.Sequential(nn.Conv2d(4*3, 12, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(12),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(4 * 3, 4, kernel_size=KERNEL, padding=PADDING),)


    def encode(self, x):
        out1 = self.conv_seq1(x)
        out2 = self.conv_seq2(self.maxpool(out1))
        out3 = self.conv_seq3(self.maxpool(out2))
        out4 = self.conv_seq4(self.maxpool(out3))
        out5 = self.conv_seq5(self.maxpool(out4))

        deout1 = self.deconv_seq1(torch.cat((self.convt1(out5), out4), 1))
        deout2 = self.deconv_seq2(torch.cat((self.convt2(deout1), out3), 1))

        deout3 = self.deconv_seq3(torch.cat((self.convt3(deout2), out2), 1))

        deout4 = self.deconv_seq4(torch.cat((self.convt4(deout3), out1), 1))
        deout5 = self.deconv_seq5(deout4)

        return deout2,deout3,deout5


    def forward(self, x):
        deout2,deout3,deout5 = self.encode(x)
        pred = self.soft(deout5)
        pred2 = self.segdown2_seq(deout2)
        pred3 = self.segdown3_seq(deout3)
        fusion_seg = self.segfusion(torch.cat((pred,self.upsample4(pred2),self.upsample2(pred3)),dim=1))

        return fusion_seg

######################################################################
# 基本卷积块
class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(

            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # 防止过拟合
            nn.Dropout(0.3),
            nn.LeakyReLU(),

            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # 防止过拟合
            nn.Dropout(0.4),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


# 下采样模块
class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            # 使用卷积进行2倍的下采样，通道数不变
            nn.Conv2d(C, C, 3, 2, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)


# 上采样模块
class UpSampling(nn.Module):

    def __init__(self, C):
        super(UpSampling, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        # 使用邻近插值进行下采样
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        # 拼接，当前上采样的，和之前下采样过程中的
        return torch.cat((x, r), 1)


# 主干网络
class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        # 4次下采样
        self.C1 = Conv(1, 64)
        self.D1 = DownSampling(64)
        self.C2 = Conv(64, 128)
        self.D2 = DownSampling(128)
        self.C3 = Conv(128, 256)
        self.D3 = DownSampling(256)
        self.C4 = Conv(256, 512)
        self.D4 = DownSampling(512)
        self.C5 = Conv(512, 1024)

        # 4次上采样
        self.U1 = UpSampling(1024)
        self.C6 = Conv(1024, 512)
        self.U2 = UpSampling(512)
        self.C7 = Conv(512, 256)
        self.U3 = UpSampling(256)
        self.C8 = Conv(256, 128)
        self.U4 = UpSampling(128)
        self.C9 = Conv(128, 64)

        self.Th = torch.nn.Sigmoid()
        self.pred = torch.nn.Conv2d(64, 4, 3, 1, 1)

    def forward(self, x):
        # 下采样部分
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))

        # 上采样部分
        # 上采样的时候需要拼接起来
        O1 = self.C6(self.U1(Y1, R4))
        O2 = self.C7(self.U2(O1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))

        # 输出预测，这里大小跟输入是一致的
        # 可以把下采样时的中间抠出来再进行拼接，这样修改后输出就会更小
        return self.Th(self.pred(O4))
