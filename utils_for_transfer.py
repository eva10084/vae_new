import torch
from torch import nn
from torch.utils.data import Dataset
import os
import SimpleITK as sitk
#import nibabel as nib
import numpy as np
import glob
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.backends import cudnn
from torch import optim
import torchvision
import torchvision.transforms as transforms
import time
import random
from skimage import transform
import torch.nn.init as init
import torch.nn.functional as F


source = 'C0'
target = 'LGE'


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# device = torch.device("cpu")

def init_conv(conv):
    init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()

# CBMA注意力模块
class CBAM_Attention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM_Attention, self).__init__()

        # 通道注意力模块CAM
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)

        # 空间注意力模块SAM
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道注意力模块CAM
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        avg_pool = self.fc2(self.relu(self.fc1(avg_pool)))
        max_pool = self.fc2(self.relu(self.fc1(max_pool)))
        channel_attention = self.sigmoid(avg_pool + max_pool)

        # 空间注意力模块SAM
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_attention = self.sigmoid(self.conv(torch.cat([max_pool, avg_pool], dim=1)))

        # 综合
        x = x * channel_attention * spatial_attention

        return x

# SENet注意力模块
class SE_Attention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


class Spatial_Attention(nn.Module):
    def __init__(self, in_channel):
        super(Spatial_Attention, self).__init__()
        self.activate = nn.Sequential(nn.Conv2d(in_channel, 1,kernel_size = 1),
                                      )

    def forward(self, x):
        actition = self.activate(x)
        out = torch.mul(x, actition)

        return out

class VAE(nn.Module):
    def __init__(self, KERNEL=3,PADDING=1):
        super(VAE, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)   # 池化层
        self.convt1=nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)   # 上采样层，1014->512
        self.convt2=nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)    # 对输入特征图进行2倍上采样操作，将每个像素点扩大为2x2的区域
        self.convt3=nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)    #  并使用反卷积操作来填充像素
        self.convt4=nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.attention = nn.MultiheadAttention(embed_dim=1024, num_heads=8)
        self.se1 = SE_Attention(64)
        self.se2 = SE_Attention(128)
        self.se3 = SE_Attention(256)
        self.se4 = SE_Attention(512)
        self.se5 = SE_Attention(1024)


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

        self.down4fc1 = nn.Sequential(CBAM_Attention(256),   # 对中间层特征图进行空间注意力机制的加强。
                                      nn.InstanceNorm2d(256),
                                      nn.Tanh())
        self.down4fc2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=KERNEL, padding=PADDING),
                                      nn.InstanceNorm2d(256),
                                      nn.Tanh())
        self.segdown4_seq = nn.Sequential(nn.Conv2d(256, 4, kernel_size=KERNEL, padding=PADDING),)

        self.deconv_seq3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(128),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout2d(p=0.5),
                                       nn.Conv2d(128, 128, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(128),
                                       nn.ReLU(inplace=True))

        self.down2fc1 = nn.Sequential(CBAM_Attention(128),
                                      nn.InstanceNorm2d(128),
                                      nn.Tanh())
        self.down2fc2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=KERNEL, padding=PADDING),
                                      nn.InstanceNorm2d(128),
                                      nn.Tanh())
        self.segdown2_seq = nn.Sequential(nn.Conv2d(128, 4, kernel_size=KERNEL, padding=PADDING),)

        self.deconv_seq4 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(64),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout2d(p=0.5),
                                       nn.Conv2d(64, 64, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(64),
                                       nn.ReLU(inplace=True),)

        self.fc1 = nn.Sequential(CBAM_Attention(64),
                                 nn.InstanceNorm2d(64),
                                 nn.Tanh())
        self.fc2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=KERNEL, padding=PADDING),
                                 nn.InstanceNorm2d(64),
                                 nn.Tanh())

        self.deconv_seq5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(64),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(64, 4, kernel_size=KERNEL, padding=PADDING))
        self.soft = nn.Softmax2d()

        self.upsample2 = nn.Upsample(scale_factor=2,mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=4,mode='bilinear')
        # 分割融合
        self.segfusion = nn.Sequential(nn.Conv2d(4*3, 12, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(12),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(4 * 3, 4, kernel_size=KERNEL, padding=PADDING),)

    def reparameterize(self, mu, logvar,gate):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp*gate
        return z

    def bottleneck(self, h, gate):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar, gate)
        return z, mu, logvar

    def bottleneckdown2(self, h, gate):
        mu, logvar = self.down2fc1(h), self.down2fc2(h)
        z = self.reparameterize(mu, logvar, gate)
        return z, mu, logvar

    def bottleneckdown4(self, h, gate):
        mu, logvar = self.down4fc1(h), self.down4fc2(h)
        z = self.reparameterize(mu, logvar, gate)
        return z, mu, logvar

    def encode(self, x,gate):
        out1 = self.conv_seq1(x)
        # out1 = self.se1(out1)
        out2 = self.conv_seq2(self.maxpool(out1))
        # out2 = self.se2(out2)
        out3 = self.conv_seq3(self.maxpool(out2))
        # out3 = self.se3(out3)
        out4 = self.conv_seq4(self.maxpool(out3))
        out4 = self.se4(out4)
        out5 = self.conv_seq5(self.maxpool(out4))
        # out5 = self.se5(out5)

        x = out5.view(out5.size(0), out5.size(1), -1).permute(2, 0, 1)
        attn_output, _ = self.attention(x, x, x)
        attn_out5 = attn_output.view(out5.size(2), out5.size(3), out5.size(0), out5.size(1)).permute(2, 3, 0, 1)

        deout1 = self.deconv_seq1(torch.cat((self.convt1(out5), out4), 1))
        deout1 = self.se4(deout1)
        deout2 = self.deconv_seq2(torch.cat((self.convt2(deout1), out3), 1))
        # deout2 = self.se3(deout2)
        feat_down4, down4_mu, down4_logvar = self.bottleneckdown4(deout2, gate)
        segout_down4 = self.segdown4_seq(feat_down4)
        pred_down4 = self.soft(segout_down4)
        deout3 = self.deconv_seq3(torch.cat((self.convt3(feat_down4), out2), 1))
        # deout3 = self.se2(deout3)
        feat_down2, down2_mu, down2_logvar = self.bottleneckdown2(deout3, gate)
        segout_down2 = self.segdown2_seq(feat_down2)
        pred_down2 = self.soft(segout_down2)
        deout4 = self.deconv_seq4(torch.cat((self.convt4(feat_down2), out1), 1))
        # deout4 = self.se1(deout4)
        z, mu, logvar = self.bottleneck(deout4, gate)
        return z, mu, logvar,pred_down2,segout_down2,feat_down2,down2_mu,down2_logvar,pred_down4,segout_down4,feat_down4,down4_mu,down4_logvar,out5


    def forward(self, x,gate):
        z, mu, logvar,pred_down2, segout_down2, feat_down2, down2_mu, down2_logvar,pred_down4, segout_down4, feat_down4, down4_mu, down4_logvar,out5 = self.encode(x,gate)
        out= self.deconv_seq5(z)
        pred = self.soft(out)
        fusion_seg = self.segfusion(torch.cat((pred,self.upsample2(pred_down2),self.upsample4(pred_down4)),dim=1))

        return fusion_seg,pred,out,z, mu, logvar,pred_down2, segout_down2, feat_down2, down2_mu, down2_logvar,pred_down4, segout_down4, feat_down4, down4_mu, down4_logvar,out5

class InfoNet(nn.Module):
    def __init__(self, KERNEL=3, PADDING=1):
        super(InfoNet, self).__init__()

        self.info_seq=nn.Sequential(nn.Linear(1024*10*10,256),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(256,6))
    def forward(self, z):
        z=self.info_seq(z.view(z.size(0),-1))
        return z

class VAEDecode(nn.Module):
    def __init__(self, KERNEL=3, PADDING=1):
        super(VAEDecode, self).__init__()

        self.decoderB=nn.Sequential(
            nn.Conv2d(64+4, 128, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(128),  # 输出进行归一化
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=KERNEL, padding=PADDING),
            nn.Sigmoid(),
        )

    def forward(self, z,y):
        z=self.decoderB(torch.cat((z,y),dim=1))
        return z

class VAEDecode_down2(nn.Module):
    def __init__(self, KERNEL=3, PADDING=1):
        super(VAEDecode_down2, self).__init__()

        self.decoderB=nn.Sequential(
            nn.Conv2d(128 + 4, 128, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=KERNEL, padding=PADDING),
            nn.Sigmoid(),
        )

    def forward(self, z,y):
        z=self.decoderB(torch.cat((z,y),dim=1))
        return z

class VAEDecode_down4(nn.Module):
    def __init__(self, KERNEL=3, PADDING=1):
        super(VAEDecode_down4, self).__init__()

        self.decoderB=nn.Sequential(
            nn.Conv2d(256 + 4, 128, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=KERNEL, padding=PADDING),
            nn.Sigmoid(),
        )

    def forward(self, z,y):
        z=self.decoderB(torch.cat((z,y),dim=1))
        return z


class Discriminator(nn.Module):
    def __init__(self, KERNEL=3, PADDING=1):
        super(Discriminator, self).__init__()

        self.decoder=nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, stride=2),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, kernel_size=3),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 8, kernel_size=3, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, dilation=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2,dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.linear_seq=nn.Sequential(nn.Linear(288,256),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(256, 64),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(64, 1),
                                      )

    def forward(self, y):
        out= self.decoder(y)
        out = self.linear_seq(out.view(out.size(0),-1))
        out = out.mean()
        out = out.sigmoid()
        return out


class source_TrainSet(Dataset):
    def __init__(self,extra):
        self.imgdir = extra+'/' + source +'/'
        self.imgsname = glob.glob(self.imgdir + '*' + source + '.nii' + '*')

        imgs = np.zeros((1,192,192))
        labs = np.zeros((1,192,192))
        self.info = []
        for img_num in range(len(self.imgsname)):
            itkimg = sitk.ReadImage(self.imgsname[img_num].replace('\\', '/'))
            npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,220*240*1

            imgs = np.concatenate((imgs,npimg),axis=0)

            labname = self.imgsname[img_num].replace('.nii','_manual.nii')   # 获得对应图片的标注名
            itklab = sitk.ReadImage(labname)
            nplab = sitk.GetArrayFromImage(itklab)
            nplab = (nplab == 200) * 1 + (nplab == 500) * 2 + (nplab == 600) * 3

            labs = np.concatenate((labs, nplab), axis=0)

            spacing = itkimg.GetSpacing()[2]
            media_slice = int(npimg.shape[0] / 2)
            for i in range(npimg.shape[0]):
                a, _ = divmod((i - media_slice) * spacing, 20.0)
                info = int(a) + 3
                if info < 0:
                    info = 0
                elif info > 5:
                    info = 5

                self.info.append(info)
        self.imgs = imgs[1:,:,:]
        self.labs = labs[1:,:,:]
        self.imgs.astype(np.float32)
        self.labs.astype(np.float32)

    def __getitem__(self, item):
        imgindex,crop_indice = divmod(item,4)

        npimg = self.imgs[imgindex,:,:]
        nplab = self.labs[imgindex,:,:]

        randx = np.random.randint(-16,16)
        randy = np.random.randint(-16, 16)
        npimg=npimg[96+randx-80:96+randx+80,96+randy-80:96+randy+80]
        nplab=nplab[96+randx-80:96+randx+80,96+randy-80:96+randy+80]

        npimg_down2 = transform.resize(npimg, (80,80 ), order=3,mode='edge', preserve_range=True)
        npimg_down4 = transform.resize(npimg, (40,40 ), order=3,mode='edge', preserve_range=True)

        nplab_down2 = transform.resize(nplab, (80,80 ), order=0,mode='edge', preserve_range=True)
        nplab_down4 = transform.resize(nplab, (40,40), order=0,mode='edge', preserve_range=True)

        return torch.from_numpy(npimg).unsqueeze(0).type(dtype=torch.FloatTensor),torch.from_numpy(npimg_down2).unsqueeze(0).type(dtype=torch.FloatTensor),torch.from_numpy(npimg_down4).unsqueeze(0).type(dtype=torch.FloatTensor),torch.from_numpy(nplab).type(dtype=torch.LongTensor),torch.from_numpy(nplab_down2).type(dtype=torch.LongTensor),torch.from_numpy(nplab_down4).type(dtype=torch.LongTensor),torch.tensor(self.info[imgindex]).type(dtype=torch.LongTensor)

    def __len__(self):
        return self.imgs.shape[0]*4


class target_TrainSet(Dataset):
    def __init__(self,extra):
        self.imgdir = extra+'/' + target + '/'
        self.imgsname = glob.glob(self.imgdir + '*' + target + '.nii' + '*')

        imgs = np.zeros((1,192,192))
        self.info = []
        for img_num in range(len(self.imgsname)):
            itkimg = sitk.ReadImage(self.imgsname[img_num].replace('\\', '/'))
            npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,220*240*1
            npimg = npimg.astype(np.float32)

            imgs = np.concatenate((imgs,npimg),axis=0)
            spacing = itkimg.GetSpacing()[2]
            media_slice = int(npimg.shape[0] / 2)
            for i in range(npimg.shape[0]):
                a, _ = divmod((i - media_slice) * spacing, 20.0)
                info = int(a) + 3
                if info < 0:
                    info = 0
                elif info > 5:
                    info = 5

                self.info.append(info)
        self.imgs = imgs[1:,:,:]
        # print (imgs.shape)

    def __getitem__(self, item):
        imgindex,crop_indice = divmod(item,4)

        npimg = self.imgs[imgindex,:,:]
        randx = np.random.randint(-16,16)
        randy = np.random.randint(-16, 16)

        npimg=npimg[96+randx-80:96+randx+80,96+randy-80:96+randy+80]
        npimg_down2 = transform.resize(npimg, (80,80 ), order=3,mode='edge', preserve_range=True)
        npimg_down4 = transform.resize(npimg, (40,40 ), order=3,mode='edge', preserve_range=True)

        return torch.from_numpy(npimg).unsqueeze(0).type(dtype=torch.FloatTensor),torch.from_numpy(npimg_down2).unsqueeze(0).type(dtype=torch.FloatTensor),torch.from_numpy(npimg_down4).unsqueeze(0).type(dtype=torch.FloatTensor),torch.tensor(self.info[imgindex]).type(dtype=torch.LongTensor)

    def __len__(self):
        return self.imgs.shape[0]*4


def dice_compute(pred, groundtruth):
    dice = []
    for i in range(4):
        dice_i = 2*(np.sum((pred==i)*(groundtruth==i),dtype=np.float32)+0.0001)/(np.sum(pred==i,dtype=np.float32)+np.sum(groundtruth==i,dtype=np.float32)+0.0001)
        dice = dice+[dice_i]

    return np.array(dice, dtype=np.float32)


def IOU_compute(pred, groundtruth):
    iou = []
    for i in range(4):
        iou_i = (np.sum((pred==i)*(groundtruth==i),dtype=np.float32)+0.0001)/(np.sum(pred==i,dtype=np.float32)+np.sum(groundtruth==i,dtype=np.float32)-np.sum((pred==i)*(groundtruth==i),dtype=np.float32)+0.0001)
        iou=iou+[iou_i]

    return np.array(iou,dtype=np.float32)


def Hausdorff_compute(pred, groundtruth, spacing):
    pred = np.squeeze(pred)
    groundtruth = np.squeeze(groundtruth)

    ITKPred = sitk.GetImageFromArray(pred, isVector=False)
    ITKPred.SetSpacing(spacing)
    ITKTrue = sitk.GetImageFromArray(groundtruth, isVector=False)
    ITKTrue.SetSpacing(spacing)

    overlap_results = np.zeros((1, 4, 5))
    surface_distance_results = np.zeros((1, 4, 5))

    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    for i in range(4):
        pred_i = (pred==i).astype(np.float32)
        if np.sum(pred_i)==0:
            overlap_results[0,i,:]=0
            surface_distance_results[0,i,:]=0
        else:
            overlap_measures_filter.Execute(ITKTrue==i, ITKPred==i)
            overlap_results[0,i, 0] = overlap_measures_filter.GetJaccardCoefficient()
            overlap_results[0,i, 1] = overlap_measures_filter.GetDiceCoefficient()
            overlap_results[0,i, 2] = overlap_measures_filter.GetVolumeSimilarity()
            overlap_results[0,i, 3] = overlap_measures_filter.GetFalseNegativeError()
            overlap_results[0,i, 4] = overlap_measures_filter.GetFalsePositiveError()

            hausdorff_distance_filter.Execute(ITKTrue==i, ITKPred==i)

            surface_distance_results[0,i, 0] = hausdorff_distance_filter.GetHausdorffDistance()


            reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(ITKTrue == i, squaredDistance=False, useImageSpacing=True))
            reference_surface = sitk.LabelContour(ITKTrue == i)
            statistics_image_filter = sitk.StatisticsImageFilter()
            statistics_image_filter.Execute(reference_surface)
            num_reference_surface_pixels = int(statistics_image_filter.GetSum())

            segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(ITKPred==i, squaredDistance=False, useImageSpacing=True))
            segmented_surface = sitk.LabelContour(ITKPred==i)
            statistics_image_filter.Execute(segmented_surface)
            num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

            seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
            ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)

            seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
            seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
            seg2ref_distances = seg2ref_distances + list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
            ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
            ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
            ref2seg_distances = ref2seg_distances + list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))

            all_surface_distances = seg2ref_distances + ref2seg_distances

            surface_distance_results[0,i, 1] = np.mean(all_surface_distances)
            surface_distance_results[0,i, 2] = np.median(all_surface_distances)
            surface_distance_results[0,i, 3] = np.std(all_surface_distances)
            surface_distance_results[0,i, 4] = np.max(all_surface_distances)

    return overlap_results,surface_distance_results


def multi_dice_iou_compute(pred,label):
    truemax, truearg = torch.max(pred, 1, keepdim=False)
    truearg = truearg.detach().cpu().numpy()
    nplabs = np.stack((truearg == 0, truearg == 1, truearg == 2, truearg == 3, truearg == 4, truearg == 5), 1)

    dice = dice_compute(nplabs, label.cpu().numpy())
    Iou = IOU_compute(nplabs, label.cpu().numpy())

    return dice, Iou


class BalancedBCELoss(nn.Module):
    def __init__(self, target):
        super(BalancedBCELoss,self).__init__()
        self.eps=1e-6
        weight = torch.tensor([torch.reciprocal(torch.sum(target==0).float()+self.eps),torch.reciprocal(torch.sum(target==1).float()+self.eps),torch.reciprocal(torch.sum(target==2).float()+self.eps),torch.reciprocal(torch.sum(target==3).float()+self.eps)])
        self.criterion = nn.CrossEntropyLoss(weight)

    def forward(self, output, target):
        loss = self.criterion(output, target)

        return loss

# focal_loss，对类别加权
# alpha：一个长度为C的向量，其中每个元素表示该类别在训练数据中所占比例的倒数。
# gamma：是一个可调节的参数，用于调整困难样本对损失的贡献，一般设定为2。
class Focal_Loss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            loss = torch.mean(F_loss)
        elif self.reduction == 'sum':
            loss = torch.sum(F_loss)
        else:
            loss = F_loss

        return loss


class Boundary_Loss(nn.Module):
    def __init__(self, delta=1.0, reduction='mean'):
        super(Boundary_Loss, self).__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, logits, targets):
        logits = logits.float()
        targets = targets.float()

        # 计算梯度
        logits_grad_x = torch.abs(logits[:, :, :, :-1] - logits[:, :, :, 1:])
        logits_grad_y = torch.abs(logits[:, :, :-1, :] - logits[:, :, 1:, :])
        targets_grad_x = torch.abs(targets[:, :, :, :-1] - targets[:, :, :, 1:])
        targets_grad_y = torch.abs(targets[:, :, :-1, :] - targets[:, :, 1:, :])

        # 计算边界损失
        loss_x = torch.exp((1 - logits_grad_x) / self.delta) * (logits_grad_x < targets_grad_x).float()
        loss_y = torch.exp((1 - logits_grad_y) / self.delta) * (logits_grad_y < targets_grad_y).float()

        if self.reduction == 'mean':
            loss = (loss_x.mean() + loss_y.mean()) / 2
        elif self.reduction == 'sum':
            loss = loss_x.sum() + loss_y.sum()
        else:
            loss = (loss_x + loss_y) / 2

        return loss

def new_loss(inputs, targets):
    focal_loss = Focal_Loss()
    boundary_loss = Boundary_Loss()

    loss = 0.9*focal_loss(inputs, targets) + 0.1*boundary_loss(inputs, targets)

    return loss

class Gaussian_Kernel_Function(nn.Module):
    def __init__(self,std):
        super(Gaussian_Kernel_Function, self).__init__()
        self.sigma=std**2

    def forward(self, fa,fb):
        asize = fa.size()
        bsize = fb.size()

        fa1 = fa.view(-1, 1, asize[1])
        fa2 = fa.view(1, -1, asize[1])

        fb1 = fb.view(-1, 1, bsize[1])
        fb2 = fb.view(1, -1, bsize[1])

        aa = fa1-fa2
        vaa = torch.mean(torch.exp(torch.div(-torch.pow(torch.norm(aa,2,dim=2),2),self.sigma)))

        bb = fb1-fb2
        vbb = torch.mean(torch.exp(torch.div(-torch.pow(torch.norm(bb,2,dim=2),2),self.sigma)))

        ab = fa1-fb2
        vab = torch.mean(torch.exp(torch.div(-torch.pow(torch.norm(ab,2,dim=2),2),self.sigma)))

        loss = vaa+vbb-2.0*vab

        return loss


class Gaussian_Distance(nn.Module):
    def __init__(self,kern=1):
        super(Gaussian_Distance, self).__init__()
        self.kern=kern
        self.avgpool = nn.AvgPool2d(kernel_size=kern, stride=kern)

    def forward(self, mu_a,logvar_a,mu_b,logvar_b):
        mu_a = self.avgpool(mu_a)
        mu_b = self.avgpool(mu_b)
        var_a = self.avgpool(torch.exp(logvar_a))/(self.kern*self.kern)
        var_b = self.avgpool(torch.exp(logvar_b))/(self.kern*self.kern)

        mu_a1 = mu_a.view(mu_a.size(0),1,-1)
        mu_a2 = mu_a.view(1,mu_a.size(0),-1)
        var_a1 = var_a.view(var_a.size(0),1,-1)
        var_a2 = var_a.view(1,var_a.size(0),-1)

        mu_b1 = mu_b.view(mu_b.size(0),1,-1)
        mu_b2 = mu_b.view(1,mu_b.size(0),-1)
        var_b1 = var_b.view(var_b.size(0),1,-1)
        var_b2 = var_b.view(1,var_b.size(0),-1)

        vaa = torch.sum(torch.div(torch.exp(torch.mul(torch.div(torch.pow(mu_a1-mu_a2,2),var_a1+var_a2),-0.5)),torch.sqrt(var_a1+var_a2)))
        vab = torch.sum(torch.div(torch.exp(torch.mul(torch.div(torch.pow(mu_a1-mu_b2,2),var_a1+var_b2),-0.5)),torch.sqrt(var_a1+var_b2)))
        vbb = torch.sum(torch.div(torch.exp(torch.mul(torch.div(torch.pow(mu_b1-mu_b2,2),var_b1+var_b2),-0.5)),torch.sqrt(var_b1+var_b2)))

        loss = vaa+vbb-torch.mul(vab,2.0)

        return loss



