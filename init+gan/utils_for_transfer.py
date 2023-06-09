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

''' 正常'''
'''用于迁移学习的工具'''

# 目前只能是源域C0，目标域LGE

source = 'C0'
target = 'LGE'


if torch.cuda.is_available():
    device = torch.device("cuda")  # GPU 可用
else:
    device = torch.device("cpu")   # 只能使用 CPU

def init_conv(conv):
    init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()

# 定义鉴别器
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(1, 64, 4, 2, 1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 128, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128, 256, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(256, 512, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(512*4*4, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, img):
#         out = self.conv(img)
#         out = out.view(-1, 512*4*4)
#         out = self.fc(out)
#         return out

# 实现空间注意力机制， 将输入特征图压缩成一个单通道的空间特征图
class Spatial_Attention(nn.Module):
    def __init__(self, in_channel):
        super(Spatial_Attention, self).__init__()
        self.activate = nn.Sequential(nn.Conv2d(in_channel, 1,kernel_size = 1),
                                      )

    def forward(self, x):
        actition = self.activate(x)  # 压缩后的单通道特征图 actition
        out = torch.mul(x, actition)  # 压缩后的特征图 actition 和原始的输入特征图 x 进行逐元素乘法操作

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
# 卷积
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

# 反卷积
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
# 下4采样，其中seg割成4类
        self.down4fc1 = nn.Sequential(Spatial_Attention(256),   # 对中间层特征图进行空间注意力机制的加强。
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
# 下2采样，其中seg割成4类
        self.down2fc1 = nn.Sequential(Spatial_Attention(128),
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
# deconv_seq5分成4类
        self.fc1 = nn.Sequential(Spatial_Attention(64),
                                 nn.InstanceNorm2d(64),
                                 nn.Tanh())
        self.fc2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=KERNEL, padding=PADDING),
                                 nn.InstanceNorm2d(64),
                                 nn.Tanh())

        self.deconv_seq5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(64),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(64, 4, kernel_size=KERNEL, padding=PADDING))
        self.soft = nn.Softmax2d()  # 归一化操作，可以将特征图中每个位置的值归一化到 [0,1] 区间内
# 上采样，segfusion割成4类
        self.upsample2 = nn.Upsample(scale_factor=2,mode='bilinear')  # 双线性插值，并将特征图的尺寸沿着宽和高的维度分别扩大了2倍
        self.upsample4 = nn.Upsample(scale_factor=4,mode='bilinear')
        # 分割融合
        self.segfusion = nn.Sequential(nn.Conv2d(4*3, 12, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(12),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(4 * 3, 4, kernel_size=KERNEL, padding=PADDING),)

# 计算VAE的loss：重建损失加KL
    def reparameterize(self, mu, logvar,gate):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp*gate
        return z

# 3中不同的fc，返回loss，均值，方差的log
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
        out2 = self.conv_seq2(self.maxpool(out1))
        out3 = self.conv_seq3(self.maxpool(out2))
        out4 = self.conv_seq4(self.maxpool(out3))
        out5 = self.conv_seq5(self.maxpool(out4))
        # out5, _ = self.attention(out5, out5, out5)

        deout1 = self.deconv_seq1(torch.cat((self.convt1(out5),out4),1))
        deout2 = self.deconv_seq2(torch.cat((self.convt2(deout1),out3),1))
        feat_down4,down4_mu,down4_logvar = self.bottleneckdown4(deout2,gate)
        segout_down4 = self.segdown4_seq(feat_down4)
        pred_down4 = self.soft(segout_down4)
        deout3 = self.deconv_seq3(torch.cat((self.convt3(feat_down4),out2),1))
        feat_down2,down2_mu,down2_logvar = self.bottleneckdown2(deout3,gate)
        segout_down2 = self.segdown2_seq(feat_down2)
        pred_down2 = self.soft(segout_down2)
        deout4 = self.deconv_seq4(torch.cat((self.convt4(feat_down2),out1),1))
        z, mu, logvar = self.bottleneck(deout4,gate)
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

# 判别器
class Discriminator(nn.Module):
    def __init__(self, KERNEL=3, PADDING=1):
        super(Discriminator, self).__init__()

        self.decoder=nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, stride=2),  # 190
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, kernel_size=3),  # (190-3)/2+1=94
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 8, kernel_size=3, stride=2),  # 190
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3),  # (190-3)/2+1=94
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, dilation=2),  # 190
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3),  # (190-3)/2+1=94
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2,dilation=2),  # 190
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3),  # (190-3)/2+1=94
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.linear_seq=nn.Sequential(nn.Linear(288,256),
                                      # nn.Linear(32*5*5,256),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(256, 64),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(64, 1),
                                      )

    def forward(self, y):
        out= self.decoder(y)
        # print(out.shape)
        # print(out.view(out.size(0),-1).shape)
        out = self.linear_seq(out.view(out.size(0),-1))  # 将卷积层或池化层的输出展平成一维张量，以便可以将其传递给全连接层
        out = out.mean()  # 计算张量 out 所有元素的平均值
        out = out.sigmoid()
        return out

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

        # npimg_o = transform.resize(npimg, (80, 80),
        #                      order=3, mode='edge', preserve_range=True)
        #npimg_resize = transform.resize(npimg, (96, 96), order=3,mode='edge', preserve_range=True)
        npimg_down2 = transform.resize(npimg, (80,80 ), order=3,mode='edge', preserve_range=True)
        npimg_down4 = transform.resize(npimg, (40,40 ), order=3,mode='edge', preserve_range=True)

        return torch.from_numpy(npimg).unsqueeze(0).type(dtype=torch.FloatTensor),torch.from_numpy(npimg_down2).unsqueeze(0).type(dtype=torch.FloatTensor),torch.from_numpy(npimg_down4).unsqueeze(0).type(dtype=torch.FloatTensor),torch.tensor(self.info[imgindex]).type(dtype=torch.LongTensor)

    def __len__(self):

        return self.imgs.shape[0]*4


class source_TrainSet(Dataset):
    def __init__(self,extra):
        self.imgdir = extra+'/' + source +'/'  # 加文件夹名

        # 获取一个路径列表，这些路径是指定目录下所有以 C0.nii 结尾的文件
        self.imgsname = glob.glob(self.imgdir + '*' + source + '.nii' + '*')  # 图片

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

        # npimg = transform.resize(npimg, (96, 96), order=3,mode='edge', preserve_range=True)
        # nplab = transform.resize(nplab, (96, 96), order=0,mode='edge', preserve_range=True)
        randx = np.random.randint(-16,16)
        randy = np.random.randint(-16, 16)
        npimg=npimg[96+randx-80:96+randx+80,96+randy-80:96+randy+80]
        nplab=nplab[96+randx-80:96+randx+80,96+randy-80:96+randy+80]

        # npimg_o=transform.resize(npimg, (80,80 ), order=3,mode='edge', preserve_range=True)
        # nplab_o=transform.resize(nplab, (80,80 ), order=0,mode='edge', preserve_range=True)

        npimg_down2 = transform.resize(npimg, (80,80 ), order=3,mode='edge', preserve_range=True)
        npimg_down4 = transform.resize(npimg, (40,40 ), order=3,mode='edge', preserve_range=True)

        nplab_down2 = transform.resize(nplab, (80,80 ), order=0,mode='edge', preserve_range=True)
        nplab_down4 = transform.resize(nplab, (40,40), order=0,mode='edge', preserve_range=True)

        return torch.from_numpy(npimg).unsqueeze(0).type(dtype=torch.FloatTensor),torch.from_numpy(npimg_down2).unsqueeze(0).type(dtype=torch.FloatTensor),torch.from_numpy(npimg_down4).unsqueeze(0).type(dtype=torch.FloatTensor),torch.from_numpy(nplab).type(dtype=torch.LongTensor),torch.from_numpy(nplab_down2).type(dtype=torch.LongTensor),torch.from_numpy(nplab_down4).type(dtype=torch.LongTensor),torch.tensor(self.info[imgindex]).type(dtype=torch.LongTensor)

    def __len__(self):

        return self.imgs.shape[0]*4



# Dice系数越接近1，表示模型的分割效果越好
def dice_compute(pred, groundtruth):           #batchsize*channel*W*W
    dice=[]
    for i in range(4):
        dice_i = 2*(np.sum((pred==i)*(groundtruth==i),dtype=np.float32)+0.0001)/(np.sum(pred==i,dtype=np.float32)+np.sum(groundtruth==i,dtype=np.float32)+0.0001)
        dice = dice+[dice_i]

    return np.array(dice, dtype=np.float32)



# IOU也越接近1，表示模型的分割效果越好，类似于dice
def IOU_compute(pred, groundtruth):
    iou=[]
    for i in range(4):
        iou_i = (np.sum((pred==i)*(groundtruth==i),dtype=np.float32)+0.0001)/(np.sum(pred==i,dtype=np.float32)+np.sum(groundtruth==i,dtype=np.float32)-np.sum((pred==i)*(groundtruth==i),dtype=np.float32)+0.0001)
        iou=iou+[iou_i]


    return np.array(iou,dtype=np.float32)


# 广泛应用于评估模型对于分割边缘和形状的准确性，越小，分割效果越好
# 输入原图地址和目标地址
# 参数spacing是一个元组或列表，用于指定各个维度上的像素间距
# 例如，spacing=(1,1,1)表示三维图像中各个维度上的像素间距均为1，
def Hausdorff_compute(pred, groundtruth, spacing):
    pred = np.squeeze(pred)
    groundtruth = np.squeeze(groundtruth)

    ITKPred = sitk.GetImageFromArray(pred, isVector=False)
    ITKPred.SetSpacing(spacing)  # SetSpacing方法用于设置图像的像素间距
    ITKTrue = sitk.GetImageFromArray(groundtruth, isVector=False)
    ITKTrue.SetSpacing(spacing)

    overlap_results = np.zeros((1,4, 5))
    surface_distance_results = np.zeros((1,4, 5))

    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    for i in range(4):
        pred_i = (pred==i).astype(np.float32)
        if np.sum(pred_i)==0:
            overlap_results[0,i,:]=0
            surface_distance_results[0,i,:]=0
        else:
            # Overlap measures
            overlap_measures_filter.Execute(ITKTrue==i, ITKPred==i)
            overlap_results[0,i, 0] = overlap_measures_filter.GetJaccardCoefficient()
            overlap_results[0,i, 1] = overlap_measures_filter.GetDiceCoefficient()
            overlap_results[0,i, 2] = overlap_measures_filter.GetVolumeSimilarity()
            overlap_results[0,i, 3] = overlap_measures_filter.GetFalseNegativeError()
            overlap_results[0,i, 4] = overlap_measures_filter.GetFalsePositiveError()
            # Hausdorff distance
            hausdorff_distance_filter.Execute(ITKTrue==i, ITKPred==i)

            surface_distance_results[0,i, 0] = hausdorff_distance_filter.GetHausdorffDistance()
            # Symmetric surface distance measures

            reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(ITKTrue == i, squaredDistance=False, useImageSpacing=True))
            reference_surface = sitk.LabelContour(ITKTrue == i)
            statistics_image_filter = sitk.StatisticsImageFilter()
            # Get the number of pixels in the reference surface by counting all pixels that are 1.
            statistics_image_filter.Execute(reference_surface)
            num_reference_surface_pixels = int(statistics_image_filter.GetSum())

            segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(ITKPred==i, squaredDistance=False, useImageSpacing=True))
            segmented_surface = sitk.LabelContour(ITKPred==i)
            # Get the number of pixels in the reference surface by counting all pixels that are 1.
            statistics_image_filter.Execute(segmented_surface)
            num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

            # Multiply the binary surface segmentations with the distance maps. The resulting distance
            # maps contain non-zero values only on the surface (they can also contain zero on the surface)
            seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
            ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)

            # Get all non-zero distances and then add zero distances if required.
            seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
            seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
            seg2ref_distances = seg2ref_distances + \
                                list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
            ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
            ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
            ref2seg_distances = ref2seg_distances + \
                                list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))

            all_surface_distances = seg2ref_distances + ref2seg_distances

            # The maximum of the symmetric surface distances is the Hausdorff distance between the surfaces. In
            # general, it is not equal to the Hausdorff distance between all voxel/pixel points of the two
            # segmentations, though in our case it is. More on this below.
            surface_distance_results[0,i, 1] = np.mean(all_surface_distances)
            surface_distance_results[0,i, 2] = np.median(all_surface_distances)
            surface_distance_results[0,i, 3] = np.std(all_surface_distances)
            surface_distance_results[0,i, 4] = np.max(all_surface_distances)


    return overlap_results,surface_distance_results


def multi_dice_iou_compute(pred,label):
    truemax, truearg = torch.max(pred, 1, keepdim=False)
    truearg = truearg.detach().cpu().numpy()
    # nplabs = np.stack((truearg == 0, truearg == 1, truearg == 2, truearg == 3, \
    #                    truearg == 4, truearg == 5, truearg == 6, truearg == 7), 1)
    nplabs = np.stack((truearg == 0, truearg == 1, truearg == 2, truearg == 3, truearg == 4, truearg == 5), 1)
    # truelabel = (truearg == 0) * 550 + (truearg == 1) * 420 + (truearg == 2) * 600 + (truearg == 3) * 500 + \
    #             (truearg == 4) * 250 + (truearg == 5) * 850 + (truearg == 6) * 820 + (truearg == 7) * 0

    dice = dice_compute(nplabs, label.cpu().numpy())
    Iou = IOU_compute(nplabs, label.cpu().numpy())

    return dice, Iou

# 交叉熵loss
class BalancedBCELoss(nn.Module):
    def __init__(self,target):
        super(BalancedBCELoss,self).__init__()
        self.eps=1e-6
        # 计算一个加权的交叉熵损失函数中的权重
        # torch.sum(target==0)表示标签中像素值为0的数量
        # torch.reciprocal表示取倒数，即求该类别像素所占的比例。
        weight = torch.tensor([torch.reciprocal(torch.sum(target==0).float()+self.eps),torch.reciprocal(torch.sum(target==1).float()+self.eps),torch.reciprocal(torch.sum(target==2).float()+self.eps),torch.reciprocal(torch.sum(target==3).float()+self.eps)])
        self.criterion = nn.CrossEntropyLoss(weight)

    def forward(self, output,target):
        loss = self.criterion(output,target)

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
        # 一个平均池化层，它将输入的特征图按照给定的kernel_size进行划分，对每个划分区域内的数值求平均，然后输出新的特征图。

    def forward(self, mu_a,logvar_a,mu_b,logvar_b):
        mu_a = self.avgpool(mu_a)
        mu_b = self.avgpool(mu_b)
        # var_a = torch.exp(logvar_a)
        # var_b = torch.exp(logvar_b)
        var_a = self.avgpool(torch.exp(logvar_a))/(self.kern*self.kern)
        var_b = self.avgpool(torch.exp(logvar_b))/(self.kern*self.kern)
        # var_a = torch.exp(logvar_a)
        # var_b = torch.exp(logvar_b)


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



