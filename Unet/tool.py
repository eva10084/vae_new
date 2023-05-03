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

# source = 'C0'
# train = 'C0'
train = 'LGE'
valid = 'LGE_test'


if torch.cuda.is_available():
    device = torch.device("cuda")  # GPU 可用
else:
    device = torch.device("cpu")   # 只能使用 CPU

# device = torch.device("cpu")   # 只能使用 CPU


# 训练集
class source_TrainSet(Dataset):
    def __init__(self, extra):
        self.imgdir = extra+'/' + train +'/'  # 加文件夹名

        # 获取一个路径列表，这些路径是指定目录下所有以 C0.nii 结尾的文件
        self.imgsname = glob.glob(self.imgdir + '*' + train + '.nii' + '*')  # 图片


        imgs = np.zeros((1,192,192))
        labs = np.zeros((1,192,192))
        self.info = []
        for img_num in range(len(self.imgsname)):
            itkimg = sitk.ReadImage(self.imgsname[img_num].replace('\\', '/'))
            npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,220*240*1

            imgs = np.concatenate((imgs,npimg),axis=0)

            labname = self.imgsname[img_num].replace('.nii', '_manual.nii')   # 获得对应图片的标注名
            itklab = sitk.ReadImage(labname)
            nplab = sitk.GetArrayFromImage(itklab)
            nplab = (nplab == 200) * 1 + (nplab == 500) * 2 + (nplab == 600) * 3

            labs = np.concatenate((labs, nplab), axis=0)

        self.imgs = imgs[1:,:,:]
        self.labs = labs[1:,:,:]
        self.imgs.astype(np.float32)
        self.labs.astype(np.float32)


    def __getitem__(self, item):
        imgindex,crop_indice = divmod(item,4)  # 商，余数

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



        return torch.from_numpy(npimg).unsqueeze(0).type(dtype=torch.FloatTensor), torch.from_numpy(nplab).unsqueeze(0).type(dtype=torch.LongTensor)


    def __len__(self):
        return self.imgs.shape[0]*4

# 验证集
class target_TrainSet(Dataset):
    def __init__(self, extra):
        self.imgdir = extra+'/' + valid +'/'  # 加文件夹名

        # 获取一个路径列表，这些路径是指定目录下所有以 C0.nii 结尾的文件
        self.imgsname = glob.glob(self.imgdir + '*' + valid + '.nii' + '*')  # 图片

        imgs = np.zeros((1,192,192))
        labs = np.zeros((1,192,192))
        self.info = []
        for img_num in range(len(self.imgsname)):
            itkimg = sitk.ReadImage(self.imgsname[img_num].replace('\\', '/'))
            npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,220*240*1

            imgs = np.concatenate((imgs,npimg),axis=0)

            labname = self.imgsname[img_num].replace('.nii', '_manual.nii')   # 获得对应图片的标注名
            itklab = sitk.ReadImage(labname)
            nplab = sitk.GetArrayFromImage(itklab)
            nplab = (nplab == 200) * 1 + (nplab == 500) * 2 + (nplab == 600) * 3

            labs = np.concatenate((labs, nplab), axis=0)

            spacing = itkimg.GetSpacing()[2]  # 获取3D图像的z轴spacing（间距） x、y、z
            media_slice = int(npimg.shape[0] / 2)   # npimg.shape[0]，每张3D图片按照z周切片的个数，media为中间层
            for i in range(npimg.shape[0]):  # 这样每张切片图片就会被打上一个属于0-5中某一个类别的标签，便于后续处理和分类。
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
        imgindex,crop_indice = divmod(item,4)  # 商，余数

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


        return torch.from_numpy(npimg).unsqueeze(0).type(dtype=torch.FloatTensor), torch.from_numpy(nplab).unsqueeze(0).type(dtype=torch.LongTensor)


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