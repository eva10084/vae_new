import torch
torch.set_printoptions(profile="full")
from unet import *
from tool import *
from scipy.spatial.distance import directed_hausdorff
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 2)

sns.set(rc={'figure.figsize': (11.7, 8.27)})
palette = sns.color_palette("bright", 2)


TestDir = 'Dataset/small_Patch192/C0/'

# model_dir = 'experiments/loss_tSNE/model/0.70/0.703438.pkl'
# model_dir = 'gdrive/MyDrive/vae/experiments/loss_tSNE/save_param0.001/best_model'  # Google云盘
model_dir = 'res.pkl'

result_save_dir = 'result_image'
name = 'patient1_C0.nii'
slice = 6

if torch.cuda.is_available():
    device = torch.device("cuda")  # GPU 可用
else:
    device = torch.device("cpu")  # 只能使用 CPU

# device = torch.device("cpu")  # 只能使用 CPU


def image_to_index(image):
  image[image == 0] = 0
  image[image == 200] = 1
  image[image == 500] = 2
  image[image == 600] = 3
  return image

def index_to_image(image):
  image[image == 0] = 0
  image[image == 1] = 200
  image[image == 2] = 500
  image[image == 3] = 600
  return image


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

def hausdorff_compute(pred, groundtruth, classes=4):
    hausdorff = []
    for i in range(classes):
        pred_indices = np.argwhere(pred == i)
        gt_indices = np.argwhere(groundtruth == i)

        if len(pred_indices) == 0 or len(gt_indices) == 0:
            hausdorff.append(0)
            continue

        d1 = directed_hausdorff(pred_indices, gt_indices)[0]
        d2 = directed_hausdorff(gt_indices, pred_indices)[0]

        hausdorff_i = max(d1, d2)
        hausdorff.append(hausdorff_i)

    return np.array(hausdorff, dtype=np.float32)

def SegNet(dir, SegNet, gate):
    name = glob.glob(dir + '*C0.nii*')
    SegNet.eval()

    for i in range(len(name)):
        name[i] = name[i].replace('\\', '/')
        itkimg = sitk.ReadImage(name[i])
        npimg = sitk.GetArrayFromImage(itkimg)
        npimg = npimg.astype(np.float32)

        data = torch.from_numpy(np.expand_dims(npimg,axis=1)).type(dtype=torch.FloatTensor).to(device)
        result  = np.zeros((data.size(0), data.size(2), data.size(3)))

        # 对每个切片进行操作
        for slice in range(data.size(0)):
            output = SegNet(data[slice:slice+1,:,:,:])

            truemax, result0 = torch.max(output, 1, keepdim=False)
            result0 = result0.detach().cpu().numpy()
            result[slice:slice+1,:,:]=result0


        sitk_img = sitk.GetImageFromArray(result)
        sitk_ref = sitk.ReadImage(name[i])
        sitk_img.CopyInformation(sitk_ref)

        sitk.WriteImage(sitk_img, (result_save_dir+'/res_'+name[i].split('/')[-1]))


def show(name, slice):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

    # 原图LGE
    name = name.replace('\\', '/')
    itkimg = sitk.ReadImage(name)
    npimg = sitk.GetArrayFromImage(itkimg)
    npimg = npimg.astype(np.float32)
    axs[0].imshow(npimg[slice, :, :], cmap='gray')
    axs[0].set_title('init')

    # 预测LGE
    itres = sitk.ReadImage(result_save_dir+ '/res_'+name.split('/')[-1])
    npres = sitk.GetArrayFromImage(itres)
    npres = npres.astype(np.float32)
    npres = index_to_image(npres)
    axs[1].imshow(npres[slice, :, :], cmap='gray')
    axs[1].set_title('res')

    # 标注
    lab = name.replace('.nii', '_manual.nii')
    itklab = sitk.ReadImage(lab)
    nplab = sitk.GetArrayFromImage(itklab)
    nplab = nplab.astype(np.float32)
    axs[2].imshow(nplab[slice, :, :], cmap='gray')
    axs[2].set_title('real')

    plt.savefig('result.png')
    plt.show()

def calculate_data(name, slice):
    # 预测LGE
    itres = sitk.ReadImage(result_save_dir+'/res_'+name.split('/')[-1])
    npres = sitk.GetArrayFromImage(itres)
    npres = npres.astype(np.float32)
    # [print(i) for i in npres[slice, :, :]]

    # 标注
    lab = name.replace('.nii', '_manual.nii')
    itklab = sitk.ReadImage(lab)
    nplab = sitk.GetArrayFromImage(itklab)
    nplab = nplab.astype(np.float32)
    nplab = image_to_index(nplab)
    # [print(i) for i in nplab[slice, :, :]]

    dice = dice_compute(npres, nplab)
    IOU =  IOU_compute(npres, nplab)
    Hausdorff = hausdorff_compute(npres, nplab)
    print("total:", nplab.shape[0])
    print(" BACK,MYO,LV,RV")
    print("total_dice:", dice)
    print("total_IOU:", IOU)
    print("total_Hausdorff:", Hausdorff)


if __name__ == '__main__':

    model = UNet()
    model.load_state_dict(torch.load(model_dir, map_location=torch.device("cpu")))
    SegNet(TestDir, model.to(device), 0)

    show(TestDir + name, slice)
    # calculate_data(TestDir + name, slice)
