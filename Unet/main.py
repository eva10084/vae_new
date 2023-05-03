import torch
torch.set_printoptions(profile='full')
import torch.nn.functional as F
from tqdm import tqdm
from tool import *
from unet import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize': (11.7, 8.27)})
palette = sns.color_palette("bright", 2)

EPOCH = 1  # 轮数


WORKERSNUM = 0  # 代表用于数据加载的进程数  PS 初始为10，只有0时可以运行
prefix = 'experiments/model'  # 返回上一级目录，代表实验结果保存的路径
# prefix = 'gdrive/MyDrive/vae/experiments/loss_tSNE'  # Google云盘
dataset_dir = 'Dataset/Patch192'  # 返回上一级目录，代表数据集所在的路径
# dataset_dir = 'Dataset/small_Patch192'  # 返回上一级目录，代表数据集所在的路径

source = 'C0'  # 训练集
target = 'LGE'

# ValiDir = dataset_dir + '/' + target + '_test/'  # 代表验证集数据所在的路径

SAVE_DIR = prefix + '/result' # 保存参数路径

image_road = 'experiments/valid_image/'

BatchSize = 5  # 代表每个批次的样本数
KERNEL = 4  # 代表卷积核的大小

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


if torch.cuda.is_available():
    print("GPU")
    device = torch.device("cuda")  # GPU 可用
else:
    print("CPU")
    device = torch.device("cpu")  # 只能使用 CPU

# device = torch.device("cpu")  # 只能使用 CPU

if not os.path.exists(image_road):
    os.makedirs(image_road)


def one_hot(label):
    label_onehot = torch.nn.functional.one_hot(label, num_classes=4)
    label_onehot = torch.squeeze(label_onehot, dim=1).permute(0, 3, 1, 2)
    return label_onehot.to(device)


def from_onehot_to_label(label_onehot):
    # print(label_onehot.shape)
    label = torch.argmax(label_onehot, dim=0).unsqueeze(0)
    return label

def image_to_index(image):
  image[image == 0] = 0
  image[image == 200] = 1
  image[image == 500] = 2
  image[image == 600] = 3
  return image

def index_to_image(image):
  image[image == 0] = 0
  image[image == 1] = 0.2
  image[image == 2] = 0.6
  image[image == 3] = 1
  return image


def show(model,epoch):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
    name = dataset_dir+'/LGE_test/'+'patient44_LGE.nii'
    slice = 12

    # 原图LGE
    itkimg = sitk.ReadImage(name)
    npimg = sitk.GetArrayFromImage(itkimg)
    npimg = npimg.astype(np.float32)
    axs[0].imshow(npimg[slice, :, :], cmap='gray')
    axs[0].set_title('init')

    # 预测LGE
    data = torch.from_numpy(np.expand_dims(npimg, axis=1)).type(dtype=torch.FloatTensor).to(device)
    output = model(data[slice:slice + 1, :, :, :])
    _, result0 = torch.max(output, 1, keepdim=False)
    result0 = result0.detach().cpu().numpy().squeeze(0)
    axs[1].imshow(result0 , cmap='gray')
    axs[1].set_title('result')

   # 标注
    lab = name.replace('.nii', '_manual.nii')
    itklab = sitk.ReadImage(lab)
    nplab = sitk.GetArrayFromImage(itklab)
    nplab = nplab.astype(np.float32)
    axs[2].imshow(nplab[slice, :, :], cmap='gray')
    axs[2].set_title('real')

    road = image_road+str(epoch)+'.png'
    plt.savefig(road)
    # plt.show()

    dice = dice_compute(result0, image_to_index(nplab[slice, :, :]))
    return dice


def main():

    model = UNet()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = optim.RMSprop(model.parameters(), lr=1e-3, weight_decay=1e-8, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # 调整学习率
    # criterion = nn.CrossEntropyLoss()

# 训练集
    SourceData = source_TrainSet(dataset_dir)
    dataloader1 = DataLoader(SourceData, batch_size=BatchSize, num_workers=WORKERSNUM,
                            pin_memory=True, drop_last=True)

# 验证集
#     TargetData = target_TrainSet(dataset_dir)
#     dataloader2 = DataLoader(TargetData, batch_size=BatchSize, num_workers=WORKERSNUM,
#                             pin_memory=True, drop_last=True)

    if not os.path.exists(SAVE_DIR):  # 如果保存训练结果的目录不存在，则创建该目录
        os.mkdir(SAVE_DIR)

    criterion = 0
    best_epoch = 0
    for epoch in range(EPOCH):
        # 设置为训练模式
        model.train()

        for image, label in tqdm(dataloader1):
            image = image.to(device)
            label = label.to(device, dtype=torch.int64)  # 需要int64参与运算
            label_onehot = one_hot(label)

            out = model(image)

            loss = nn.BCELoss()(out.float(), label_onehot.float())
            # loss = dice_coefficient(out.float(), label_onehot.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        dice = show(model, epoch)
        mean_dice = (dice[0]+dice[1]+dice[2]+dice[3])/4
        if mean_dice >= criterion:
            criterion = mean_dice
            best_epoch = epoch
            torch.save(model.state_dict(), SAVE_DIR + '/best' + '.pkl')

        print(f"\nEpoch: {epoch}/{EPOCH}, Loss: {loss}, dice:{dice}")
        torch.save(model.state_dict(), SAVE_DIR+'/res.pkl')

    print(f"\nbest_epoch: {best_epoch}")



if __name__ == '__main__':
    main()