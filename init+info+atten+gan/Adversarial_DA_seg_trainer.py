import torch
from torch import nn
from torch.utils.data import Dataset
import os
import math
import SimpleITK as sitk
#import nibabel as nib
import numpy as np
import glob
from torch.nn import DataParallel
import torch.nn.functional as F
from tqdm import tqdm
from torch.backends import cudnn
from torch import optim
from utils_for_transfer import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 2)

'''
完整版：init+gan+position_infomation+self_attention
'''

EPOCH = 30  # 轮数
LR = 1e-3         # 代表Adam优化器的初始学习率
WEIGHT_DECAY =1e-5   # 代表Adam优化器的权重衰减系数
WORKERSNUM = 0

source = 'C0'
target = 'LGE'
prefix = 'experiments'   # 代表实验结果保存的路径
# prefix = 'gdrive/MyDrive/vae/experiments/loss_tSNE'  # Google云盘
dataset_dir = 'Dataset/small_Patch192'  # 小数据集
# dataset_dir = 'Dataset/Patch192'  # 大数据集
ValiDir = dataset_dir +'/' +target+'_Vali/'  # 代表验证集数据所在的路径，mri测试集
SAVE_DIR=prefix+'/save_model'   # 保存参数路径


BatchSize = 4  # 代表每个批次的样本数
KERNEL = 4

alpha = 1e0
beta = 1e-3
gama = 1e-3
infolamda = 1e-1
kldlamda = 1.0   # Kullback-Leibler散度的权重
predlamda=1e3



if torch.cuda.is_available():
    print("GPU")
    device = torch.device("cuda")  # GPU 可用
else:
    print("CPU")
    device = torch.device("cpu")   # 只能使用 CPU

# device = torch.device("cpu")

if not os.path.exists(prefix):
    os.mkdir(prefix)

def ADA_Train(Infonet, discrim, discrim_criter, discrim_optimizer, source_vae_loss_list,source_seg_loss_list,target_vae_loss_list,distance_loss_list, Train_LoaderA,Train_LoaderB,encoder,decoderA,decoderAdown2,decoderAdown4,decoderB,decoderBdown2,decoderBdown4,gate,DistanceNet,lr,epoch,optim, savedir):
    source_label = 0
    target_label = 1
    lr = lr*(0.9**(epoch))
    for param_group in optim.param_groups:
        param_group['lr'] = lr

    A_iter = iter(Train_LoaderA)
    B_iter = iter(Train_LoaderB)

    num_iter = min(len(Train_LoaderA)-1, len(Train_LoaderB)-1)
    for i in tqdm(range(num_iter)):
        ct,ct_down2,ct_down4,label,label_down2,label_down4 ,info_ct= next(A_iter)
        mr,mr_down2,mr_down4,info_mr= next(B_iter)

        ct= ct.to(device)
        ct_down2= ct_down2.to(device)
        ct_down4= ct_down4.to(device)
        info_ct = info_ct.to(device)

        mr= mr.to(device)
        mr_down4= mr_down4.to(device)
        mr_down2= mr_down2.to(device)
        info_mr = info_mr.to(device)

        label= label.to(device)
        label_onehot =torch.FloatTensor(label.size(0), 4,label.size(1),label.size(2)).to(device)
        label_onehot.zero_()
        label_onehot.scatter_(1, label.unsqueeze(dim=1), 1)

        label_down2= label_down2.to(device)
        label_down2_onehot =torch.FloatTensor(label_down2.size(0), 4,label_down2.size(1),label_down2.size(2)).to(device)
        label_down2_onehot.zero_()
        label_down2_onehot.scatter_(1, label_down2.unsqueeze(dim=1), 1)

        label_down4= label_down4.to(device)
        label_down4_onehot =torch.FloatTensor(label_down4.size(0), 4,label_down4.size(1),label_down4.size(2)).to(device)
        label_down4_onehot.zero_()
        label_down4_onehot.scatter_(1, label_down4.unsqueeze(dim=1), 1)

        fusionseg,_, out_ct,feat_ct, mu_ct,logvar_ct, _, outdown2_ct,featdown2_ct, mudown2_ct,logvardown2_ct,_, outdown4_ct,featdown4_ct, mudown4_ct,logvardown4_ct,info_pred_ct= encoder(ct,gate)
        info_pred_ct = Infonet(info_pred_ct)

        info_cri = nn.CrossEntropyLoss().cuda()
        infoloss_ct = info_cri(info_pred_ct,info_ct)

        seg_criterian = BalancedBCELoss(label)
        seg_criterian = seg_criterian.to(device)
        segloss_output = seg_criterian(out_ct, label)
        fusionsegloss_output = seg_criterian(fusionseg, label)

        segdown2_criterian = BalancedBCELoss(label_down2)
        segdown2_criterian = segdown2_criterian.to(device)
        segdown2loss_output = segdown2_criterian(outdown2_ct, label_down2)

        segdown4_criterian = BalancedBCELoss(label_down4)
        segdown4_criterian = segdown4_criterian.to(device)
        segdown4loss_output = segdown4_criterian(outdown4_ct, label_down4)

        recon_ct = decoderA(feat_ct, label_onehot)
        BCE_ct = F.binary_cross_entropy(recon_ct, ct)
        KLD_ct = -0.5 * torch.mean(1 + logvar_ct - mu_ct.pow(2) - logvar_ct.exp())

        recondown2_ct=decoderAdown2(featdown2_ct, label_down2_onehot)
        BCE_down2_ct = F.binary_cross_entropy(recondown2_ct, ct_down2)
        KLD_down2_ct = -0.5 * torch.mean(1 + logvardown2_ct - mudown2_ct.pow(2) - logvardown2_ct.exp())

        recondown4_ct=decoderAdown4(featdown4_ct,label_down4_onehot)
        BCE_down4_ct = F.binary_cross_entropy(recondown4_ct, ct_down4)
        KLD_down4_ct = -0.5 * torch.mean(1 + logvardown4_ct - mudown4_ct.pow(2) - logvardown4_ct.exp())

        _,pred_mr, _,feat_mr, mu_mr,logvar_mr, preddown2_mr, _,featdown2_mr, mudown2_mr,logvardown2_mr,preddown4_mr, _,featdown4_mr, mudown4_mr,logvardown4_mr,info_pred_mr= encoder(mr,gate)
        info_pred_mr = Infonet(info_pred_mr)
        infoloss_mr = info_cri(info_pred_mr,info_mr)


        recon_mr = decoderB(feat_mr, pred_mr)
        BCE_mr = F.binary_cross_entropy(recon_mr, mr)
        KLD_mr = -0.5 * torch.mean(1 + logvar_mr - mu_mr.pow(2) - logvar_mr.exp())

        recondown2_mr = decoderBdown2(featdown2_mr, preddown2_mr)
        BCE_down2_mr = F.binary_cross_entropy(recondown2_mr, mr_down2)
        KLD_down2_mr = -0.5 * torch.mean(1 + logvardown2_mr - mudown2_mr.pow(2) - logvardown2_mr.exp())

        recondown4_mr = decoderBdown4(featdown4_mr,preddown4_mr)
        BCE_down4_mr = F.binary_cross_entropy(recondown4_mr, mr_down4)
        KLD_down4_mr = -0.5 * torch.mean(1 + logvardown4_mr - mudown4_mr.pow(2) - logvardown4_mr.exp())

        discrim_optimizer.zero_grad()
        source_labels = torch.full((1,), source_label, device=device)
        target_labels = torch.full((1,), target_label, device=device)

        source_outputs = discrim(out_ct).view(-1)
        target_outputs = discrim(pred_mr).view(-1)
        D_loss = discrim_criter(source_outputs.float(), source_labels.float()) + discrim_criter(target_outputs.float(), target_labels.float())
        D_loss.backward(retain_graph=True)
        discrim_optimizer.step()

        G_loss = discrim_criter(source_outputs.float(), target_labels.float()) + discrim_criter(target_outputs.float(), source_labels.float())

        distance_loss = DistanceNet(mu_ct,logvar_ct,mu_mr,logvar_mr)  # 全分辨率
        distance_down2_loss = DistanceNet(mudown2_ct,logvardown2_ct,mudown2_mr,logvardown2_mr) # 下采样2倍
        distance_down4_loss = DistanceNet(mudown4_ct,logvardown4_ct,mudown4_mr,logvardown4_mr) # 下采样4倍

        source_loss = 10.0*BCE_ct+kldlamda*KLD_ct+predlamda*(segloss_output+fusionsegloss_output)+10.0*BCE_down2_ct + kldlamda*KLD_down2_ct +predlamda * segdown2loss_output+10.0*BCE_down4_ct + kldlamda*KLD_down4_ct +predlamda * segdown4loss_output
        target_loss = 10.0*BCE_mr+kldlamda*KLD_mr+10.0*BCE_down2_mr + kldlamda*KLD_down2_mr +10.0*BCE_down4_mr + kldlamda*KLD_down4_mr
        discrepancy_loss = distance_loss+distance_down2_loss + 1e-1*distance_down4_loss

        source_vae_loss_list.append(1*(BCE_ct+BCE_down2_ct+BCE_down4_ct+KLD_ct+KLD_down2_ct+KLD_down4_ct).item())
        source_seg_loss_list.append(1 * (segloss_output+fusionsegloss_output+segdown2loss_output+segdown4loss_output).item())
        target_vae_loss_list.append(1 * (BCE_mr+BCE_down2_mr+BCE_down4_mr+KLD_mr+KLD_down2_mr+KLD_down4_mr).item())
        distance_loss_list.append(1e-5 * discrepancy_loss.item())


        balanced_loss = source_loss + alpha*target_loss + beta*discrepancy_loss + gama*G_loss + infolamda*(infoloss_mr+infoloss_ct)

        optim.zero_grad()
        balanced_loss.backward()
        optim.step()

        if i == num_iter-1:
            print('epoch %d , %d th iter: seglr %.6f,ADA_totalloss %.3f,segloss %.3f' \
                    % (epoch, i, lr, balanced_loss.item(), BCE_mr.item()))


def SegNet_vali(dir, SegNet, gate,epoch, save_DIR): # gate=0

    labsname = glob.glob(dir + '*manual.nii*')

    total_dice = np.zeros((4,))
    total_Iou = np.zeros((4,))

    total_overlap =np.zeros((1,4, 5))
    total_surface_distance=np.zeros((1,4, 5))

    num = 0
    SegNet.eval()

    for i in range(len(labsname)):
        itklab = sitk.ReadImage(labsname[i].replace('\\', '/'))
        nplab = sitk.GetArrayFromImage(itklab)
        nplab = (nplab == 200) * 1 + (nplab == 500) * 2 + (nplab == 600) * 3

        imgname = labsname[i].replace('_manual.nii', '.nii')
        itkimg = sitk.ReadImage(imgname.replace('\\', '/'))
        npimg = sitk.GetArrayFromImage(itkimg)
        npimg = npimg.astype(np.float32)


        data=torch.from_numpy(np.expand_dims(npimg,axis=1)).type(dtype=torch.FloatTensor).to(device)
        label = torch.from_numpy(nplab).to(device)
        truearg  = np.zeros((data.size(0), data.size(2), data.size(3)))


        for slice in range(data.size(0)):
            output,_,_, _, _, _ ,_,_,_,_,_,_,_,_,_,_,_= SegNet(data[slice:slice+1,:,:,:], gate)

            truemax, truearg0 = torch.max(output, 1, keepdim=False)
            truearg0 = truearg0.detach().cpu().numpy()
            truearg[slice:slice+1,:,:]=truearg0


        dice = dice_compute(truearg,label.cpu().numpy())
        Iou = IOU_compute(truearg,label.cpu().numpy())
        overlap_result, surface_distance_result = Hausdorff_compute(truearg,label.cpu().numpy(),itkimg.GetSpacing())

        total_dice = np.vstack((total_dice, dice))
        total_Iou = np.vstack((total_Iou, Iou))

        total_overlap = np.concatenate((total_overlap,overlap_result),axis=0)
        total_surface_distance = np.concatenate((total_surface_distance,surface_distance_result),axis=0)

        num+=1

    if num==0:
        return
    else:
        meanDice = np.mean(total_dice[1:],axis=0)
        stdDice = np.std(total_dice[1:],axis=0)

        meanIou = np.mean(total_Iou[1:],axis=0)
        stdIou = np.std(total_Iou[1:],axis=0)

        mean_overlap = np.mean(total_overlap[1:], axis=0)
        std_overlap = np.std(total_overlap[1:], axis=0)

        mean_surface_distance = np.mean(total_surface_distance[1:], axis=0)
        std_surface_distance = np.std(total_surface_distance[1:], axis=0)

        criterion = np.mean(meanDice[1:])
        phase='validate'

        print("epoch:", epoch, " ", "meanDice:", meanDice, " ", "meanIOU:", meanIou)

        with open("%s/validate_infomation.txt" % (save_DIR), "a") as f:
            f.writelines(["\n\nepoch:", str(epoch), " ",phase," ", "\n","meanDice:",""\
                             ,str(meanDice.tolist()),"stdDice:","",str(stdDice.tolist()),"","\n","meanIou:","",str(meanIou.tolist()),"stdIou:","",str(stdIou.tolist()), \
                              "", "\n\n","jaccard, dice, volume_similarity, false_negative, false_positive:", "\n","mean:", str(mean_overlap.tolist()),"\n", "std:", "", str(std_overlap.tolist()), \
                              "", "\n\n","hausdorff_distance, mean_surface_distance, median_surface_distance, std_surface_distance, max_surface_distance:", "\n","mean:", str(mean_surface_distance.tolist()), "\n","std:", str(std_surface_distance.tolist())])
    return criterion


def t_SNE_plot(Train_LoaderA,Train_LoaderB,net,save_dir,mode):

    A_iter = iter(Train_LoaderA)
    B_iter = iter(Train_LoaderB)
    print(len(Train_LoaderA))
    print(len(Train_LoaderB))
    net.eval()

    features_A = np.zeros((64,))
    features_B = np.zeros((64,))


    num_iter = min(len(Train_LoaderA) - 1, len(Train_LoaderB) - 1)
    for i in tqdm(range(num_iter)):
        sour, sour_down2, sour_down4, label, label_down2, label_down4, info_sour = next(A_iter)
        tar, tar_down2, tar_down4, info_tar = next(B_iter)

        sour = sour.to(device)
        tar = tar.to(device)

        _, _, _, feat_sour, _, _, _, _, _, _, _, _, _, _, _, _, _ = net(sour, 0.0)
        _, _, _, feat_tar, _, _, _, _, _, _, _, _, _, _, _, _, _  = net(tar, 0.0)

        features_A = np.vstack((features_A, feat_sour.cpu().detach().numpy().mean(axis=(2,3)).reshape(sour.size(0),-1)))
        features_B = np.vstack((features_B, feat_tar.cpu().detach().numpy().mean(axis=(2,3)).reshape(tar.size(0),-1)))


    tsne = TSNE()
    X_embedded = tsne.fit_transform(np.concatenate((features_A[1:], features_B[1:]),axis=0))
    Y = ['source']*features_A[1:].shape[0]+['target']*features_A[1:].shape[0]

    tag = features_A[1:].shape[0]
    colors = ['green' if y == 'source' else 'blue' for y in Y]
    plt.scatter(X_embedded[:tag, 0], X_embedded[:tag, 1], c=colors[:tag])
    plt.scatter(X_embedded[tag:, 0], X_embedded[tag:, 1], c=colors[tag:])
    plt.savefig(os.path.join(save_dir, '{}.png'.format(mode)))
    plt.close()
    # # 将t-SNE结果保存到文件中
    # np.save(os.path.join(save_dir, '{}_X.npy'.format(mode)), X_embedded)
    # np.save(os.path.join(save_dir, '{}_Y.npy'.format(mode)), np.array(Y))


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def show_loss(source_vae_loss_list,source_seg_loss_list,target_vae_loss_list,distance_loss_list,SAVE_DIR):
    print('\n iter num: %d' % (len(source_vae_loss_list)))
    np.save(os.path.join(SAVE_DIR, 'source_bce_loss_list.npy'), source_vae_loss_list)
    np.save(os.path.join(SAVE_DIR, 'source_seg_loss_list.npy'), source_seg_loss_list)
    np.save(os.path.join(SAVE_DIR, 'target_bce_loss_list.npy'), target_vae_loss_list)
    np.save(os.path.join(SAVE_DIR, 'distance_loss_list.npy'), distance_loss_list)

    plt.plot(np.arange(0, source_vae_loss_list.shape[0]),
             source_vae_loss_list, 'r', linestyle="-",
             label=r'$\widetilde{\mathcal{L}}_{S/seg}$')
    plt.plot(np.arange(0, source_seg_loss_list.shape[0]), source_seg_loss_list,
             'b',
             label=r'$\widetilde{\mathcal{L}}_{S:seg}$')
    plt.plot(np.arange(0, target_vae_loss_list.shape[0]),
             target_vae_loss_list, 'g', linestyle="-",
             label=r'$\widetilde{\mathcal{L}}_{T}$')
    plt.plot(np.arange(0, distance_loss_list.shape[0]), distance_loss_list,
             'm',
             label=r'$\widetilde{\mathcal{D}}$')

    plt.legend()
    plt.xlabel('iterations', fontsize=15)
    plt.ylabel(" Losses", fontsize=15)
    plt.grid(axis="y", linestyle='--')
    plt.margins(x=0)
    plt.savefig(os.path.join(SAVE_DIR, 'loss.png'))
    plt.close()

def model_init():
    vaeencoder = VAE()
    vaeencoder = vaeencoder.to(device)

    source_vaedecoder = VAEDecode()
    source_vaedecoder = source_vaedecoder.to(device)

    source_down2_vaedecoder = VAEDecode_down2()
    source_down2_vaedecoder = source_down2_vaedecoder.to(device)

    source_down4_vaedecoder = VAEDecode_down4()
    source_down4_vaedecoder = source_down4_vaedecoder.to(device)

    target_vaedecoder = VAEDecode()
    target_vaedecoder = target_vaedecoder.to(device)

    target_down2_vaedecoder = VAEDecode_down2()
    target_down2_vaedecoder = target_down2_vaedecoder.to(device)

    target_down4_vaedecoder = VAEDecode_down4()
    target_down4_vaedecoder = target_down4_vaedecoder.to(device)

    discrim = Discriminator()  # 判断分割后的图片来自于ct，还是mri
    discrim = discrim.to(device)

    Infonet = InfoNet().to(device)

    return vaeencoder, source_vaedecoder, source_down2_vaedecoder, source_down4_vaedecoder ,target_vaedecoder,target_down2_vaedecoder, target_down4_vaedecoder, discrim, Infonet

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    cudnn.benchmark = True

    vaeencoder,source_vaedecoder, source_down2_vaedecoder, source_down4_vaedecoder ,target_vaedecoder,target_down2_vaedecoder, target_down4_vaedecoder, discrim, Infonet= model_init()

    DistanceNet = Gaussian_Distance(KERNEL)
    DistanceNet = DistanceNet.to(device)

    DA_optim = torch.optim.Adam([{'params': vaeencoder.parameters()},{'params': source_vaedecoder.parameters()},{'params': source_down2_vaedecoder.parameters()},{'params': source_down4_vaedecoder.parameters()},{'params': target_vaedecoder.parameters()},{'params': target_down2_vaedecoder.parameters()},{'params': target_down4_vaedecoder.parameters()}],lr=LR,weight_decay=WEIGHT_DECAY)

    # 定义 C0 数据集并将其加载到 DataLoader 中，用于源域数据的训练，调用工具库
    SourceData = source_TrainSet(dataset_dir)
    SourceData_loader = DataLoader(SourceData, batch_size=BatchSize, shuffle=True, num_workers=WORKERSNUM,pin_memory=True,drop_last = True)

    # 定义 LGE 数据集并将其加载到 DataLoader 中，用于目标域数据的训练，，调用工具库
    TargetData = target_TrainSet(dataset_dir)
    TargetData_loader = DataLoader(TargetData, batch_size=BatchSize, shuffle=True, num_workers=WORKERSNUM,pin_memory=True,drop_last = True)


    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    vaeencoder.apply(init_weights)
    source_vaedecoder.apply(init_weights)
    source_down2_vaedecoder.apply(init_weights)
    source_down4_vaedecoder.apply(init_weights)
    target_vaedecoder.apply(init_weights)
    target_down2_vaedecoder.apply(init_weights)
    target_down4_vaedecoder.apply(init_weights)
    discrim.apply(init_weights)
    Infonet.apply(init_weights)


    source_vae_loss_list=[]
    source_seg_loss_list = []
    target_vae_loss_list=[]
    distance_loss_list=[]

    # 训练前的VAE的效果
    print ('start init tsne')
    t_SNE_plot(SourceData_loader, TargetData_loader, vaeencoder, SAVE_DIR, 'init_tsne')
    print ('finish init tsne')

    print('\nstart  training')
    criterion = 0
    best_epoch = 0

    discrim_criter = nn.BCELoss()
    discrim_optimizer = optim.Adam(discrim.parameters(), lr=1e-3)
    for epoch in range(EPOCH):
        # 设置为训练模式
        vaeencoder.train()
        source_vaedecoder.train()
        source_down2_vaedecoder.train()
        source_down4_vaedecoder.train()
        target_vaedecoder.train()
        target_down2_vaedecoder.train()
        target_down4_vaedecoder.train()
        discrim.train()
        Infonet.train()

        # 进行训练，调用程序
        ADA_Train(Infonet, discrim, discrim_criter, discrim_optimizer,
                  source_vae_loss_list, source_seg_loss_list, target_vae_loss_list, distance_loss_list,
                  SourceData_loader, TargetData_loader,
                  vaeencoder,
                  source_vaedecoder, source_down2_vaedecoder, source_down4_vaedecoder,
                  target_vaedecoder, target_down2_vaedecoder, target_down4_vaedecoder,
                  1.0, DistanceNet, LR, epoch, DA_optim, SAVE_DIR)

        vaeencoder.eval()
        criter =SegNet_vali(ValiDir, vaeencoder, 0, epoch, SAVE_DIR)
        print('criter : %.6f' % criter)
        if criter > criterion:
            best_epoch = epoch
            criterion = criter
            # 将当前模型参数保存到文件中
            torch.save(vaeencoder.state_dict(), os.path.join(SAVE_DIR, 'encoder_param.pkl').replace('\\', '/'))
            torch.save(source_vaedecoder.state_dict(), os.path.join(SAVE_DIR, 'decoderA_param.pkl'))
            torch.save(source_down2_vaedecoder.state_dict(), os.path.join(SAVE_DIR, 'decoderAdown2_param.pkl'))
            torch.save(source_down4_vaedecoder.state_dict(), os.path.join(SAVE_DIR, 'decoderAdown4_param.pkl'))
            torch.save(target_vaedecoder.state_dict(), os.path.join(SAVE_DIR, 'decoderB_param.pkl'))
            torch.save(target_down2_vaedecoder.state_dict(), os.path.join(SAVE_DIR, 'decoderBdown2_param.pkl'))
            torch.save(target_down4_vaedecoder.state_dict(), os.path.join(SAVE_DIR, 'decoderBdown4_param.pkl'))
            torch.save(discrim.state_dict(), os.path.join(SAVE_DIR, 'discriminator_param.pkl'))

        if criter >= 0.67:
            torch.save(vaeencoder.state_dict(), SAVE_DIR + '/' + str(criter) + '_encoder_param.pkl')
            torch.save(discrim.state_dict(), SAVE_DIR + '/' + str(criter) + '_discriminator_param.pkl')



    torch.save(vaeencoder.state_dict(), SAVE_DIR + '/' + 'final_' + 'encoder_param.pkl')
    torch.save(discrim.state_dict(), SAVE_DIR + '/' + 'final_' + 'discriminator_param.pkl')
    print('\n')
    print('best epoch:%d' % (best_epoch))
    with open("%s/best_model_information.txt" % (SAVE_DIR), "a") as f:
        f.writelines(["\n\nbest epoch:%d, iter num:%d" % (best_epoch, len(source_vae_loss_list))])

    vaeencoder.load_state_dict(torch.load(SAVE_DIR + '/' + 'encoder_param.pkl', map_location=torch.device("cpu")))
    vaeencoder.to(device)

    # 使用DA模型中的编码器vaeencoder对源域和目标域进行编码，然后进行t-SNE可视化
    print('\nstart res tsne')
    t_SNE_plot(SourceData_loader, TargetData_loader, vaeencoder, SAVE_DIR, 'res_tsne')
    print('final res tsne')

    show_loss(np.array(source_vae_loss_list), np.array(source_seg_loss_list),
              np.array(target_vae_loss_list) ,np.array(distance_loss_list), SAVE_DIR)



if __name__ == '__main__':
    main()
