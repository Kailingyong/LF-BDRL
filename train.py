import time
import argparse
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import random
from utils.utility import *
from model.DAnet_DAPM import Net
from model.DLnet_DAPM import DLnet
from einops import rearrange
from utils.dataloader_newdata import *
from utils.traindata_LISR_blind import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"




parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--parallel', type=bool, default=False)
parser.add_argument('--num_workers', type=int, default=4)  
parser.add_argument('--model_name_1', type=str, default='LF-DLnet_DAPM')
parser.add_argument('--model_name_2', type=str, default='LF-DAnet_DAPM')
parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
parser.add_argument("--upfactor", type=int, default=4, help="upscale factor") 
parser.add_argument('--load_pretrain', type=bool, default=False)
parser.add_argument('--model1_path', type=str, default='./log_1/LF-DLnet_4xSR.tar')
parser.add_argument('--model2_path', type=str, default='./log_1/LF-DAnet_4xSR.tar')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--patchsize_train', type=int, default=32, help='patchsize of LR images for training')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--n_epochs', type=int, default=70, help='number of epochs to train')
parser.add_argument('--epochs_encoder', type=int, default=20,
                    help='number of epochs to train the degradation encoder')
parser.add_argument('--n_steps', type=int, default=15, help='number of epochs to update learning rate')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')
parser.add_argument('--crop', type=bool, default=True)
parser.add_argument("--patchsize_test", type=int, default=32, help="patchsize of LR images for inference")
parser.add_argument("--minibatch_test", type=int, default=10, help="size of minibatch for inference")
parser.add_argument('--trainset_dir', type=str, default='/mnt/sda/ykl/LFSR_codes/LISR_blind/Train_4x_bicubic/')
parser.add_argument('--testset_dir', type=str, default='/mnt/sda/ykl/ykl/Data-DM/Test_MDSR/')


parser.add_argument('--task', type=str, default='SR', help='SR, RE')


# Optimization specifications
parser.add_argument('--lr_encoder', type=float, default=1e-3,
                    help='learning rate to train the degradation encoder')
parser.add_argument('--lr_sr', type=float, default=1e-4,
                    help='learning rate to train the whole network')
parser.add_argument('--lr_decay_encoder', type=int, default=5,
                    help='learning rate decay per N epochs')
parser.add_argument('--lr_decay_sr', type=int, default=15,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma_encoder', type=float, default=0.1,
                    help='learning rate decay factor for step decay')
parser.add_argument('--gamma_sr', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='resume from the snapshot, and the start_epoch')



args = parser.parse_args()


def train(args):
    net = Net(factor=args.upfactor, angRes=args.angRes)
    net.to(args.device)

    dlnet = DLnet(args)
    dlnet.to(args.device)

    cudnn.benchmark = True  
    epoch_state = 0

    if args.load_pretrain:
        model1 = torch.load(args.model1_path, map_location={'gpu': args.device})
        dlnet.load_state_dict(model1['state_dict'], strict=False)

        epoch_state = model1["epoch"]


    if args.parallel:
        net = torch.nn.DataParallel(net, device_ids=[0, 1]) 
        dlnet = torch.nn.DataParallel(dlnet, device_ids=[0, 1]) 

    criterion_Loss = torch.nn.L1Loss().to(args.device)


    optimizer_dl = torch.optim.Adam([paras for paras in dlnet.parameters() if paras.requires_grad == True], lr=args.lr_encoder)
    optimizer_sr = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=args.lr_sr)


    scheduler_dl = torch.optim.lr_scheduler.StepLR(optimizer_dl, step_size=args.lr_decay_encoder, gamma=args.gamma_encoder)
    scheduler_sr = torch.optim.lr_scheduler.StepLR(optimizer_sr, step_size=args.lr_decay_sr, gamma=args.gamma_sr)

    scheduler_dl._step_count = epoch_state
    scheduler_sr._step_count = epoch_state


    loss_epoch = []

    for idx_epoch in range(epoch_state, args.n_epochs):
        torch.cuda.empty_cache()



        train_set = TrainSetLoader(args)
        train_loader = DataLoader(dataset=train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)

        losses_contrast = AverageMeter()
        losses_sr = AverageMeter()
        contrast_loss = torch.nn.CrossEntropyLoss().to(args.device)
        dlnet.train()
        net.train()



        for idx_iter, (data,label) in tqdm(enumerate(train_loader), total=len(train_loader), ncols=70):


            label, data = augmentation(label, data)

            bdr = 12 // args.upfactor
            label1, data1 = label[:, :, :, :, 12:-12, 12:-12], data[:, :, :, :, bdr:-bdr, bdr:-bdr]

            order = [0, 1, 2, 3]
            random.shuffle(order)
            th, tw = label.shape[-2:]
            th1, tw1 = data.shape[-2:]

            if order[0] == 0:  
                label2, data2 = label[:, :, :, :, 0:args.patchsize_train * args.upfactor,
                                0:args.patchsize_train * args.upfactor], data[:, :, :, :, 0:args.patchsize_train,
                                                                         0:args.patchsize_train]

            elif order[0] == 1:  
                label2, data2 = label[:, :, :, :, th - (args.patchsize_train * args.upfactor):,
                                0:args.patchsize_train * args.upfactor], data[:, :, :, :, th1 - args.patchsize_train:,
                                                                         0:args.patchsize_train]

            elif order[0] == 2:  
                label2, data2 = label[:, :, :, :, 0:args.patchsize_train * args.upfactor,
                                tw - (args.patchsize_train * args.upfactor):], data[:, :, :, :, 0:args.patchsize_train,
                                                                               tw1 - args.patchsize_train:]

            elif order[0] == 3:  
                label2, data2 = label[:, :, :, :, th - (args.patchsize_train * args.upfactor):,
                                tw - (args.patchsize_train * args.upfactor):], data[:, :, :, :,
                                                                               th1 - args.patchsize_train:,
                                                                               tw1 - args.patchsize_train:]


            B, U, V, C, H, W = data1.size()


            data3 = torch.zeros_like(data1)
            for u in range(U):
                for v in range(V):
                    if  u == 4 and v ==4:
                        data3[:, u, v, :, :, :] = data2[:, 0, 0, :, :, :]
                    elif u == 4:
                        data3[:, u, v, :, :, :] = data2[:, 0, v+1, :, :, :]
                    elif v == 4:
                        data3[:, u, v, :, :, :] = data2[:, u+1, 0, :, :, :]
                    else:
                        data3[:, u, v, :, :, :] = data2[:, u+1, v+1, :, :, :]

            optimizer_dl.zero_grad()
            optimizer_sr.zero_grad()


            if idx_epoch+1 <= args.epochs_encoder:

                fea, output, target = dlnet(data1.to(args.device), data3.to(args.device))

                loss_constrast = contrast_loss(output.to(args.device), target.to(args.device))
                loss = loss_constrast
                losses_contrast.update(loss_constrast.item())
                loss.backward()  
                optimizer_dl.step() 

            else:

                fea, output, target = dlnet(data1.to(args.device), data3.to(args.device))


                out_sr = net((data1.to(args.device), fea.to(args.device)))

                loss_SR = criterion_Loss(out_sr.to(args.device), label1.to(args.device))  
                loss_constrast = contrast_loss(output.to(args.device), target.to(args.device))
                loss = loss_constrast + loss_SR

                losses_sr.update(loss_SR.item())
                losses_contrast.update(loss_constrast.item())
                loss.backward()  
                optimizer_dl.step() 
                optimizer_sr.step()



        if idx_epoch+1 <= args.epochs_encoder:
            scheduler_dl.step()
        else:
            scheduler_dl.step()
            scheduler_sr.step()



        if idx_epoch+1 <= args.epochs_encoder:
            print(time.ctime()[4:-5] + ' Epoch----%5d, loss_cl---%f' % (idx_epoch + 1, losses_contrast.avg))
            save_ckpt_1(args, dlnet, idx_epoch + 1)

        else:
            print(time.ctime()[4:-5] + ' Epoch----%5d, loss_cl---%f, loss_sr---%f' % (idx_epoch + 1, losses_contrast.avg, losses_sr.avg))
            save_ckpt_1(args, dlnet, idx_epoch + 1)
            save_ckpt_2(args, net, idx_epoch + 1)



def valid(test_loader, net, dlnet):
    dlnet.eval()
    net.eval()
    psnr_iter_test_tensor = torch.zeros(len(test_loader), device=args.device)  
    ssim_iter_test_tensor = torch.zeros(len(test_loader), device=args.device) 

    for idx_iter, (data, label, sigma, noise_level) in (enumerate(test_loader)):


        if args.crop == False:
            with torch.no_grad():
                outLF = net(data)
                outLF = outLF.squeeze()
        else:
            patch_size = args.patchsize_test
            data = data.squeeze()
            sub_lfs = LFdivide(data, patch_size, patch_size // 2)

            n1, n2, u, v, c, h, w = sub_lfs.shape
            sub_lfs =  rearrange(sub_lfs, 'n1 n2 u v c h w -> (n1 n2) u v c h w')
            mini_batch = args.minibatch_test
            num_inference = (n1 * n2) // mini_batch
            with torch.no_grad():
                out_lfs = []
                for idx_inference in range(num_inference):
                    input_lfs = sub_lfs[idx_inference * mini_batch : (idx_inference+1) * mini_batch, :, :, :, :, :]
                    fea = dlnet(input_lfs.to(args.device),input_lfs.to(args.device))
                    lf_out = net((input_lfs.to(args.device), fea.to(args.device)))
                    if idx_inference == 0:
                        out_lfs = lf_out
                    else:
                        out_lfs = torch.cat((out_lfs, lf_out), dim=0)  
                if (n1 * n2) % mini_batch:
                    input_lfs = sub_lfs[(idx_inference+1) * mini_batch :, :, :, :, :, :]
                    fea = dlnet(input_lfs.to(args.device),input_lfs.to(args.device))
                    lf_out = net((input_lfs.to(args.device), fea.to(args.device)))
                    out_lfs = torch.cat((out_lfs, lf_out), dim=0)  


            out_lfs = rearrange(out_lfs, '(n1 n2) u v c h w -> n1 n2 u v c h w', n1=n1, n2=n2)
            outLF = LFintegrate(out_lfs, patch_size * args.upfactor, patch_size * args.upfactor // 2)
            outLF = outLF[:, :, :, 0 : data.shape[3] * args.upfactor, 0 : data.shape[4] * args.upfactor]

        psnr, ssim = cal_metrics(label[0, ...].squeeze(), outLF)
        psnr_iter_test_tensor[idx_iter] = psnr  
        ssim_iter_test_tensor[idx_iter] = ssim  

    psnr_epoch_test = float(psnr_iter_test_tensor.mean())
    ssim_epoch_test = float(ssim_iter_test_tensor.mean())

    return psnr_epoch_test, ssim_epoch_test


def augmentation(x, y):
    if random.random() < 0.5:  # flip along U-H direction
        x = torch.flip(x, dims=[1, 4])
        y = torch.flip(y, dims=[1, 4])
    if random.random() < 0.5:  # flip along W-V direction
        x = torch.flip(x, dims=[2, 5])
        y = torch.flip(y, dims=[2, 5])
    if random.random() < 0.5: # transpose between U-V and H-W
        x = x.permute(0, 2, 1, 3, 5, 4)
        y = y.permute(0, 2, 1, 3, 5, 4)

    "random color shuffling"
    if random.random() < 0.5:
        color = [0, 1, 2]
        random.shuffle(color)
        x, y = x[:, :, :, color, :, :], y[:, :, :, color, :, :]

    return x, y

def augmentation1(x):
    if random.random() < 0.5:  # flip along U-H direction
        x = torch.flip(x, dims=[1, 4])
    if random.random() < 0.5:  # flip along W-V direction
        x = torch.flip(x, dims=[2, 5])
    if random.random() < 0.5: # transpose between U-V and H-W
        x = x.permute(0, 2, 1, 3, 5, 4)

    "random color shuffling"
    if random.random() < 0.5:
        color = [0, 1, 2]
        random.shuffle(color)
        x = x[:, :, :, color, :, :]

    return x


def get_patch_label(img, patch_size=48, scale=1):
    th, tw = img.shape[-2:]  ## HR image

    tp = round(scale * patch_size)

    tx = random.randrange(0, (tw-tp))
    ty = random.randrange(0, (th-tp))

    return img[:, :, :, :, ty:ty + tp, tx:tx + tp]

def get_patch_data(img, patch_size=48, scale=1):
    th, tw = img.shape[-2:]  ## HR image

    tp = round(patch_size)

    tx = random.randrange(0, (tw-tp))
    ty = random.randrange(0, (th-tp))

    return img[:, :, :, :, ty:ty + tp, tx:tx + tp]


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')    
    train(args)
