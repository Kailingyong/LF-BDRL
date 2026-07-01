import time
import argparse
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import random
from utils.utility import *
from utils.dataloader_newdata import *
from model.DAnet_DAPM import Net
from model.DLnet_DAPM import DLnet
import torch
from torchvision import transforms
import imageio
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--parallel', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=4)  
    parser.add_argument('--model_name_1', type=str, default='LF-DLnet')
    parser.add_argument('--model_name_2', type=str, default='LF-DAnet')
    parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
    parser.add_argument("--upfactor", type=int, default=4, help="upscale factor") 
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='./log/DAnet_4xSR.tar')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--patchsize_train', type=int, default=32, help='patchsize of LR images for training')
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--n_epochs', type=int, default=1200, help='number of epochs to train')
    parser.add_argument('--epochs_encoder', type=int, default=200,
                        help='number of epochs to train the degradation encoder')
    parser.add_argument('--n_steps', type=int, default=300, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')
    parser.add_argument('--crop', type=bool, default=True)
    parser.add_argument("--patchsize_test", type=int, default=32, help="patchsize of LR images for inference")
    parser.add_argument("--minibatch_test", type=int, default=10, help="size of minibatch for inference")
    parser.add_argument('--trainset_dir', type=str, default='/data/infrared/ykl/LF-DMnet/Data/Train_MDSR/')
    parser.add_argument('--testset_dir', type=str, default='/mnt/sda/ykl/LFSR_codes/LISR_blind/paper1_dataset/all/Test_4x_5/')



    return parser.parse_args()


def test_on_datasets(args):

    net = Net(factor=args.upfactor, angRes=args.angRes)
    net.to(args.device)

    dlnet = DLnet(args)
    dlnet.to(args.device)

    cudnn.benchmark = True
    model_path_1 = '/mnt/sda/ykl/LFSR_codes/LF-DMnet-main-ours/log_DAPM/1/LF-DLnet_DAPM_4xSR.tar'
    if os.path.isfile(model_path_1):
        model = torch.load(model_path_1, map_location={'cuda:0': args.device})
        dlnet.load_state_dict(model['state_dict'])
    else:
        print("=> no model found at '{}'".format(args.model_name_1))

    model_path_2 = '/mnt/sda/ykl/LFSR_codes/LF-DMnet-main-ours/log_DAPM/1/LF-DAnet_DAPM_4xSR.tar'
    if os.path.isfile(model_path_2):
        model = torch.load(model_path_2, map_location={'cuda:0': args.device})
        net.load_state_dict(model['state_dict'])
    else:
        print("=> no model found at '{}'".format(args.model_name_2))



    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
    psnr_name = []
    ssim_name = []


    for index, test_name in enumerate(test_Names):
        torch.cuda.empty_cache()

        test_loader = test_Loaders[index]
        psnr_iter_test_tensor, ssim_iter_test_tensor = valid(test_loader, net, dlnet)


        psnr_epoch_test = float(np.array(psnr_iter_test_tensor).mean())
        ssim_epoch_test = float(np.array(ssim_iter_test_tensor).mean())

        print('Dataset--%15s,\t PSNR--%.2f, \t SSIM---%.3f' % (
            test_name, psnr_epoch_test, ssim_epoch_test))
        psnr_name.append(psnr_epoch_test)
        ssim_name.append(ssim_epoch_test)




    psnr = float(np.array(psnr_name).mean())
    ssim = float(np.array(ssim_name).mean())
    print('PSNR--%.2f, \t SSIM---%.3f' % (psnr, ssim))

def valid(test_loader, net, dlnet):
    dlnet.eval()
    net.eval()

    psnr_iter_test_tensor = []
    ssim_iter_test_tensor = []

    for idx_iter, (data, label) in tqdm(enumerate(test_loader), total=len(test_loader), ncols=70):


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
                    torch.cuda.empty_cache()
                    input_lfs = sub_lfs[idx_inference * mini_batch : (idx_inference+1) * mini_batch, :, :, :, :, :]
                    fea = dlnet(input_lfs.to(args.device), input_lfs.to(args.device))
                    lf_out = net((input_lfs.to(args.device), fea.to(args.device)))
                    if idx_inference == 0:
                        out_lfs = lf_out
                    else:
                        out_lfs = torch.cat((out_lfs, lf_out), dim=0)  

                if (n1 * n2) % mini_batch:
                    torch.cuda.empty_cache()
                    input_lfs = sub_lfs[(idx_inference+1) * mini_batch :, :, :, :, :, :]
                    fea = dlnet(input_lfs.to(args.device), input_lfs.to(args.device))
                    lf_out = net((input_lfs.to(args.device), fea.to(args.device)))
                    out_lfs = torch.cat((out_lfs, lf_out), dim=0)  



            out_lfs = rearrange(out_lfs, '(n1 n2) u v c h w -> n1 n2 u v c h w', n1=n1, n2=n2)
            outLF = LFintegrate(out_lfs, patch_size * args.upfactor, patch_size * args.upfactor // 2)
            outLF = outLF[:, :, :, 0 : data.shape[3] * args.upfactor, 0 : data.shape[4] * args.upfactor]

        psnr, ssim = cal_metrics(label[0, ...].squeeze(), outLF)


        psnr_iter_test_tensor.append(psnr)  
        ssim_iter_test_tensor.append(ssim)  


    return psnr_iter_test_tensor, ssim_iter_test_tensor


if __name__ == '__main__':
    args = parse_args()
    test_on_datasets(args)
