import torch
import numpy as np
from skimage import metrics
import torch.nn.functional as F
from einops import rearrange
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import xlwt
import math



class ExcelFile():
    def __init__(self):
        self.xlsx_file = xlwt.Workbook()
        self.worksheet = self.xlsx_file.add_sheet(r'sheet1', cell_overwrite_ok=True)
        self.worksheet.write(0, 0, 'Datasets')
        self.worksheet.write(0, 1, 'Scenes')
        self.worksheet.write(0, 2, 'PSNR')
        self.worksheet.write(0, 3, 'SSIM')
        self.worksheet.col(0).width = 256 * 16
        self.worksheet.col(1).width = 256 * 22
        self.worksheet.col(2).width = 256 * 10
        self.worksheet.col(3).width = 256 * 10
        self.sum = 1

    def write_sheet(self, test_name, LF_name, psnr_iter_test, ssim_iter_test):
        ''' Save PSNR & SSIM '''
        for i in range(len(psnr_iter_test)):
            self.add_sheet(test_name, i, psnr_iter_test[i], ssim_iter_test[i])

        psnr_epoch_test = float(np.array(psnr_iter_test).mean())
        ssim_epoch_test = float(np.array(ssim_iter_test).mean())
        self.add_sheet(test_name, 'average', psnr_epoch_test, ssim_epoch_test)
        self.sum = self.sum + 1

    def add_sheet(self, test_name, LF_name, psnr_iter_test, ssim_iter_test):
        ''' Save PSNR & SSIM '''
        self.worksheet.write(self.sum, 0, test_name)
        self.worksheet.write(self.sum, 1, LF_name)
        self.worksheet.write(self.sum, 2, '%.2f' % psnr_iter_test)
        self.worksheet.write(self.sum, 3, '%.3f' % ssim_iter_test)
        self.sum = self.sum + 1

    def add_sheet_times(self, test_name, LF_name, times):
        ''' Save PSNR & SSIM '''
        self.worksheet.write(self.sum, 0, test_name)
        self.worksheet.write(self.sum, 1, LF_name)
        self.worksheet.write(self.sum, 2, '%.3f' % times)
        self.sum = self.sum + 1



def save_ckpt_1(args, net, idx_epoch):
    if args.parallel:
        torch.save({'epoch': idx_epoch, 'state_dict': net.module.state_dict()},
                   './log_1/' + args.model_name_1 + '_' + str(args.upfactor) + 'xSR.tar')
    else:
        torch.save({'epoch': idx_epoch, 'state_dict': net.state_dict()},
                   './log_1/' + args.model_name_1 + '_' + str(args.upfactor) + 'xSR.tar')

    if idx_epoch % 10 == 0:
        if args.parallel:
            torch.save({'epoch': idx_epoch, 'state_dict': net.module.state_dict()},
                       './log_arxiv_1/' + args.model_name_1 + '_' + str(args.upfactor) + 'xSR' + '_epoch_' + str(idx_epoch) + '.tar')
        else:
            torch.save({'epoch': idx_epoch, 'state_dict': net.state_dict()},
                       './log_arxiv_1/' + args.model_name_1 + '_' + str(args.upfactor) + 'xSR' + '_epoch_' + str(idx_epoch) + '.tar')

def save_ckpt_2(args, net, idx_epoch):
    if args.parallel:
        torch.save({'epoch': idx_epoch, 'state_dict': net.module.state_dict()},
                   './log_1/' + args.model_name_2 + '_' + str(args.upfactor) + 'xSR.tar')
    else:
        torch.save({'epoch': idx_epoch, 'state_dict': net.state_dict()},
                   './log_1/' + args.model_name_2 + '_' + str(args.upfactor) + 'xSR.tar')

    if idx_epoch % 10 == 0:
        if args.parallel:
            torch.save({'epoch': idx_epoch, 'state_dict': net.module.state_dict()},
                       './log_arxiv_1/' + args.model_name_2 + '_' + str(args.upfactor) + 'xSR' + '_epoch_' + str(idx_epoch) + '.tar')
        else:
            torch.save({'epoch': idx_epoch, 'state_dict': net.state_dict()},
                       './log_arxiv_1/' + args.model_name_2 + '_' + str(args.upfactor) + 'xSR' + '_epoch_' + str(idx_epoch) + '.tar')








def cal_metrics(label, out):

    U, V, C, H, W = label.size()
    label_y = (65.481 * label[:, :, 0, :, :] + 128.553 * label[:, :, 1, :, :] + 24.966 * label[:, :, 2, :, :] + 16) / 255.0
    out_y = (65.481 * out[:, :, 0, :, :] + 128.553 * out[:, :, 1, :, :] + 24.966 * out[:, :, 2, :, :] + 16) / 255.0

    label_y = label_y.data.cpu().numpy().clip(0, 1)
    out_y = out_y.data.cpu().numpy().clip(0, 1)

    PSNR = np.zeros(shape=(U, V), dtype='float32')
    SSIM = np.zeros(shape=(U, V), dtype='float32')
    for u in range(U):
        for v in range(V):
            PSNR[u, v] = metrics.peak_signal_noise_ratio(label_y[u, v, :, :], out_y[u, v, :, :])
            SSIM[u, v] = metrics.structural_similarity(label_y[u, v, :, :], out_y[u, v, :, :], gaussian_weights=True)


    PSNR_mean = PSNR.sum() / np.sum(PSNR > 0)
    SSIM_mean = SSIM.sum() / np.sum(SSIM > 0)

    return PSNR_mean, SSIM_mean

def cal_metrics_Lytro(label, out):

    U, V, C, H, W = out.size()
    label_y = label
    out_y = (65.481 * out[:, :, 0, :, :] + 128.553 * out[:, :, 1, :, :] + 24.966 * out[:, :, 2, :, :] + 16) / 255.0

    label_y = label_y.data.cpu().numpy().clip(0, 1)
    out_y = out_y.data.cpu().numpy().clip(0, 1)

    PSNR = np.zeros(shape=(U, V), dtype='float32')
    SSIM = np.zeros(shape=(U, V), dtype='float32')
    for u in range(U):
        for v in range(V):
            PSNR[u, v] = metrics.peak_signal_noise_ratio(label_y[u, v, :, :], out_y[u, v, :, :])
            SSIM[u, v] = metrics.structural_similarity(label_y[u, v, :, :], out_y[u, v, :, :], gaussian_weights=True)

    PSNR_mean = PSNR.sum() / np.sum(PSNR > 0)
    SSIM_mean = SSIM.sum() / np.sum(SSIM > 0)

    return PSNR_mean, SSIM_mean


def ImageExtend(Im, bdr):
    [_, _, h, w] = Im.size()
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[:, :, h - bdr[0]: 2 * h + bdr[1], w - bdr[2]: 2 * w + bdr[3]]

    return Im_out

def ImageExtend1(Im, bdr):
    [_, _, h, w] = Im.size()
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[:, :, h - bdr[0]: 2 * h + bdr[1], w - bdr[2]: 2 * w + bdr[3]]

    return Im_out



def LFdivide(lf, patch_size, stride):
    U, V, C, H, W = lf.shape
    data = rearrange(lf, 'u v c h w -> (u v) c h w')

    bdr = (patch_size - stride) // 2
    numU = (H + bdr * 2 - 1) // stride
    numV = (W + bdr * 2 - 1) // stride
    data_pad = ImageExtend(data, [bdr, bdr + stride - 1, bdr, bdr + stride - 1])
    subLF = F.unfold(data_pad, kernel_size=patch_size, stride=stride)
    subLF = rearrange(subLF, '(u v) (c h w) (n1 n2) -> n1 n2 u v c h w',
                      n1=numU, n2=numV, u=U, v=V, h=patch_size, w=patch_size)

    return subLF

def LFdivide1(lf, patch_size, stride):
    U, V, H, W = lf.shape
    data = rearrange(lf, 'u v h w -> (u v) 1 h w')

    bdr = (patch_size - stride) // 2
    numU = (H + bdr * 2 - 1) // stride
    numV = (W + bdr * 2 - 1) // stride
    data_pad = ImageExtend1(data, [bdr, bdr + stride - 1, bdr, bdr + stride - 1])
    subLF = F.unfold(data_pad, kernel_size=patch_size, stride=stride)
    subLF = rearrange(subLF, '(u v) (h w) (n1 n2) -> n1 n2 u v h w',
                      n1=numU, n2=numV, u=U, v=V, h=patch_size, w=patch_size)

    return subLF



def LFintegrate(subLFs, patch_size, stride):
    n1, n2, u, v, c, h, w = subLFs.shape
    bdr = (patch_size - stride) // 2
    outLF = subLFs[:, :, :, :, :, bdr:bdr+stride, bdr:bdr+stride]
    outLF = rearrange(outLF, 'n1 n2 u v c h w -> u v c (n1 h) (n2 w)')

    return outLF


def rgb2ycbcr(x):
    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  65.481 * x[:, :, 0] + 128.553 * x[:, :, 1] +  24.966 * x[:, :, 2] +  16.0
    y[:,:,1] = -37.797 * x[:, :, 0] -  74.203 * x[:, :, 1] + 112.000 * x[:, :, 2] + 128.0
    y[:,:,2] = 112.000 * x[:, :, 0] -  93.786 * x[:, :, 1] -  18.214 * x[:, :, 2] + 128.0
    y = y / 255.0
    return y


def ycbcr2rgb(x):
    mat = np.array(
        [[65.481, 128.553, 24.966],
         [-37.797, -74.203, 112.0],
         [112.0, -93.786, -18.214]])
    mat_inv = np.linalg.inv(mat)
    offset = np.matmul(mat_inv, np.array([16, 128, 128]))
    mat_inv = mat_inv * 255
    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  mat_inv[0,0] * x[:, :, 0] + mat_inv[0,1] * x[:, :, 1] + mat_inv[0,2] * x[:, :, 2] - offset[0]
    y[:,:,1] =  mat_inv[1,0] * x[:, :, 0] + mat_inv[1,1] * x[:, :, 1] + mat_inv[1,2] * x[:, :, 2] - offset[1]
    y[:,:,2] =  mat_inv[2,0] * x[:, :, 0] + mat_inv[2,1] * x[:, :, 1] + mat_inv[2,2] * x[:, :, 2] - offset[2]

    return y

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, my_optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay_sr,
            gamma=args.gamma_sr,
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )

    scheduler.step(args.start_epoch - 1)

    return scheduler

def wave(inp, xfm): ## 小波变换
    if inp.shape[-1]==3:  ## 通常判断的是是否为RGB图像
        tem = int(math.sqrt(inp.shape[-2]))
        inp = inp.permute(0, 2, 1).reshape(-1, 3, tem, tem)  ## 它使用 permute 方法改变张量的维度顺序，将通道维度移到最后，接着使用 reshape 方法将每个图像调整为 tem x tem 的大小。
    # wavelet transform, to encoder
    _, Yh = xfm(inp.view(-1, 3, inp.shape[-2], inp.shape[-1]))
    HL, LH, HH = torch.unbind(Yh[0], dim=2)
    HH = HH.reshape(-1, 3, HH.shape[-2], HH.shape[-1])
    LH = LH.reshape(-1, 3, LH.shape[-2], LH.shape[-1])
    HL = HL.reshape(-1, 3, HL.shape[-2], HL.shape[-1])
    H = torch.cat((HH, LH, HL), dim=1)
    return H
