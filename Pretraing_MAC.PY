"""
Pretrain MAC with RML-X dataset
"""
from __future__ import print_function
import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import argparse
import torch.nn as nn
import tensorboard_logger as tb_logger
from util import adjust_learning_rate, AverageMeter, load_RML2016
from models.backbone import MAC_backbone
from NCE.NCEAverage import NCEAverage
from NCE.NCECriterion import NCECriterion
from NCE.NCECriterion import NCESoftmaxLoss
from torch.utils.tensorboard import SummaryWriter

# Import matplotlib for plotting graphs ans seaborn for attractive graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import numpy as np

try:
    from apex import amp, optimizers
except ImportError:
    pass

def parse_option():
    parser = argparse.ArgumentParser('argument for MAC pretraining')
    # MAC
    # Which domain of the signal should be used for comparison
    # WT is wavelet, AN is amplitude phase, AF is instantaneous frequency, FFT is spectrum
    parser.add_argument('--mod_l', type=str, default='AN', choices=['WT', 'AN', 'AF', 'FFT'])

    # VIEW
    parser.add_argument('--view_chose', type=str, default='ALL', choices=['ALL', 'DB'])

    # PRETRAIN
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=120, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--n_1', type=float, default=1, help='weighting coefffcients SD')
    parser.add_argument('--n_t', type=float, default=1, help='weighting coefffcients TD')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160,200', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='decay rate for learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # RESUME
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='MAC_backbone', choices=['MAC_backbone'])
    parser.add_argument('--softmax', action='store_true', help='using softmax contrastive loss rather than NCE')
    parser.add_argument('--nce_k', type=int, default=16384)  #16384 32768 8192 4096
    parser.add_argument('--nce_t', type=float, default=0.07)
    parser.add_argument('--nce_m', type=float, default=0.9)
    parser.add_argument('--feat_dim', type=int, default=128, help='dim of feat for inner product')

    # DATASET
    parser.add_argument('--snr_tat', type=int,
                        default=6, help='Signal to noise ratio for training and testing')
    parser.add_argument('--threads', type=int, default=4,
                        help='number of threads for data loader to use. default=4')
    parser.add_argument("--N_shot", default=50, type=int, help='Sample ratio N for fine-tuning')
    parser.add_argument('--ab_choose', type=str, default="RML201610A",
                        help="RML201610A, RML201610B, RML2018")

    # PATH
    parser.add_argument('--log_path', type=str,
                        default="D:\\PYALL\\RML2016\\RML2016_10B\\log_1.txt", help="log")
    parser.add_argument('--RML2016b_path', type=str, default="D:\\PYALL\\RML2016\\RML2016_10B\\pre_snr_train_test\\",
                        help="RML2016B-MT4 Training and testing dataset path")
    parser.add_argument('--RML2018_path', type=str, default="D:\pyALL\调制识别数据集\RML2018MV\\",
                        help="RML2018A-MT4 Training and testing dataset path")
    parser.add_argument('--RML2016a_path', type=str, default="D:\\pyALL\\调制识别数据集\\RML201610a_new\\py\\",
                        help="RML2016A-MT4 Training and testing dataset path")
    parser.add_argument('--model_path', type=str, default="result\\", help='path to save model')
    parser.add_argument('--tb_path', type=str, default="result\\", help='path to tensorboard')

    # THY
    parser.add_argument("--channel",default=50,type=int)
    parser.add_argument("--latent_size",default=256,type=int)
    parser.add_argument("--tempature_cls",default=10,type=int)
    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.method = 'softmax' if opt.softmax else 'nce'
    opt.model_name = opt.model
    opt.model_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    return opt


def set_model(args, n_data):
    # set the model
    if args.ab_choose == "RML201610A":
        args.num_class = 11
        model = MAC_backbone(args.feat_dim, args.num_class)
    elif args.ab_choose == "RML201610B":
        args.num_class = 10
        model = MAC_backbone(args.feat_dim, args.num_class)
    elif args.ab_choose == "RML2018":
        args.num_class = 24
        model = MAC_backbone(args.feat_dim, args.num_class)
    else:
        print('Dataset selection for non RML series:', args.ab_choose)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # Intra-inter domain dual-cycle criterion
    contrast = NCEAverage(args.feat_dim, n_data, args.nce_k, args.nce_t, args.nce_m, args.softmax)
    criterion_l = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    criterion_ab = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)

    if torch.cuda.is_available():
        model = model.cuda()
        contrast = contrast.cuda()
        criterion_ab = criterion_ab.cuda()
        criterion_l = criterion_l.cuda()
        cudnn.benchmark = True

    return model, contrast, criterion_ab, criterion_l


def set_optimizer(args, model):
    # return optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    return optimizer

# "I-Q single centralization" strategy
def IQ_SC(feat_ab, feat_l, contrast, criterion_l, criterion_ab, index):
    # feature normalization
    feat_ab = nn.functional.normalize(feat_ab, dim=1)
    feat_l = nn.functional.normalize(feat_l, dim=1)

    out_l, out_ab = contrast(feat_l, feat_ab, index)
    l_loss = criterion_l(out_l)
    ab_loss = criterion_ab(out_ab)
    return l_loss, ab_loss

def train(epoch, train_loader, model, contrast, criterion_l, criterion_ab, optimizer, opt):
    """
    MAC one epoch pretraining
    """
    model.train()
    contrast.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    l_loss_meter = AverageMeter()
    ab_loss_meter = AverageMeter()

    T_loss_meter = AverageMeter()
    T1_loss_meter = AverageMeter()
    T2_loss_meter = AverageMeter()
    T3_loss_meter = AverageMeter()

    abT_loss_meter = AverageMeter()
    abT1_loss_meter = AverageMeter()
    abT2_loss_meter = AverageMeter()
    abT3_loss_meter = AverageMeter()

    feat_bnsum = []
    label_bnsum = []
    end = time.time()
    for idx, (inputs, _, index) in enumerate(train_loader):
        # The index number of the sample in the dataset
        data_time.update(time.time() - end)

        bsz = inputs.size(0)
        inputs = inputs.float()
        if torch.cuda.is_available():
            # with torch.no_grad():
            index = index.cuda(non_blocking=True)
            inputs = inputs.cuda()

        # ===================forward=====================
        if opt.view_chose == 'DB':
            feat_l, feat_ab, feat_SD = model(inputs, opt.mod_l, opt.view_chose, 'pretrain')
            l_loss, ab_loss = IQ_SC(feat_ab, feat_l, contrast, criterion_l, criterion_ab, index)
            l_lossSD, ab_lossSD = IQ_SC(feat_SD, feat_l, contrast, criterion_l, criterion_ab, index)

            l_lossT1 = l_loss
            l_lossT2 = l_loss
            l_lossT3 = l_loss
            ab_lossT1 = ab_loss
            ab_lossT2 = ab_loss
            ab_lossT3 = ab_loss

            loss = opt.n_1 * (l_lossSD + ab_lossSD) + opt.n_t * (l_loss + ab_loss)

        elif opt.view_chose == 'ALL':
            feat_l, feat_TD, feat_TD1, feat_TD2, feat_TD3, feat_SD1 = model(inputs, opt.mod_l, opt.view_chose, 'pretrain')
            # Inter-domain contrastive learning
            l_loss, ab_loss = IQ_SC(feat_TD, feat_l, contrast, criterion_l, criterion_ab, index)
            l_lossT1, ab_lossT1 = IQ_SC(feat_TD1, feat_l, contrast, criterion_l, criterion_ab, index)
            l_lossT2, ab_lossT2 = IQ_SC(feat_TD2, feat_l, contrast, criterion_l, criterion_ab, index)
            l_lossT3, ab_lossT3 = IQ_SC(feat_TD3, feat_l, contrast, criterion_l, criterion_ab, index)

            # Intra-domain contrastive learning
            l_lossSD, ab_lossSD = IQ_SC(feat_SD1, feat_l, contrast, criterion_l, criterion_ab, index)
            # feat_l, feat_ab, feat_ab2, feat_ab3 = model(inputs, opt.mod_l, opt.view_chose, 'pretrain')

            loss = opt.n_1 * (l_lossSD + ab_lossSD) + opt.n_t * (l_loss + l_lossT1 + l_lossT2 + l_lossT3) + opt.n_t*(ab_loss + ab_lossT1 + ab_lossT2 + ab_lossT3)
            # loss = l_loss + l_loss2 + l_loss3 + ab_loss + ab_loss2 + ab_loss3
        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        losses.update(loss.item(), bsz)
        l_loss_meter.update(l_lossSD.item(), bsz)
        ab_loss_meter.update(ab_lossSD.item(), bsz)

        T_loss_meter.update(l_loss.item(), bsz)
        T1_loss_meter.update(l_lossT1.item(), bsz)
        T2_loss_meter.update(l_lossT2.item(), bsz)
        T3_loss_meter.update(l_lossT3.item(), bsz)

        abT_loss_meter.update(ab_loss.item(), bsz)
        abT1_loss_meter.update(ab_lossT1.item(), bsz)
        abT2_loss_meter.update(ab_lossT2.item(), bsz)
        abT3_loss_meter.update(ab_lossT3.item(), bsz)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'ANloss {ANloss.val:.3f} ({ANloss.avg:.3f})\t'
                  'WTloss {WTloss.val:.3f} ({WTloss.avg:.3f})\t'
                  'AFloss {AFloss.val:.3f} ({AFloss.avg:.3f})\t'
                  'FFTloss {FFTloss.val:.3f} ({FFTloss.avg:.3f})\t'
                  'l_SD {l_loss_SD.val:.3f} ({l_loss_SD.avg:.3f})\t'
                  'ab_SD {ab_loss_SD.val:.3f} ({ab_loss_SD.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, ANloss = T_loss_meter, WTloss=T1_loss_meter,
                AFloss = T2_loss_meter, FFTloss= T3_loss_meter,l_loss_SD=l_loss_meter,
                ab_loss_SD=ab_loss_meter))

            # print(feat_SD1.shape)
            sys.stdout.flush()

            #特征合并
            # if opt.view_chose == 'DB':
            #     feat = torch.cat((feat_l.detach(), feat_ab.detach()), dim=1).cpu()
            # elif opt.view_chose == 'ALL':
            #     feat = torch.cat((feat_l.detach(), feat_ab.detach(), feat_ab2.detach(), feat_ab3.detach()), dim=1).cpu()
            # label = index.cpu()
            # feat_bnsum.append((feat.detach().numpy()))
            # label_bnsum.append((label.detach().numpy()))

    #特征图可视化
    # X = np.concatenate(feat_bnsum)
    # Y = np.concatenate(label_bnsum)

    def plot(x, colors):
        # Choosing color palette
        # https://seaborn.pydata.org/generated/seaborn.color_palette.html
        if opt.ab_choose == 'RML201610A':
            num = 11
        else:
            num = 10
        # pastel, husl, and so on
        palette = np.array(sns.color_palette("deep", num))
        # Create a scatter plot.
        a1 = plt.figure(figsize=(8, 8))
        ax = plt.subplot()  #aspect='equal'
        sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int8)])
        # Add the labels for each digit.
        txts = []

        for i in range(num):
            # Position of each label.
            if opt.ab_choose == 'RML201610A':
                sig_name = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
            else:
                sig_name = ['8PSK', 'AM-DSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
            xtext, ytext = np.median(x[colors == i, :], axis=0)
            txt = ax.text(xtext, ytext, sig_name[i], fontsize=24)
            txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
            txts.append(txt)
        plt.xticks([])
        plt.yticks([])
        # plt.axis("off")
        plt.savefig("D:/PYALL/2016B_16384_8_lr0.03_t=0.07_view_chose_"+str(opt.view_chose)+"_mod_l_"+str(opt.mod_l)+"_snr_"+str(opt.snr_tat)+"_epoch_"+str(epoch)+".png", dpi=120)
        plt.close(a1)
        # return f, ax, txts  _view_chose_{args.view_chose}_mod_l_{args.mod_l}
        return txts

    # 绘制特征可视化图
    # if epoch % 40 == 0 or epoch == 1:
    # # Implementing the TSNE Function - ah Scikit learn makes it so easy!
    #     digits_final = TSNE(perplexity=60).fit_transform(X)
    # # Play around with varying the parameters like perplexity, random_state to get different plot
    #     plot(digits_final, Y)
    return l_loss_meter.avg, T_loss_meter.avg, ab_loss_meter.avg, abT_loss_meter.avg



def main():
    """
    Main of MAC

    """
    # parse the args
    args = parse_option()
    # for snr in range(-20, 30, 2):
    # for mon in [0.5, 0.7, 0.9, 0.99, 0.999]:
    # for km in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
    print(str(args.snr_tat))
    # args.snr_tat = "ALL"
    train_loader, test_loader, n_data, index = load_RML2016(args)
    train_sampler = None
    # set the model
    model, contrast, criterion_ab, criterion_l = set_model(args, n_data)

    # set the optimizer
    optimizer = set_optimizer(args, model)

    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            contrast.load_state_dict(checkpoint['contrast'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)
    log_dir = rf'./2018pretrain_logs_'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print(f'{log_dir} exist')
    tb_writer = SummaryWriter(
        log_dir=f'{log_dir}/5.11_2018_k_{args.nce_k}_mon_{args.nce_m}_batch_{args.batch_size}_lr_{args.learning_rate}_view_chose_{args.view_chose}_mod_l_{args.mod_l}_snr_{args.snr_tat}')  # 保存日志到 logs/lr_1e-4_batch_16/ 下
    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        l_loss, l_prob, ab_loss, ab_prob = train(epoch, train_loader, model, contrast, criterion_l, criterion_ab,
                                                 optimizer, args)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('l_loss', l_loss, epoch)
        logger.log_value('l_prob', l_prob, epoch)
        logger.log_value('ab_loss', ab_loss, epoch)
        logger.log_value('ab_prob', ab_prob, epoch)

        tb_writer.add_scalar('l_loss', l_loss, epoch)
        tb_writer.add_scalar('l_prob', l_prob, epoch)
        tb_writer.add_scalar('ab_loss', ab_loss, epoch)
        tb_writer.add_scalar('ab_prob', ab_prob, epoch)
        tb_writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], epoch)

        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'model': model.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            save_file = os.path.join(tb_writer.log_dir, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
            # help release GPU memory
            del state
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
