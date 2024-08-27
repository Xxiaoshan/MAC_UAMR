"""
Fine-tuning MAC with RML-X dataset
"""
from __future__ import print_function
import os
import sys
import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import argparse
import tensorboard_logger as tb_logger
from torch.utils.tensorboard import SummaryWriter
from util import adjust_learning_rate, AverageMeter, accuracy, load_RML2016
from models.LinearModel import Linear_CNN2, Linear_DA
from models.backbone import MAC_backbone
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from den_map import cos_midu
def parse_option():
    # MAC
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=30, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=120, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='30,40,50', help='where to decay  lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for Adam')
    parser.add_argument('--nce_k', type=int, default=16384)  #16384
    parser.add_argument('--nce_t', type=float, default=0.07)
    parser.add_argument('--nce_m', type=float, default=0.9)
    parser.add_argument('--feat_dim', type=int, default=256, help='dim of feat for inner product')

    # RESUME
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--model', type=str, default='MAC_backbone', choices=['MAC_backbone'])
    parser.add_argument('--model_path', type=str, default="result/ckpt_epoch_240.pth", help='the model to test')
    parser.add_argument('--layer', type=int, default=6, help='which layer to evaluate')
    # Which domain of the signal should be used for comparison
    # WT is wavelet, AN is amplitude phase, AF is instantaneous frequency, FFT is spectrum
    parser.add_argument('--mod_l', type=str, default='AN', choices=['WT', 'AN', 'AF', 'FFT'])

    # VIEW
    parser.add_argument('--view_chose', type=str, default='ALL', choices=['ALL', 'DB'])

    # data crop threshold
    parser.add_argument('--crop_low', type=float, default=0.2, help='low area in crop')

    # log file
    parser.add_argument('--log', type=str, default='time_linear.txt', help='log file')

    # GPU setting
    parser.add_argument('--gpu', default=torch.cuda.current_device(), type=int, help='GPU id to use.')

    # DATASET
    parser.add_argument('--snr_tat', type=int,
                        default=0, help='训练测试的信噪比，如果为None则为0dB以上全训练测试')
    parser.add_argument('--threads', type=int, default=2,
                        help='number of threads for data loader to use. default=4')
    parser.add_argument('--ab_choose', type=str, default="RML201610A",
                        help="可选择RML201610A, RML201610B, RML2018")
    parser.add_argument("--N_shot", default=1000, type=int)

    # path definition
    parser.add_argument('--RML2016b_path', type=str, default="D:\\PYALL\\RML2016\\RML2016_10B\\pre_snr_train_test\\",
                        help="RML2016B-MT4 Training and testing dataset path")
    parser.add_argument('--RML2018_path', type=str, default="D:\\PYALL\\RML2018\\dataset2018_py_new\\",
                        help="RML2018A-MT4 Training and testing dataset path")
    parser.add_argument('--RML2016a_path', type=str, default="D:\\PYALL\\RML2016\\datasetsave\\2016MV_NEW\\",
                        help="RML2016A-MT4 Training and testing dataset path")
    parser.add_argument('--data_folder', type=str,
                        default="D:\\pyALL\\调制识别数据集\\RML2016\\数据\\datasetsave\\2016IQAA归一化\\", help='path to data')
    parser.add_argument('--save_path', type=str, default="liner_result\\", help='path to save linear classifier')
    parser.add_argument('--tb_path', type=str, default="liner_result\\", help='path to tensorboard')



    opt = parser.parse_args()

    if (opt.data_folder is None) or (opt.save_path is None) or (opt.tb_path is None):
        raise ValueError('one or more of the folders is None: data_folder | save_path | tb_path')

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = opt.model_path.split('/')[-2]
    opt.model_name = '{}_{}'.format(opt.model_name, opt.view_chose)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.save_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    return opt

def set_model(args):
    # set the model
    if args.ab_choose == "RML201610A":
        args.num_class = 11
        model = MAC_backbone(args.feat_dim, args.feat_dim)
    elif args.ab_choose == "RML201610B":
        args.num_class = 10
        model = MAC_backbone(args.feat_dim, args.num_class)
    elif args.ab_choose == "RML2018":
        args.num_class = 24
        model = MAC_backbone(args, args.num_class)
    else:
        print('Dataset selection for non RML series:', args.ab_choose)

    # In order to ensure fair comparison, the DA module will be closed during the linear evaluation phase
    classifier = Linear_DA(args.num_class)
    # classifier = Linear_CNN2(args.num_class)

    # load pre-trained model
    print('==> loading pre-trained model')
    ckpt = torch.load(args.model_path)
    print('==> num_class:', args.num_class)
    model.load_state_dict(ckpt['model'])
    print("==> loaded checkpoint '{}' (epoch {})".format(args.model_path, ckpt['epoch']))
    print('==> done')

    model = model.cuda()
    classifier = classifier.cuda()

    # Linear evaluation will freeze the model and only train the classifier
    # During the fine-tuning phase, both the model and classifier will be trained simultaneously

    model.eval()
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    return model, classifier, criterion


def train(epoch, train_loader, model, classifier, criterion, optimizer, opt):
    """
    MAC one epoch fine-tune
    """
    # model.eval()
    model.train()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    Feature_sim_sum = []
    label_choose_sum = []
    for idx, (input, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.float()
        if opt.gpu is not None:
            input = input.cuda(opt.gpu, non_blocking=True)
        target = target.cuda(opt.gpu, non_blocking=True)

        # ===================forward=====================
        if opt.view_chose == 'DB':
            with torch.no_grad():
                feat_l, feat_ab = model(input, opt.mod_l, opt.view_chose, 'Finetuning')
                # feat = torch.cat((feat_l.detach(), feat_ab.detach()), dim=1)
                feat_DA = torch.cat((feat_l.unsqueeze(1).detach(), feat_ab.unsqueeze(1).detach()), dim=1)
        # elif opt.view_chose == 'ALL':
        #     feat_l, feat_ab, feat_ab1, feat_ab2, feat_ab3 = model(input, opt.mod_l, opt.view_chose, 'linerProbing')
        #     feat = torch.cat((feat_l, feat_ab, feat_ab1, feat_ab2, feat_ab3),dim=1)
        elif opt.view_chose == 'ALL':
        # with torch.no_grad():
            feat_l, feat_ab, feat_ab1, feat_ab2, feat_ab3 = model(input, opt.mod_l, opt.view_chose, 'Finetuning')
            # feat_l, feat_ab, feat_ab2, feat_ab3 = model(input, opt.mod_l, opt.view_chose, 'linerProbing')
            # feat = SE_model(
            #     (feat_l.detach(), feat_ab.detach(), feat_ab1.detach(), feat_ab2.detach(), feat_ab3.detach()), dim=1)
            # feat = torch.cat((feat_l.detach(), feat_ab.detach(), feat_ab1.detach(), feat_ab2.detach(), feat_ab3.detach()),dim=1)
            feat_DA = torch.cat((feat_l.unsqueeze(1).detach(), feat_ab.unsqueeze(1).detach(), feat_ab1.unsqueeze(1).detach(), feat_ab2.unsqueeze(1).detach(), feat_ab3.unsqueeze(1).detach()),dim=1)
            # feat_DA = torch.cat((s
            #                 feat_l.unsqueeze(1), feat_ab.unsqueeze(1), feat_ab1.unsqueeze(1),
            #                 feat_ab2.unsqueeze(1), feat_ab3.unsqueeze(1)), dim=1)
        output, DA_out, Feature_sim = classifier(feat_DA)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, opt, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()
        if opt.ab_choose == 'RML201610A':
            label = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
        else:
            label = ['8PSK', 'AM-DSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
        label_choose = target.cpu()
        # if epoch % 60 == 0:
        #     # Feature visualization:
        #     DA_out = DA_out.squeeze(2)
        #     num_label = len(label)
        #     DA_out = DA_out.cpu().detach().numpy()
        #
        #     for sig_type in range(0, num_label, 1):
        #         index_list = np.where(label_choose == sig_type)
        #         index_num = index_list[0]
        #         DA_out_choose = DA_out[index_num]
        #         DA_out_choose_sum = sum(DA_out_choose).reshape(1, 5)
        #         sum_SE = sum(DA_out_choose_sum, 1)
        #         DA_out_sum_01 = DA_out_choose_sum/sum(sum_SE)
        #         d = plt.figure(figsize=(8, 8))
        #         name_mx = 'SE_view_sig/sum3/acc' + str(top1.avg.item()) + 'epoch' + str(epoch) + '_label_' + label[sig_type] +'_snr_'+ str(opt.snr_tat) +'.png'
        #         sns.heatmap(DA_out_sum_01, annot=True, cmap="YlGnBu")
        #         plt.xlabel("domain")
        #         plt.ylabel("sig_" + label[sig_type])
        #         # plt.show()
        #         plt.savefig(name_mx, transparent=True, dpi=600)
        #         plt.close(d)
        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()
        Feature_sim = Feature_sim.cpu()
        Feature_sim = Feature_sim.detach().numpy()
        label_choose = label_choose.detach().numpy()
        if idx == 0:
            Feature_sim_sum = Feature_sim
            label_choose_sum = label_choose
        else:
            Feature_sim_sum = np.vstack([Feature_sim_sum, Feature_sim])
            label_choose_sum = np.hstack([label_choose_sum, label_choose])

    return top1.avg, top5.avg, losses.avg, Feature_sim_sum, label_choose_sum


def validate(val_loader, model, classifier, criterion, opt):
    """
    MAC evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    # Test setting both the model and classifier to the eval() state simultaneously
    model.eval()
    classifier.eval()
    predsum = []
    targetsum = []

    with torch.no_grad():
        end = time.time()
        best_acc1 = 0
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if opt.gpu is not None:
                input = input.cuda(opt.gpu, non_blocking=True)
            target = target.cuda(opt.gpu, non_blocking=True)

            # compute output
            if opt.view_chose == 'DB':
                with torch.no_grad():
                    feat_l, feat_ab = model(input, opt.mod_l, opt.view_chose, 'linerProbing')
                    feat = torch.cat((feat_l.detach(), feat_ab.detach()), dim=1)
                    feat_DA = torch.cat((feat_l.unsqueeze(1).detach(), feat_ab.unsqueeze(1).detach()), dim=1)
            elif opt.view_chose == 'ALL':
                with torch.no_grad():
                    feat_l, feat_ab, feat_ab1, feat_ab2, feat_ab3 = model(input, opt.mod_l, opt.view_chose,
                                                                          'linerProbing')
                    # feat_l, feat_ab, feat_ab2, feat_ab3 = model(input, opt.mod_l, opt.view_chose,
                    #                                                       'linerProbing')
                    # feat = torch.cat(
                    #     (feat_l.detach(), feat_ab.detach(), feat_ab1.detach(), feat_ab2.detach(), feat_ab3.detach()),
                    #     dim=1)
                    feat_DA = torch.cat((feat_l.unsqueeze(1).detach(), feat_ab.unsqueeze(1).detach(),
                                         feat_ab1.unsqueeze(1).detach(), feat_ab2.unsqueeze(1).detach(),
                                         feat_ab3.unsqueeze(1).detach()), dim=1)

            output, DA_out,_ = classifier(feat_DA)
            loss = criterion(output, target)
            # target



            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, opt, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            app_output = output.data.max(1, keepdim=True)[1].cpu()
            app_target_indices = target.cpu()
            predsum.append(app_output.detach().numpy())
            targetsum.append(app_target_indices.detach().numpy())


            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
# Draw confusion matrix
        # if top1.avg.item() > 90:
        #     if opt.ab_choose == 'RML201610A':
        #         label = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
        #     else:
        #         label = ['8PSK', 'AM-DSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
        #     predsum = np.concatenate(predsum)
        #     targetsum = np.concatenate(targetsum)
        #     trans_mat = confusion_matrix(targetsum, predsum)  # pre 和target分别是预测的和真实的标签，计算时采用真实标签在前，预测标签在后
        #     [m, n] = np.shape(trans_mat)
        #     trans_mat_01 = np.zeros([m, n])
        #     for i in range(0, m):
        #         trans_row = sum(trans_mat[i, :])
        #         for j in range(0, n):
        #             trans_mat_01[i, j] = trans_mat[i, j] / trans_row
        #     formatted_trans_mat_01 = format_matrix(trans_mat_01)
        #     name_mx = 'result/2016b/acc' +str(opt.snr_tat) + str(opt.ab_choose) + str(top1.avg.item()) + '.png'
        #     # 经典蓝色
        #     # plot_confusion_matrix(trans_mat_01, label, normalize=False, title=name_mx)
        #     # opt.matrix = False
        #     # 黄黑渐变
        #     if opt.matrix:
        #         np.random.seed(19680801)
        #         f = plt.figure(figsize=(8, 8))
        #         ax = plt.subplot()
        #         # ax = plt.plot()
        #
        #         # y = ["Patt {}".format(i) for i in range(1, trans_mat.shape[0] + 1)]
        #         # x = ["Patt {}".format(i) for i in range(1, trans_mat.shape[1] + 1)]
        #
        #         y = label
        #         x = label
        #
        #         im, _ = utils.heatmap(formatted_trans_mat_01, y, x, ax=ax, vmin=0,
        #                               cmap="magma_r", cbarlabel="transition countings")
        #         # annotate_heatmap(im, valfmt="{x:d}", size=10, threshold=20,
        #         #                  textcolors=("red", "white"), fontsize=12)
        #         utils.annotate_heatmap(im, data=formatted_trans_mat_01, valfmt="{x:f}", threshold=0.1,
        #                                textcolors=("black", "white"), fontsize=12)
        #         # threshold = 20 代表的是字体颜色的控制
        #
        #         # 紧致图片效果，方便保存
        #         plt.tight_layout()
        #         plt.savefig(name_mx, transparent=True, dpi=800)
        #         plt.close(f)

# Feature visualization

    # def sim_calculate(feature, target, label):
    #     num_label = len(label)
    #     for sig_type in range(0, num_label, 1):
    #         index_list = np.where(target == sig_type)
    #         index_num = index_list[0]
    #         DA_out_choose = feature[index_num]
    #
    #         DA_out_choose_sum = sum(DA_out_choose).reshape(1, 5)
    #         sum_SE = sum(DA_out_choose_sum, 1)
    #         DA_out_sum_01 = DA_out_choose_sum / sum(sum_SE)
    #         d = plt.figure(figsize=(8, 8))
    #         name_mx = 'SE_view_sig/sum3/acc' + str(top1.avg.item()) + 'epoch' + str(epoch) + '_label_' + label[
    #             sig_type] + '_snr_' + str(opt.snr_tat) + '.png'
    #         sns.heatmap(DA_out_sum_01, annot=True, cmap="YlGnBu")
    #         plt.xlabel("domain")
    #         plt.ylabel("sig_" + label[sig_type])
    #         # plt.show()
    #         plt.savefig(name_mx, transparent=True, dpi=600)
    #         plt.close(d)

    return top1.avg, top5.avg, losses.avg


def format_matrix(matrix):
    formatted_matrix = np.where((matrix > 0 ) & (matrix < 1), np.round(matrix, 2), np.round(matrix).astype(int))
    return formatted_matrix



def main():
    args = parse_option()
    # for km in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
    # for mon in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999]:
    M_times = 100     # The number of Monte Carlo experiments
    # for snr in [0, 12, 24]:
    acc_times_list = []
    for times in range(0, M_times, 1):
        print("Current number of Monte Carlo experiments=={}".format(times))
        # for snr in range(0, 2, 2):
        global best_acc1
        best_acc1 = 0
        # args.snr_tat = snr
        # args.nce_k = km
        # args.nce_k = km
        # args.snr_tat = "ALL"
        # 2016A Pre training path
        args.model_path = rf'./ALL_logs_/new_32768_t=0.07_batch_32_lr_0.03_view_chose_{args.view_chose}_mod_l_{args.mod_l}_snr_{args.snr_tat}/ckpt_epoch_240.pth'
        # 2018 Pre training path
        # args.model_path = rf'./2018pretrain_logs_/THY_4.30_2018_k_{args.nce_k}_mon_{args.nce_m}_batch_64_lr_0.01_view_chose_{args.view_chose}_mod_l_{args.mod_l}_snr_{args.snr_tat}/ckpt_epoch_240.pth'
        # args.model_path = rf'./2016B_logs_ablation_DX/D1_2016B_8192_0.9_batch_32_lr_0.01_view_chose_DB_mod_l_AN_snr_{args.snr_tat}/ckpt_epoch_240.pth'
        # args.model_path = rf'./ALL_logs_/new_32768_t=0.07_batch_32_lr_0.03_view_chose_ALL_mod_l_AN_snr_{args.snr_tat}/ckpt_epoch_240.pth'
        if args.gpu is not None:
            print("Use GPU: {} for training".format(args.gpu))

        train_loader, val_loader, n_data, indics = load_RML2016(args)
        # set the model
        model, classifier, criterion = set_model(args)
        lr_model = 0.0001
        lr_class = 0.0001
        # The choice of optimizer, if it is a linear evaluation, only updates the classifier
        optimizer = optim.Adam(
            [{"params":filter(lambda p: p.requires_grad, model.parameters()),'lr':lr_model},
             {"params":filter(lambda y: y.requires_grad, classifier.parameters()),'lr':lr_class}],
             betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True)
        # optimizer = optim.SGD(
        #     [{"params":filter(lambda p: p.requires_grad, model.parameters()),'lr':0.01},
        #       {"params":filter(lambda y: y.requires_grad, classifier.parameters()),'lr':0.1}],
        #                       lr=args.learning_rate,
        #                       momentum=args.momentum,
        #                       weight_decay=args.weight_decay)

        cudnn.benchmark = True

        # optionally resume linear classifier
        args.start_epoch = 1
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch'] + 1
                best_acc1 = checkpoint['best_acc1']
                if args.gpu is not None:
                    # best_acc1 may be from a checkpoint from a different GPU
                    best_acc1 = best_acc1.to(args.gpu)
                classifier.load_state_dict(checkpoint['classifier'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        args.start_epoch = 1
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location='cpu')
                args.start_epoch = checkpoint['epoch'] + 1
                classifier.load_state_dict(checkpoint['classifier'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                best_acc1 = checkpoint['best_acc1']
                best_acc1 = best_acc1.cuda()
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
                del checkpoint
                torch.cuda.empty_cache()
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        # tensorboard
        logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)
        # log_dir = rf'./part_funeting_logs'
        log_dir = rf'./2016_finetuning_shot_log'
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        else:
            print(f'{log_dir} exist')
        tb_writer = SummaryWriter(
            # log_dir=f'{log_dir}/10shot_time{times}_detAdam0.001-0.001_2016A_{args.batch_size}_lr_{args.learning_rate}_{args.view_chose}_mod_l_{args.mod_l}_snr_{args.snr_tat}')
            log_dir=f'{log_dir}/N_shot_{args.N_shot}_2016A_Adam_lr_model_{lr_model}_lr_class_{lr_class}_{args.batch_size}_{args.view_chose}_mod_l_{args.mod_l}_snr_{args.snr_tat}')# 保存日志到
        if args.ab_choose == 'RML201610A':
            label = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
        else:
            label = ['8PSK', 'AM-DSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
        for epoch in range(args.start_epoch, args.epochs + 1):

            adjust_learning_rate(epoch, args, optimizer)
            print("==> training...")

            time1 = time.time()
            train_acc, train_acc5, train_loss, Feature_sim_sum, label_choose_sum = train(epoch, train_loader, model, classifier, criterion, optimizer, args)
            time2 = time.time()
            print('train epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

            logger.log_value('train_acc', train_acc, epoch)
            logger.log_value('train_acc5', train_acc5, epoch)
            logger.log_value('train_loss', train_loss, epoch)
            # tensorboard logger
            pass
        print("==> testing...")
        test_acc, test_acc5, test_loss = validate(val_loader, model, classifier, criterion, args)
        # save model
        print('==> Saving...')
        state = {
            'opt': args,
            'epoch': epoch,
            'classifier': classifier.state_dict(),
            'model': model.state_dict(),
            'best_acc1': test_acc,
            'optimizer': optimizer.state_dict(),
        }
        save_name = 'testacc_{acc}ckpt_epoch_{epoch}.pth'.format(acc=test_acc.item(), epoch=args.epochs)
        save_name = os.path.join(tb_writer.log_dir, save_name)
        print('saving last model!')
        torch.save(state, save_name)

        acc_times_list.append(test_acc.item())
    min_acc = min(acc_times_list)
    max_acc = max(acc_times_list)
    avg_acc = sum(acc_times_list)/len(acc_times_list)
    txt_name = log_dir + '/snr_' + str(args.snr_tat) + '_'+ str(args.N_shot) \
               + 'shots_Times_result_' + str(min_acc) +'_' + str(max_acc) + '_' + str(avg_acc) +'.txt'
    np.savetxt(txt_name, acc_times_list)
    print('acc_times_list = ', acc_times_list)
    print('min_acc{}_max_acc{}_avg_acc{}'.format(min_acc, max_acc, avg_acc))

if __name__ == '__main__':
    best_acc1 = 0
    main()


