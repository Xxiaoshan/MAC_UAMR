from __future__ import print_function

import pywt
import torch
from torch.utils import data
from torch.utils.data import TensorDataset
import pickle
from torch.utils.data import Dataset
import random
import numpy as np

def load_pickle(file_name):
    f = open(file_name, "rb+")
    data = pickle.load(f)
    f.close()
    return data


def load_RML2016(args):
    """Load RML2016 dataset.
    The data is split and normalized between train and test sets.
    """
    args.cudaTF = torch.cuda.is_available()
    kwargs = {'num_workers': args.threads,
              'pin_memory': True} if args.cudaTF else {}

    if args.ab_choose == 'RML201610A':
        if args.snr_tat == "ALL":
            filename_train_sne = args.RML2016a_path + "train_ALL_SNR_MV_dataset"
            filename_test_sne = args.RML2016a_path + "test_ALL_SNR_MV_dataset"

            IQ_train = load_pickle(filename_train_sne)
            IQ_test = load_pickle(filename_test_sne)
            train_loader_IQ = data.DataLoader(dataset=IQ_train, batch_size=args.batch_size, shuffle=True, **kwargs)
            test_loader_IQ = data.DataLoader(dataset=IQ_test, batch_size=args.batch_size, shuffle=True, **kwargs)
        else:
            filename_train_sne = args.RML2016a_path + str(args.snr_tat) + "_train_MV_dataset"
            filename_test_sne = args.RML2016a_path + str(args.snr_tat) + "_test_MV_dataset"
            IQ_train = load_pickle(filename_train_sne)
            IQ_test = load_pickle(filename_test_sne)
            # 采用随机的index
            indics = torch.randperm(len(IQ_train.tensors[0]))
            # 加载保存的index
            # indics = torch.load("73.7_6dBbest_indice.pt")

            a0 = IQ_train.tensors[0]
            a1 = IQ_train.tensors[1]
            a3 = torch.arange(0, 8800)
            shuffled_data = a0[indics]
            shuffled_label = a1[indics]
            #给予数据集读入时三个参数，数据、标签、以及每个样本在当前数据集中的标号


            a1_selected = []
            a0_selected = []
            a3_selected = []

            count_num = args.N_shot   # 50 shot
            # 从0到10遍历，每个数字选取10个

            for i in range(11):  # 0到10共11个数字
                count = 0
                cout_idx = 0
                for num in a1:
                    cout_idx = cout_idx + 1
                    if num == i:
                        a1_selected.append(shuffled_label[cout_idx].unsqueeze(dim=0))
                        a0_selected.append(shuffled_data[cout_idx, :, :].unsqueeze(dim=0))
                        a3_selected.append(a3[cout_idx].unsqueeze(dim=0))
                        count += 1
                        if count == count_num:
                            break
            # 输出选中数字列表
            print(len(a1_selected))
            a1_selected = torch.cat(a1_selected)
            a0_selected = torch.cat(a0_selected)
            a3_selected = torch.cat(a3_selected)
            new_dataset = TensorDataset(a0_selected, a1_selected, a3_selected)
            # new_dataset = TensorDataset(a0[10:120, :, :], a1[10:120], a3[10:120])   #控制选择的输入数据集大小

            # #part train sample
            # data_temp = IQ_train.tensors[0]
            # label_temp = IQ_train.tensors[1]
            # IQ_train = data.TensorDataset(data_temp[0:8800,:,:], label_temp[0:8800])

            # train_loader_IQ = data.DataLoader(dataset=IQ_train, batch_size=args.batch_size, shuffle=True,  **kwargs)
            # test_loader_IQ = data.DataLoader(dataset=IQ_test, batch_size=args.batch_size, shuffle=True,  **kwargs)
            train_sampler = None
            train_loader_IQ = data.DataLoader(new_dataset, batch_size=args.batch_size, shuffle=True,  **kwargs, sampler=train_sampler)
            test_loader_IQ = data.DataLoader(IQ_test, batch_size=args.batch_size, shuffle=True,  **kwargs, sampler=train_sampler)
    elif args.ab_choose == 'RML201610B':

        #选择2016b的数据集
        filename_train_sne = args.RML2016b_path + str(args.snr_tat) + "_MT4_train_dataset"
        filename_test_sne = args.RML2016b_path + str(args.snr_tat) + "_MT4_test_dataset"
        IQ_train = load_pickle(filename_train_sne)
        IQ_test = load_pickle(filename_test_sne)
        # 给予数据集读入时三个参数，数据、标签、以及每个样本在当前数据集中的标号
        a0 = IQ_train.tensors[0]
        a1 = IQ_train.tensors[1]
        a3 = torch.arange(0, len(IQ_train))
        new_dataset = TensorDataset(a0, a1, a3)  # 控制选择的输入数据集大小
        train_sampler = None
        train_loader_IQ = data.DataLoader(dataset=new_dataset, batch_size=args.batch_size, shuffle=True, **kwargs, sampler=train_sampler)
        test_loader_IQ = data.DataLoader(dataset=IQ_test, batch_size=args.batch_size, shuffle=True, **kwargs, sampler=train_sampler)

    else:
        assert args.ab_choose == 'RML2018'
        #选择2018的数据集

        filename_train_sne = args.RML2018_path +"_MV4_snr_"+ str(args.snr_tat) + "_train_dataset"
        filename_test_sne = args.RML2018_path +"_MV4_snr_"+ str(args.snr_tat) + "_test_dataset"
        IQ_train = load_pickle(filename_train_sne)
        IQ_test = load_pickle(filename_test_sne)
        # 采用随机的index
        indics = torch.randperm(len(IQ_train.tensors[0]))
        # 加载保存的index
        # indics = torch.load("73.7_6dBbest_indice.pt")
        a0 = IQ_train.tensors[0]
        a1 = IQ_train.tensors[1]
        a3 = torch.arange(0, len(IQ_train.tensors[0]))
        if args.N_shot == 0:
            new_dataset = TensorDataset(a0, a1, a3)
        else:
            shuffled_data = a0[indics]
            shuffled_label = a1[indics]
            # 给予数据集读入时三个参数，数据、标签、以及每个样本在当前数据集中的标号

            a1_selected = []
            a0_selected = []
            a3_selected = []

            count_num = args.N_shot  # 50 shot
            # 从0到10遍历，每个数字选取10个

            for i in range(24):  # 0到10共11个数字
                count = 0
                cout_idx = 0
                for num in a1:
                    cout_idx = cout_idx + 1
                    if num == i:
                        a1_selected.append(shuffled_label[cout_idx].unsqueeze(dim=0))
                        a0_selected.append(shuffled_data[cout_idx, :, :].unsqueeze(dim=0))
                        a3_selected.append(a3[cout_idx].unsqueeze(dim=0))
                        count += 1
                        if count == count_num:
                            break
            # 输出选中数字列表
            print(len(a1_selected))
            a1_selected = torch.cat(a1_selected)
            a0_selected = torch.cat(a0_selected)
            a3_selected = torch.cat(a3_selected)
            new_dataset = TensorDataset(a0_selected, a1_selected, a3_selected)

        train_sampler = None
        train_loader_IQ = data.DataLoader(dataset=new_dataset, batch_size=args.batch_size, shuffle=True, **kwargs, sampler=train_sampler)
        test_loader_IQ = data.DataLoader(dataset=IQ_test, batch_size=args.batch_size, shuffle=True, **kwargs, sampler=train_sampler)
        indics = 0


    # num of samples

    print('choosed dataset: ' + str(args.ab_choose))
    print('filename_train_dataset: ' + str(filename_train_sne))
    n_data = len(new_dataset)
    print('number of samples: {}'.format(n_data))

    return train_loader_IQ, test_loader_IQ, n_data, indics


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
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


def accuracy(output, target, opt, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        if opt.ab_choose == "SP_RML201610B":
            pred [pred == 2] = 12
            pred [pred > 2] -= 1
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    meter = AverageMeter()
