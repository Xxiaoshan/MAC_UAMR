import torch
import torch.nn as nn
from buildv1 import Data_Augmentation
from random import *

class CNN2IQ(nn.Module):
    """
    A simple CNN with 2 layers
    IQ road is considered as an image with a width of 2
    """
    def __init__(self, dim_pre, num_classes):
        super(CNN2IQ, self).__init__()
        self.channl_1 = 50
        self.channl_2 = 50
        self.sig_len = 121
        self.cnn1 = nn.Sequential(nn.Conv2d(1, self.channl_1, kernel_size=(1, 8), padding="same"),
                                  nn.BatchNorm2d(self.channl_1),
                                  nn.ReLU(),nn.Dropout(p=0.5))

        self.cnn2 = nn.Sequential(nn.Conv2d(self.channl_1, self.channl_2, kernel_size=(2, 8), padding="valid"),
                                  nn.BatchNorm2d(self.channl_2),
                                  nn.ReLU(),nn.Dropout(p=0.5))
        self.FL = nn.Flatten()
        self.fc1 = nn.Sequential(nn.Linear(self.sig_len * self.channl_2, dim_pre), nn.ReLU(), nn.Dropout(p=0.5))
        self.fc2 = nn.Sequential(nn.Linear(dim_pre, num_classes), nn.Softmax())

    def forward(self, x, mod):
        """
        Pre training and linear evaluation return different features during the forward process
        """
        batch_size = x.size(0)
        x3 = x[:, 0:2, :]
        x3 = torch.unsqueeze(x3, 1)
        CNN1out = self.cnn1(x3)
        CNN2out = self.cnn2(CNN1out)
        FLhid = self.FL(CNN2out)
        FC1out = self.fc1(FLhid)
        if mod == 'pretrain':
            return FC1out
        if mod == 'Finetuning':
            return FLhid

class MAC_backbone(nn.Module):
    def __init__(self, dim_pre, num_classes):
        super(MAC_backbone, self).__init__()
        self.random_seed = randint(2, 8)  #Control the selection of data augmentation
        self.l_to_SD = CNN2IQ(dim_pre, num_classes)
        self.TD_to_l = CNN2IQ(dim_pre, num_classes)
        self.TD1_to_l = CNN2IQ(dim_pre, num_classes)
        self.TD2_to_l = CNN2IQ(dim_pre, num_classes)
        self.TD3_to_l = CNN2IQ(dim_pre, num_classes)

    def forward(self, x, mod1, view, mod2):
        batch_size = x.size(0)
        if x.size(2) > x.size(1):
            batch_size = x.size(0)
        else:
            x = torch.permute(x, (0, 2, 1))
        SD = x[:, 0:2, :]  #IQ
        if view == 'ALL':
            # Inter-domain contrastive learning feature extraction
            TD = x[:, 4:6, :]  #AN
            TD1 = x[:, 2:4, :]   #wt
            TD2 = x[:, 6:8, :]   #AF
            TD3 = x[:, 8:10, :]  #FFT

            feat_l = self.l_to_SD(SD, mod2)
            feat_TD = self.TD_to_l(TD, mod2)
            feat_TD1 = self.TD1_to_l(TD1, mod2)
            feat_TD2 = self.TD2_to_l(TD2, mod2)
            feat_TD3 = self.TD3_to_l(TD3, mod2)

            # Inter-domain contrastive learning feature extraction
            SD1 = Data_Augmentation(SD, self.random_seed, None, None)
            feat_SD1 = self.l_to_SD(SD1, mod2)

            return feat_l, feat_TD, feat_TD1, feat_TD2, feat_TD3, feat_SD1

        elif view == 'DB':
            # Inter-domain contrastive learning feature extraction
            if mod1 == 'AN':
                TD = x[:, 4:6, :]
            elif mod1 == 'WT':
                TD = x[:, 2:4, :]  # wt
            elif mod1 == 'FFT':
                TD = x[:, 8:10, :]
            elif mod1 == 'AF':
                TD = x[:, 6:8, :]

            # Inter-domain contrastive learning feature extraction
            SD1 = Data_Augmentation(SD, self.random_seed, None, None)

            feat_l = self.l_to_SD(SD, mod2)
            feat_TD = self.TD_to_l(TD, mod2)
            feat_SD = self.l_to_SD(SD1, mod2)
            return feat_l, feat_TD, feat_SD

class CLDNN(nn.Module):
    def __init__(self):
        super(CLDNN, self).__init__()
        # Considering the data length of 1024, temporal network (LSTM) has been added to the backbone
        dr = 0.3
        self.fea_dim1 = 32
        self.fea_dim2 = 128
        self.conv1 = nn.Conv1d(2, self.fea_dim1, kernel_size=24)
        self.lstm1 = nn.LSTM(self.fea_dim1, self.fea_dim2, batch_first=True)
        self.dropout1 = nn.Dropout(dr)
        self.lstm2 = nn.LSTM(self.fea_dim2, self.fea_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(dr)
        self.conv2 = nn.Conv1d(self.fea_dim2, self.fea_dim2, kernel_size=8)
        self.global_max_pooling = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(self.fea_dim2, self.fea_dim2)
        self.BN = nn.BatchNorm1d(self.fea_dim2)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(self.fea_dim2, 64)
        self.BN2 = nn.BatchNorm1d(64)


    def forward(self, x, args):
        x = self.conv1(x)
        x = torch.permute(x, [0, 2, 1])
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, h = self.lstm2(x)
        x = self.dropout2(x)
        x = torch.permute(x, [0, 2, 1])
        x = self.conv2(x)
        x_fet = self.global_max_pooling(x)
        x_fet = x_fet.squeeze(dim=2)
        x = self.fc1(x_fet)
        x = self.act(self.BN(x))
        x = self.fc2(x)
        x = self.BN2(x)

        if args == "pretrain":
            return x
        elif args == "linerProbing":
            return x_fet


class MAC_backbone2(nn.Module):
    def __init__(self, args, num_classes):
        super(MAC_backbone2, self).__init__()

        self.l_to_SD = CLDNN()
        self.TD_to_l = CLDNN()
        self.TD1_to_l = CLDNN()
        self.TD2_to_l = CLDNN()
        self.TD3_to_l = CLDNN()

    def forward(self, x, mod1, view, mod2):
        # l, ab = torch.split(x, [1, 2], dim=1)
        batch_size = x.size(0)
        if x.size(2) > x.size(1):
            batch_size = x.size(0)
        else:
            x = torch.permute(x, (0, 2, 1))
        SD = x[:, 0:2, :]  # IQ
        if view == 'ALL':
            # Inter-domain contrastive learning feature extraction
            TD = x[:, 4:6, :]  # AN
            TD1 = x[:, 2:4, :]  # wt
            TD2 = x[:, 6:8, :]  # AF
            TD3 = x[:, 8:10, :]  # FFT

            feat_l = self.l_to_SD(SD, mod2)
            feat_TD = self.TD_to_l(TD, mod2)
            feat_TD1 = self.TD1_to_l(TD1, mod2)
            feat_TD2 = self.TD2_to_l(TD2, mod2)
            feat_TD3 = self.TD3_to_l(TD3, mod2)

            # Inter-domain contrastive learning feature extraction
            SD1 = Data_Augmentation(SD, self.random_seed, None, None)
            feat_SD1 = self.l_to_SD(SD1, mod2)

            return feat_l, feat_TD, feat_TD1, feat_TD2, feat_TD3, feat_SD1
            # return feat_l, feat_ab, feat_ab2, feat_ab3


        elif view == 'DB':

            # Inter-domain contrastive learning feature extraction

            if mod1 == 'AN':

                TD = x[:, 4:6, :]

            elif mod1 == 'WT':

                TD = x[:, 2:4, :]  # wt

            elif mod1 == 'FFT':

                TD = x[:, 8:10, :]

            elif mod1 == 'AF':

                TD = x[:, 6:8, :]

            # Inter-domain contrastive learning feature extraction

            SD1 = Data_Augmentation(SD, self.random_seed, None, None)

            feat_l = self.l_to_SD(SD, mod2)

            feat_TD = self.TD_to_l(TD, mod2)

            feat_SD = self.l_to_SD(SD1, mod2)

            return feat_l, feat_TD, feat_SD