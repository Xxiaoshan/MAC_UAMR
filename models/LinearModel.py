from __future__ import print_function

import torch.nn as nn

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)


class Linear_CNN2(nn.Module):
    """
    Linear evaluation classification head
    """
    def __init__(self, dim_fea, dim_class, num_classes):
        super(Linear_CNN2, self).__init__()
        self.dim_fea = dim_fea
        self.dim_class = dim_class
        self.fea_number = 6     # MAC-MT4 Intra-domain representation 2 + inter-domain representation 4
        # self.fea_number = 3  # MAC-Dx Intra-domain representation 2 + inter-domain representation 1
        # No DA attention mechanism
        self.fc1 = nn.Sequential(nn.Linear(self.dim_fea * self.fea_number, self.dim_class), nn.ReLU(), nn.Dropout(p=0.6))
        self.fc2 = nn.Sequential(nn.Linear(self.dim_class, num_classes), nn.Softmax())
        self.initilize()

    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)

    def forward(self, x):
        FC1out = self.fc1(x)
        FC2out = self.fc2(FC1out)

        return FC2out
        # return FC2out



class Linear_DA(nn.Module):
    """
    Fine-tuning of domain attention mechanism
    """
    def __init__(self, dim_fea, dim_class, num_classes):
        super(Linear_DA, self).__init__()

        self.dim_fea = dim_fea
        self.dim_class = dim_class
        self.fea_number = 6     # MAC-MT4 Intra-domain representation 2 + inter-domain representation 4
        # self.fea_number = 2  # MAC-Dx Intra-domain representation 1 + inter-domain representation 1
        self.DA_MT4 = nn.Sequential(nn.Conv1d(1, 1, kernel_size=5, padding="same"), nn.Sigmoid())
        self.avgpool_1 = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Sequential(nn.Linear(self.fea_number * self.dim_fea, self.dim_class), nn.SELU(), nn.Dropout(p=0.6))  #RML2016A and RML2016B
        # self.fc1 = nn.Sequential(nn.Linear(256*5, 256), nn.SELU(), nn.Dropout(p=0.3))  #RML2018
        self.fc2 = nn.Sequential(nn.Linear(self.dim_class, num_classes), nn.Softmax())
        self.initilize()

    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)

    def forward(self, x):
        """
        Optimize features for different representation domains
        visualize the importance of representation domains using DA scores
        """
        # DA attention
        b1, c1, len = x.size() #bn*channel*length(feature）
        pooling_out1 = self.avgpool_1(x).view(b1, 1, c1)
        DA_score = self.DA_MT4(pooling_out1).view(b1, c1, 1 )
        DA_out = DA_score*x
        DA_out = DA_out.view(b1, len*c1)

        FC1out = self.fc1(DA_out)  # bn*channel*length(feature）
        FC2out = self.fc2(FC1out)

        return FC2out, DA_score, FC1out
        # return FC2out, DA_score, FC1out
        # return FC2out


