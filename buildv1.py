import torch
import torch.nn as nn
import pickle
from torch.utils import data
from random import *
import math
import random

# intra-domain
#Perform data augmentation, flipping, rotation, and Gaussian noise on communication signals
#The inputs are input signals in the format of [batch, dim, length] eg: (32 * 4 * 128)
#Random_deed is a random type selection that can also be given 1-3 as flipping, 4-bit Gaussian noise, and 5-7 as rotation
#The default rotation angle for theta list is [0, math.pi/2, math.pi, math.pi/2*3]
#The default variance for Gaussian noise is [0, 0.05, 0.1, 0.15]
#Return a sample after data augmentation, with dimensions consistent with the original data

# Data augmentation methods used intra-domain representation learning
def Data_Augmentation(inputs, random_seed, theta_list, std_list):
    if random_seed == None:
        random_seed = randint(1, 10)
    if theta_list == None:
        theta_list = [0, math.pi / 2, math.pi, math.pi / 2 * 3]
    if std_list == None:
        std_list = [0, 0.05, 0.1, 0.15]

    if random_seed == 4:
        flip_v = torch.zeros_like(inputs)
        flip_v[:, 0, :] = inputs[:, 0, :]
        flip_v[:, 1, :] = -inputs[:, 1, :]
        Data_Aug = flip_v
    elif random_seed == 2:
        flip_h = torch.zeros_like(inputs)
        flip_h[:, 0, :] = -inputs[:, 0, :]
        flip_h[:, 1, :] = inputs[:, 1, :]
        Data_Aug = flip_h
    elif random_seed == 3:
        flip_v_h = torch.zeros_like(inputs)
        flip_v_h[:, 0, :] = -inputs[:, 0, :]
        flip_v_h[:, 1, :] = -inputs[:, 1, :]
        Data_Aug = flip_v_h

    elif random_seed == 1:
        std_temp = random.choice(std_list)
        noise = torch.normal(mean=0, std=std_temp, size=inputs.shape).cuda()
        Data_Aug = inputs + noise

    elif random_seed > 4 & random_seed <= 7:
        theta = random.choice(theta_list)
        i_data = inputs[:, 0, :]
        q_data = inputs[:, 1, :]
        i_data_gen = math.cos(theta) * i_data - math.sin(theta) * q_data
        q_data_gen = math.sin(theta) * i_data + math.cos(theta) * q_data
        Data_Aug = torch.cat([i_data_gen.unsqueeze(1), q_data_gen.unsqueeze(1)], dim=1)

    elif random_seed > 7:
        sig_len = 128 #1024
        I_random_mask_num = randint(1, sig_len)
        Q_random_mask_num = randint(1, sig_len)
        mask_len = 10
        inputs[:, 0, I_random_mask_num: I_random_mask_num + mask_len] = 0
        inputs[:, 1, Q_random_mask_num: Q_random_mask_num + mask_len] = 0
        Data_Aug = inputs

    return Data_Aug


def flip_row(inputs, label):
    flip_v = torch.zeros_like(inputs)
    flip_h = torch.zeros_like(inputs)
    flip_v_h = torch.zeros_like(inputs)
    flip_v[:, 0, :] = inputs[:, 0, :]
    flip_v[:, 1, :] = -inputs[:, 1, :]
    flip_h[:, 0, :] = -inputs[:, 0, :]
    flip_h[:, 1, :] = inputs[:, 1, :]
    flip_v_h[:, 0, :] = -inputs[:, 0, :]
    flip_v_h[:, 1, :] = -inputs[:, 1, :]
    return torch.cat([inputs, flip_v, flip_h, flip_v_h], dim=0), torch.cat([label, label, label, label], dim=0)


def flip(inputs):
    random_seed = randint(1, 3)
    if random_seed == 1:
        flip_v = torch.zeros_like(inputs)
        flip_v[:, 0, :] = inputs[:, 0, :]
        flip_v[:, 1, :] = -inputs[:, 1, :]
        flip_data = flip_v
    elif random_seed == 2:
        flip_h = torch.zeros_like(inputs)
        flip_h[:, 0, :] = -inputs[:, 0, :]
        flip_h[:, 1, :] = inputs[:, 1, :]
        flip_data = flip_h
    else:
        flip_v_h = torch.zeros_like(inputs)
        flip_v_h[:, 0, :] = -inputs[:, 0, :]
        flip_v_h[:, 1, :] = -inputs[:, 1, :]
        flip_data = flip_v_h

    return flip_data


def rotation(inputs, label, theta: list):
    out = []
    label_list = []
    for r in theta:
        i_data = inputs[:, 0, :]
        q_data = inputs[:, 1, :]
        i_data_gen = math.cos(r) * i_data - math.sin(r) * q_data
        q_data_gen = math.sin(r) * i_data + math.cos(r) * q_data
        out.append(torch.cat([i_data_gen.unsqueeze(1), q_data_gen.unsqueeze(1)], dim=1))
        label_list.append(label)

    return torch.cat(out, dim=0), torch.cat(label_list, dim=0)


def addguassiannoise(inputs, label, theta_2_all: list):
    data_gen = []
    label_gen = []
    for theta_2 in theta_2_all:
        noise = torch.normal(mean=0, std=theta_2, size=inputs.shape).cuda()
        data_gen.append(inputs + noise)
        label_gen.append(label)
    return torch.cat(data_gen, dim=0), torch.cat(label_gen, dim=0)


def rotation_flip(inputs, label):
    flip_data, flip_label = flip(inputs, label)
    r_data, r_label = rotation(inputs, label, [math.pi / 2, 3 / 2 * math.pi])
    return torch.cat([flip_data, r_data], dim=0), torch.cat([flip_label, r_label], dim=0)


def load_pickle(file_name):
    f = open(file_name, "rb+")
    data = pickle.load(f)
    f.close()
    return data
