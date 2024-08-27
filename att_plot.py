import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import random

# Set seed value
seed_value = 42
np.random.seed(seed_value)
data_sig = np.random.rand(1,2,128)

seed_value = 41
np.random.seed(seed_value)
data_att = np.random.rand(1,2,128)

def att_sig(sig, data_att, view, type):
    x = np.arange(128)
    # Create a list of line segments, with each segment having two endpoints and colors mapped from numerical values
    sig_pre_0 = sig[:, 0, :].squeeze()
    sig_pre_1 = sig[:, 1, :].squeeze()
    att_0 = data_att[:, 0, :].squeeze()
    att_1 = data_att[:, 1, :].squeeze()
    points = np.array([x, sig_pre_0]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, 1)
    lc = LineCollection(segments, cmap='coolwarm', norm=norm, label = 'I')   #coolwarm
    lc.set_array(att_0)

    points_2 = np.array([x, sig_pre_1]).T.reshape(-1, 1, 2)
    segments_2 = np.concatenate([points_2[:-1], points_2[1:]], axis=1)
    lc_2 = LineCollection(segments_2, cmap='coolwarm', norm=norm, linestyles= '-.', label = 'Q')  # coolwarm
    lc_2.set_array(att_1)
    # a1 = plt.figure()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.add_collection(lc)
    ax.add_collection(lc_2)
    ax.autoscale()
    # ax.set_xlabel('X')
    # ax.set_ylabel(str(type))

    # plt.legend(loc='upper right')
    plt.legend()
    cbar = plt.colorbar(lc, label='Color Scale')
    # 'plasma' # 'inferno'  # 'magma'  # 'cividis' # 'cool' # 'coolwarm'# 'autumn' # 'spring' # 'summer' # 'winter'  # 'rainbow'  # 'jet'
    plt.savefig("D:/PYALL/2016B_8192_lr0.01_t=0.07_view_chose_" + "_view_" + str(view) + "type" +str(type) +".png", dpi = 800)
    plt.close(fig)
    # plt.show()

att_sig(data_sig, data_att, view = 'AN', type = 'BPSK')
