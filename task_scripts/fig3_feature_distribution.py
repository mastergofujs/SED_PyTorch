import sys

sys.path.append('..')
import numpy as np
from MainClasses.SBetaVAE import SBetaVAE
from MainClasses.TorchDataset import TorchDataset
from MainClasses.DataHandler import DataHandler
import pandas as pd
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import torch
import random
import os

pd.set_option('precision', 4)


def setup_args():
    parser = ArgumentParser(description='Fig. 3: visualize the distribution of features learned from the proposed '
                                        'method')
    # NOTE: parameters below are not supposed to change.
    DATASET = 'tut_data'
    TIMESTEP = 5
    NAME = 'att_s_beta_vae'
    NFOLDS = 4
    NUM_EVENTS = 6

    parser.add_argument('-name', "--name", type=str, default=NAME)
    parser.add_argument('-nfolds', "--nfolds", type=int, default=NFOLDS)
    parser.add_argument("-dt", "--dataset", type=str, default=DATASET)
    parser.add_argument('-t', "--time_step", type=int, default=TIMESTEP)
    parser.add_argument('-o', "--result_path", type=str, default='../aed_data/' + DATASET + '/result/')
    parser.add_argument('-k', "--num_events", type=int, default=NUM_EVENTS)
    parser.add_argument('-rh', "--return_h", type=bool, default=True)
    parser.add_argument('-n', '--num_samples', type=int, default=2000)

    # Parameters below are changeable.
    parser.add_argument('-gpu', "--gpu_device", type=str, default='0')
    return parser


def running(options):
    fold = 1
    dh = DataHandler('tut_data')
    # 2.Load data.
    test_dataset = TorchDataset(options, type_='test', fold=fold)

    # build model and load weights
    sb_vae = SBetaVAE(options).cuda()
    sb_vae.name = 'att_s_beta_vae_6'
    sb_vae.load_state_dict(torch.load(options.result_path + sb_vae.name + '/fold_' + str(fold) + '_cp_weight.h5'))

    # get the bottleneck features
    h_out = torch.Tensor().cuda()
    num_to_plot = 300

    for i in range(options.num_events):
        n_ = 0
        while n_ < num_to_plot:
            item = random.randint(0, len(test_dataset) - 1)
            if test_dataset[item][1][i] == 1:  # ensure the test data contains the i-th type of event
                n_ += 1
                x_data, y_data = test_dataset[item]
                with torch.no_grad():
                    dec_out, detectors_out, z_stars, alphas, (mu, log_var), bottleneck_features = sb_vae(torch.from_numpy(x_data)
                                                                                                         .float().cuda())
                    h_out = torch.cat([h_out, torch.relu(bottleneck_features)[:, :, i]])

    # visualization
    dh.visualization(datas=h_out.cpu().numpy(), name='sbvae')


if __name__ == '__main__':
    args = setup_args()
    options = args.parse_args()
    running(options)
