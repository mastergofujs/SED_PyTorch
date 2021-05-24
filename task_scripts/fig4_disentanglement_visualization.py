import sys

sys.path.append('..')
import numpy as np
from MainClasses.SBetaVAE import SBetaVAE
from MainClasses.TorchDataset import TorchDataset
from argparse import ArgumentParser
import torch
import matplotlib.pylab as plt
import os


def setup_args():
    """
    Parameters definition
    :return:
    """
    parser = ArgumentParser(description='Fig. 4: disentanglement visualization')
    # NOTE: parameters below are not supposed to change.
    DATASET = 'freesound'
    TIMESTEP = 5
    NAME = 'att_s_beta_vae'
    NFOLDS = 4
    # NUM_EVENTS = 5
    parser.add_argument('-name', "--name", type=str, default=NAME)
    parser.add_argument('-nfolds', "--nfolds", type=int, default=NFOLDS)
    parser.add_argument("-dt", "--dataset", type=str, default=DATASET)
    parser.add_argument('-t', "--time_step", type=int, default=TIMESTEP)
    parser.add_argument('-o', "--result_path", type=str, default='../aed_data/' + DATASET + '/result/')
    parser.add_argument('-k', "--num_events", type=int, default=5)
    parser.add_argument('-rh', "--return_h", type=bool, default=False)
    parser.add_argument('-d', "--feature_dim", type=int, default=216)

    # Parameters below are changeable.
    parser.add_argument('-b', "--batch_size", type=int, default=128)
    parser.add_argument('-e', "--epoch", type=int, default=50)
    parser.add_argument('-lr', "--learning_rate", type=float, default=0.0002)
    parser.add_argument('-gpu', "--gpu_device", type=str, default='1')
    parser.add_argument('-m', "--mix_data", type=int, default=0)  # if generate new sub-dataset using freesound
    return parser


def running(options):
    fold = 1
    K = 5
    num_to_generate = 250

    # setup the configure list, which decides the events and the latent factors to visualize:
    # [event index, latent factor index, min value, mid value, max value]
    # Note please: you are supposed to alter the event index and latent factor index to find the most clear samples.
    infs = []
    infs.append([1, 6, 0, 0, 0])
    infs.append([1, 14, 0, 0, 0])
    infs.append([2, 6, 0, 0, 0])
    infs.append([2, 14, 0, 0, 0])
    infs.append([4, 6, 0, 0, 0])
    infs.append([4, 14, 0, 0, 0])
    infs = np.asarray(infs, dtype='float')

    # load testing set
    test_dataset = TorchDataset(options, type_='test', fold=fold)
    test_data = test_dataset.x_data
    test_label = test_dataset.y_data
    # build model and load weights
    sb_vae = SBetaVAE(options).cuda()
    sb_vae.name = 'att_s_beta_vae_5'
    sb_vae.load_state_dict(torch.load(options.result_path + sb_vae.name + '/fold_' + str(fold) + '_cp_weight.h5'))
    pic = []
    min_plot, mid_plot, max_plot = [], [], []
    plots = [min_plot, mid_plot, max_plot]

    for j in range(len(infs)):
        # get event index
        event_index = int(infs[j][0])
        gen_label = np.zeros((K,))
        gen_label[event_index] = 1
        indexs = list()
        for index in range(len(test_label)):
            if test_label[index][event_index] == gen_label[event_index]:
                indexs.append(index)

        # get z* and decode it to reconstruct MFCCs

        dec_out, detectors_out, z_stars, alphas, (mu, log_var) = sb_vae(torch.from_numpy(test_data[indexs][:5000]).float().cuda())
        dim = int(infs[j][1])
        max_z = z_stars[:, dim, event_index].max()
        min_z = z_stars[:, dim, event_index].min()
        mid_z = (max_z + min_z) / 2.0
        infs[j][2] = min_z
        infs[j][3] = mid_z
        infs[j][4] = max_z

        # Change the value of event-specific factor to visualize the disentanglement results
        idx = 0
        for value in [min_z, mid_z, max_z]:
            z_stars[:, dim] = value
            generated_datas = sb_vae.decoder(z_stars[:, :, event_index])
            generated_datas = generated_datas.detach().cpu().numpy()

            for i in range(0, num_to_generate):
                x = generated_datas[i][(options.num_events - 1) * options.feature_dim:]
                x = np.reshape(x, (options.feature_dim, 1))
                if i == 0:
                    plots[idx] = x
                else:
                    plots[idx] = np.concatenate([plots[idx], x], axis=1)
            idx += 1

        # delta(max, mid)
        pic.append(np.abs(plots[2] - plots[1]))
        # delta(mid, min)
        pic.append(np.abs(plots[1] - plots[0]))

    # plot delta maps
    plt.figure(figsize=(20, 15))
    for p in range(len(pic)):
        plt.subplot(3, 4, p + 1)
        plt.imshow(pic[p])

    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.025, 0.8])
    plt.colorbar(cax=cax)
    plt.show()


if __name__ == '__main__':
    args = setup_args()
    options = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu_device
    running(options)
