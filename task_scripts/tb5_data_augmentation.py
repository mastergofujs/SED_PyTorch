import sys

sys.path.append('..')
import pandas as pd
from MainClasses.SBetaVAE import SBetaVAE
from MainClasses.TorchDataset import TorchDataset
from MainClasses.DataHandler import DataHandler
from MainClasses.CustmerLosses import DisentLoss
from MainClasses.CustmerMetrics import binary_accurate, r_square, segment_metrics
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import torch
from torch import nn
import numpy as np
import os


def setup_args():
    """
    Parameters definition
    :return:
    """
    parser = ArgumentParser(description='Fig. 5: disentanglement visualization')
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
    parser.add_argument('-gpu', "--gpu_device", type=str, default='0')
    parser.add_argument('-m', "--mix_data", type=int, default=0)  # if generate new sub-dataset using freesound
    return parser


def running(options):
    dh = DataHandler(options.dataset)
    # 1.First construct polyphonic datasets by mixing single event sound, and extract MFCCs features.
    if options.mix_data:
        dh.mix_data(nevents=5, isUnbalanced=True)
    batch_size = options.batch_size
    epoch = options.epoch
    lr = options.learning_rate
    K = 5
    beta = 3
    lambda_ = 1
    latents_dim = 15
    sb_vae = SBetaVAE(options).cuda()
    sb_vae.name = 'att_s_beta_vae_5'
    beta = beta * latents_dim / options.feature_dim
    lambda_ = lambda_ * options.num_events / latents_dim
    decoder_loss_fn = nn.BCELoss(reduction='none')
    disent_loss = DisentLoss(K=options.num_events, beta=beta)
    optimizer = torch.optim.Adam(sb_vae.parameters(), lr=lr)
    # 2.Load data.
    train_dataset = TorchDataset(options, type_='train', fold=1)
    train_loader = DataLoader(train_dataset, pin_memory=True,
                              batch_size=batch_size, num_workers=0,
                              shuffle=True)

    test_dataset = TorchDataset(options, type_='test', fold=1)
    test_loader = DataLoader(test_dataset, pin_memory=True,
                             batch_size=batch_size, num_workers=0,
                             shuffle=True)

    # 3.train the model with unbalanced data
    for e in range(epoch):
        sb_vae.train()
        l_decoders = 0
        l_detectors = 0
        l_disents = 0
        loss_datas = 0
        for n_sample, (x_data, y_data) in enumerate(train_loader):
            dec_out, detectors_out, z_stars, alphas, (mu, log_var) = sb_vae(x_data.float().cuda())
            l_decoder = torch.mean(decoder_loss_fn(dec_out.reshape(len(x_data), -1), x_data.float().cuda()))
            l_disent, (l_detector, _) = disent_loss((detectors_out, z_stars, alphas, (mu, log_var)),
                                                    y_data.float().cuda())
            loss = (l_decoder + lambda_ * l_disent)
            l_decoders += l_decoder.cpu().mean().item()
            l_detectors += l_detector.cpu().mean().item()
            l_disents += l_disent.cpu().mean().item()
            loss_datas += loss.cpu().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if n_sample % 500 == 0:
                print(
                    '--- epoch/epochs: %d/%d || batch/batches: %d/%d ---' % (e, epoch, n_sample, len(train_loader)))
                binary_acc = binary_accurate(inputs=detectors_out.squeeze().detach().round().cpu().numpy(),
                                             targets=y_data.numpy().squeeze())

                r2 = r_square(dec_out.detach().cpu().numpy(), x_data.numpy())
                loss_log = {'loss': loss_datas / (n_sample + 1), 'loss recons': l_decoders / (n_sample + 1),
                            'loss disents': l_disents / (n_sample + 1),
                            'loss detectors': l_detectors / (n_sample + 1),
                            'r2': r2}
                loss_df = pd.DataFrame(data=loss_log, index=['train'])
                print(loss_df)
                print('binary accurate', binary_acc)
                print()

    f1_test, er_test, binary_acc = test(options, model=sb_vae, test_loader=test_loader, fold=1)
    print('Before data augmentation: F1={}, ER={}'.format(f1_test, er_test))
    print('ACC: ', binary_acc)

    # 4. Now generate new data
    with torch.no_grad():
        train_loader = DataLoader(train_dataset, pin_memory=True,
                                  batch_size=1, num_workers=0,
                                  shuffle=True)
        x_augmented = torch.Tensor().cuda()
        y_augmented = torch.Tensor().cuda()
        for n_sample, (x_data, y_data) in enumerate(train_loader):
            if not y_data[0, 0] == 0:
                continue
            dec_out, detectors_out, z_stars, alphas, (mu, log_var) = sb_vae(x_data.float().cuda())
            dec_x = sb_vae.decoder(z_stars[:, :, 0])  # we generate the first event default.
            x_augmented = torch.cat([x_augmented, dec_x])
            y_augmented = torch.cat([y_augmented, y_data.cuda().float()])

    # 5. Retrain a new model and evaluate it
    train_dataset.x_data = np.concatenate([train_dataset.x_data, x_augmented.cpu().float().numpy()])
    train_dataset.y_data = np.concatenate([train_dataset.y_data, y_augmented.cpu().float().numpy()])
    sb_vae = SBetaVAE(options)
    for e in range(epoch):
        sb_vae.train()
        l_decoders = 0
        l_detectors = 0
        l_disents = 0
        loss_datas = 0
        train_loader = DataLoader(train_dataset, pin_memory=True,
                                  batch_size=batch_size, num_workers=0,
                                  shuffle=True)
        for n_sample, (x_data, y_data) in enumerate(train_loader):
            dec_out, detectors_out, z_stars, alphas, (mu, log_var) = sb_vae(x_data.float().cuda())
            l_decoder = torch.mean(decoder_loss_fn(dec_out.reshape(len(x_data), -1), x_data.float().cuda()))
            l_disent, (l_detector, _) = disent_loss((detectors_out, z_stars, alphas, (mu, log_var)),
                                                    y_data.float().cuda())
            loss = (l_decoder + lambda_ * l_disent)
            l_decoders += l_decoder.cpu().mean().item()
            l_detectors += l_detector.cpu().mean().item()
            l_disents += l_disent.cpu().mean().item()
            loss_datas += loss.cpu().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if n_sample % 500 == 0:
                print(
                    '--- epoch/epochs: %d/%d || batch/batches: %d/%d ---' % (e, epoch, n_sample, len(train_loader)))
                binary_acc = binary_accurate(inputs=detectors_out.squeeze().detach().round().cpu().numpy(),
                                             targets=y_data.numpy().squeeze())

                r2 = r_square(dec_out.detach().cpu().numpy(), x_data.numpy())
                loss_log = {'loss': loss_datas / (n_sample + 1), 'loss recons': l_decoders / (n_sample + 1),
                            'loss disents': l_disents / (n_sample + 1),
                            'loss detectors': l_detectors / (n_sample + 1),
                            'r2': r2}
                loss_df = pd.DataFrame(data=loss_log, index=['train'])
                print(loss_df)
                print('binary accurate', binary_acc)
                print()
    torch.save(sb_vae.state_dict(), options.result_path + sb_vae.name + '/fold_' + str(1) + '_DA_weight.h5')
    f1_test, er_test, binary_acc = test(options, model=sb_vae, test_loader=test_loader, fold=1, weight='DA')
    print('After data augmentation: F1={}, ER={}'.format(f1_test, er_test))
    print('ACC: ', binary_acc)


def test(options, model, test_loader, fold, weight='cp'):
    """
    This function defined the testing stage, which using the test dataset of the DCASE2017 dataset(Table 3 in paper) to
    evaluate our model.
    :param options: the same one with the other function.
    :param model: the selected model in the validation stage.
    :param test_loader: the test dataset loader.
    :param fold: the K-th fold.
    :return: f1 score and ER of the test dataset.
    """
    model.load_state_dict(torch.load(options.result_path + model.name + '/fold_' + str(fold) + '_' + weight
                                     + '_weight.h5'))
    with torch.no_grad():
        model.eval()
        detector_outs, y_datas = torch.Tensor(), torch.Tensor()
        r2s = 0
        n_batches = len(test_loader)
        for n_sample, (x_data, y_data) in enumerate(test_loader):
            dec_out, detectors_out, z_stars, alphas, (mu, log_var) = model(x_data.float().cuda())
            detector_outs = torch.cat((detector_outs, detectors_out.cpu()))
            y_datas = torch.cat((y_datas, y_data.float().cpu()))
            r2s += r_square(dec_out.cpu().numpy(), x_data.float().numpy())

        detector_outs, y_datas = detector_outs.numpy(), y_datas.numpy()
        f1_score, error_rate = segment_metrics(inputs=detector_outs,
                                               targets=y_datas, K=options.num_events)
        binary_acc = binary_accurate(inputs=detector_outs.round(),
                                     targets=y_datas)
        loss_log = {'fold': fold,
                    'f1 score': f1_score,
                    'error rate': error_rate,
                    'r2': r2s / n_batches}
        loss_df = pd.DataFrame(data=loss_log, index=['test'])
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print(loss_df)
        print('binary accurate', binary_acc)
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        return f1_score, error_rate, binary_acc


if __name__ == '__main__':
    args = setup_args()
    options = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu_device
    running(options)
