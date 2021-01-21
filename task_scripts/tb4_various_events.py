import sys
sys.path.append('..')
import numpy as np
from MainClasses.SBetaVAE import SBetaVAE
from MainClasses.TorchDataset import TorchDataset
from MainClasses.DataHandler import DataHandler
from MainClasses.CustmerLosses import DisentLoss
from MainClasses.CustmerMetrics import segment_metrics, r_square, binary_accurate
import pandas as pd
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch import nn
import torch


def setup_args():
    """
    Parameters definition
    :return:
    """
    parser = ArgumentParser(description='Table 4: F1 and ER comparison with different '
                                        'number of events on the Freesound dataset.')
    # NOTE: parameters below are not supposed to change.
    DATASET = 'freesound'
    TIMESTEP = 5
    NAME = 'att_s_beta_vae'
    NFOLDS = 4
    parser.add_argument('-name', "--name", type=str, default=NAME)
    parser.add_argument('-nfolds', "--nfolds", type=int, default=NFOLDS)
    parser.add_argument("-dt", "--dataset", type=str, default=DATASET)
    parser.add_argument('-t', "--time_step", type=int, default=TIMESTEP)
    parser.add_argument('-o', "--result_path", type=str, default='../aed_data/' + DATASET + '/result/')
    parser.add_argument('-d', "--feature_dim", type=int, default=216)
    parser.add_argument('-rh', "--return_h", type=bool, default=False)

    # Parameters below are changeable.
    parser.add_argument('-b', "--batch_size", type=int, default=128)
    parser.add_argument('-e', "--epoch", type=int, default=50)
    parser.add_argument('-lr', "--learning_rate", type=float, default=0.0002)
    parser.add_argument('-k', "--num_events", type=int, default=5)  # alter it to change the number of events.
    parser.add_argument('-gpu', "--gpu_device", type=str, default='0')
    parser.add_argument('-m', "--mix_data", type=int, default=0)  # if generate new sub-dataset using freesound
    return parser


def running(options):
    dh = DataHandler('freesound')
    folds = options.nfolds
    batch_size = options.batch_size
    epoch = options.epoch
    lr = options.learning_rate
    K = options.num_events
    if options.mix_data:
        dh.mix_data(K)
    if K == 5:
        beta = 3
        lambda_ = 1
        latents_dim = 15
    elif K == 10:
        beta = 4
        lambda_ = 2
        latents_dim = 30
    elif K == 15:
        beta = 5
        lambda_ = 3
        latents_dim = 45
    elif K == 20:
        beta = 5
        lambda_ = 3
        latents_dim = 60
    else:
        return
    beta = beta * latents_dim / options.feature_dim
    lambda_ = lambda_ * options.num_events / latents_dim

    decoder_loss_fn = nn.BCELoss(reduction='none')
    disent_loss = DisentLoss(K=options.num_events, beta=beta)

    f1_list, er_list, fold_list = [], [], []
    for k in range(1, folds + 1):
        sb_vae = SBetaVAE(options).cuda()
        sb_vae.name = 'att_s_beta_vae_' + str(K)
        optimizer = torch.optim.Adam(sb_vae.parameters(), lr=lr)
        # 2.Load data.
        train_dataset = TorchDataset(options, type_='train', fold=k)
        train_loader = DataLoader(train_dataset, pin_memory=True,
                                  batch_size=batch_size, num_workers=0,
                                  shuffle=True)

        val_dataset = TorchDataset(options, type_='val', fold=k)
        val_loader = DataLoader(val_dataset, pin_memory=True,
                                batch_size=batch_size, num_workers=8,
                                shuffle=True)

        test_dataset = TorchDataset(options, type_='test', fold=k)
        test_loader = DataLoader(test_dataset, pin_memory=True,
                                 batch_size=batch_size, num_workers=4,
                                 shuffle=True)

        best_f1, best_er, best_e = 0, 0, 0
        min_loss = 1000
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

            if epoch == 0:
                print('Testing ONLY...')
            else:
                val_loss, f1_score, error_rate = validation(options, sb_vae, val_loader)
                if val_loss < min_loss:
                    print('Validation loss decreased from {} to {} at epoch {}.'.format(min_loss, val_loss, str(e)))
                    min_loss = val_loss
                    best_e = e
                    torch.save(sb_vae.state_dict(),
                               options.result_path + sb_vae.name + '/fold_' + str(k) + '_cp_weight.h5')
                else:
                    print('Validation loss dose not decreased than {}.'.format(min_loss))

        if not epoch == 0:
            print('Fold ' + str(k) + 'trained done, best validation results is: loss={}, e={}'.format(
                min_loss, best_e
            ))

        f1_test, er_test = test(options, model=sb_vae, test_loader=test_loader, fold=k)
        f1_list.append(f1_test)
        er_list.append(er_test)
        fold_list.append(k)
        torch.save(sb_vae.state_dict(), options.result_path + sb_vae.name + '/fold_' + str(k) + '_last_weight.h5')

    f1_list.append(np.mean(f1_list))
    er_list.append(np.mean(er_list))
    fold_list.append('AVR')
    result_df = pd.DataFrame({'F1': f1_list, 'ER': er_list}, index=fold_list)
    print(result_df)
    result_df.to_csv(options.result_path + options.name + '_' + str(options.num_events) + '/K_Folds_results.csv')


def validation(options, model, val_loader):
    """
    This function is defined to implement the cross validation stage with 4-Folds.
    :param options: the optional parameters (object of the ArgumentParser).
    :param model: the supervised-beta-VAE model, gain by the training stage.
    :param val_loader: the validation data loader.
    :return: the loss, f1 score and ER from the validation dataset.
    """
    K = options.num_events
    if K == 5:
        beta = 3
        lambda_ = 1
    elif K == 10:
        beta = 4
        lambda_ = 2
    elif K == 15:
        beta = 5
        lambda_ = 3
    elif K == 20:
        beta = 5
        lambda_ = 3
    else:
        return
    beta = beta * options.latents_dim / options.feature_dim
    lambda_ = lambda_ * options.num_events / options.latents_dim

    decoder_loss_fn = nn.BCELoss(reduction='none')
    disent_loss = DisentLoss(K=options.num_events, beta=beta)
    l_decoders = 0
    l_disents = 0
    loss_datas = 0
    n_batches = len(val_loader)
    with torch.no_grad():
        model.eval()
        detector_outs, y_datas = torch.Tensor(), torch.Tensor()
        r2s = 0
        for n_sample, (x_data, y_data) in enumerate(val_loader):
            dec_out, detectors_out, z_stars, alphas, (mu, log_var) = model(x_data.float().cuda())
            l_decoder = torch.mean(decoder_loss_fn(dec_out.reshape(len(x_data), -1), x_data.float().cuda()))
            l_disent, (l_detector, _) = disent_loss((detectors_out, z_stars, alphas, (mu, log_var)),
                                                    y_data.float().cuda())
            loss = (l_decoder + lambda_ * l_disent).mean()
            l_decoders += l_decoder.mean().cpu().item()
            l_disents += l_disent.mean().cpu().item()
            loss_datas += loss.mean().cpu().item()
            detector_outs = torch.cat((detector_outs, detectors_out.cpu()))
            y_datas = torch.cat((y_datas, y_data.float().cpu()))
            r2s += r_square(dec_out.cpu().numpy(), x_data.float().numpy())

        detector_outs, y_datas = detector_outs.numpy(), y_datas.numpy()
        f1_score, error_rate = segment_metrics(inputs=detector_outs,
                                               targets=y_datas, K=options.num_events)
        binary_acc = binary_accurate(inputs=detector_outs.round(),
                                     targets=y_datas)
        loss_log = {'loss': loss_datas / n_batches, 'loss recons': l_decoders / n_batches,
                    'loss disents': l_disents / n_batches, 'f1 score': f1_score,
                    'error rate': error_rate, 'r2': r2s / n_batches}
        loss_df = pd.DataFrame(data=loss_log, index=['validation'])
        print('******************************')
        print(loss_df)
        print('binary accurate', binary_acc)
        print('******************************')
        return loss_datas / n_batches, f1_score, error_rate


def test(options, model, test_loader, fold):
    """
    This function defined the testing stage, which using the test dataset of the DCASE2017 dataset(Table 3 in paper) to
    evaluate our model
    :param options: the same one with the other function.
    :param model: the selected model in the validation stage.
    :param test_loader: the test dataset loader.
    :param fold: the K-th fold.
    :return: f1 score and ER of the test dataset.
    """
    model.load_state_dict(torch.load(options.result_path + model.name + '/fold_' + str(fold) + '_cp_weight.h5'))
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
        return f1_score, error_rate


if __name__ == '__main__':
    args = setup_args()
    options = args.parse_args()
    running(options)
