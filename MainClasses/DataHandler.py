import librosa as lrs
import numpy as np
import pickle as pkl
from scipy import signal
from matplotlib import pyplot as plt
import os
from pyprind import ProgBar
import pandas as pd
import random
import json

'''
This class defines the operations of datasets.
'''


class DataHandler:
    def __init__(self, dataset):
        self.dataset = dataset
        self.train_path = '../aed_data/' + self.dataset + '/train/'
        self.test_path = '../aed_data/' + self.dataset + '/test/'
        self.frame_width = 0.1
        self.frame_step = 0.02
        self.sr = 16000

    # pre-emphasize before MFCCs extraction
    def __emphasize__(self, s):
        emphasized_s = np.append(s[0], s[1:] - 0.97 * s[:-1])
        return emphasized_s

    # enframe a signal to several frames.
    def __enframe__(self, s):
        sample_width = int(self.frame_width * self.sr)
        step = int(self.frame_step * self.sr)
        slength = len(s)
        nframes = int(np.ceil((1.0 * slength - sample_width + step) / step) + 1)
        pad_length = int((nframes - 1) * step + sample_width)
        zeros = np.zeros((pad_length - slength,))
        pad_signal = np.concatenate((s, zeros))
        indices = np.tile(np.arange(0, sample_width), (nframes, 1)) + \
                  np.tile(np.arange(0, nframes * step, step), (sample_width, 1)).T
        indices = np.array(indices, dtype=np.int32)
        frames = pad_signal[indices]
        return frames, nframes

    # add window function to every single frame
    def __windowing__(self, frames):
        frames_win = frames * signal.windows.hamming(int(self.frame_width * self.sr))
        return frames_win

    # normalize the data to [0,1]
    def normalize(self, data):
        # data = data / np.max(np.abs(data))
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min + 0.000001)
        return data

    def signal_norm(self, signal):
        signal = signal / (np.max(np.abs(signal)) + 1e-10)
        return signal

    # load TUT dataset
    def __load_data__(self, feature='mfccs', fold=0):
        train_datas = []
        test_datas = []
        val_datas = []

        train_labels = []
        test_labels = []
        val_labels = []

        val_split_config = json.load(open('../aed_data/tut_data/train_val_config.json', 'r'))
        fold_config = val_split_config['f' + str(fold)]
        train_datas_list = sorted(fold_config['train'].split(','))
        val_datas_list = sorted(fold_config['val'].split(','))


        feature_path_train = self.train_path + feature + '/data/'
        feature_path_test = self.test_path + feature + '/data/'
        label_path_train = self.train_path + feature + '/label/'
        label_path_test = self.test_path + feature + '/label/'
        test_datas_list = sorted(os.listdir(feature_path_test))

        for train in train_datas_list:
            file = feature + '_' + train + '.wav.pkl'
            data = pkl.load(open(feature_path_train + file, 'rb'))
            train_datas += list(data)

            df = pd.read_csv(label_path_train + 'labels_' + train + '.wav.csv')
            label = df.iloc[:, 1:]
            label = np.array(label)
            train_labels += list(label)

        for val in val_datas_list:
            file = feature + '_' + val + '.wav.pkl'
            data = pkl.load(open(feature_path_train + file, 'rb'))
            val_datas += list(data)

            df = pd.read_csv(label_path_train + 'labels_' + val + '.wav.csv')
            label = df.iloc[:, 1:]
            label = np.array(label)
            val_labels += list(label)

        for test in test_datas_list:
            file = (test.split('.')[0]).split('_')[1]
            data = pkl.load(open(feature_path_test + test, 'rb'))
            test_datas += list(data)

            df = pd.read_csv(label_path_test + 'labels_' + file + '.wav.csv')
            label = df.iloc[:, 1:]
            label = np.array(label)
            test_labels += list(label)
        if feature == 'mfccs':
            train_datas = self.normalize(train_datas)
            val_datas = self.normalize(val_datas)
            test_datas = self.normalize(test_datas)
        else:
            train_datas, val_datas, test_datas = np.array(train_datas), np.array(val_datas), np.array(test_datas)
        train_labels, val_labels, test_labels = np.array(train_labels), np.array(val_labels), np.array(test_labels)
        return train_datas, train_labels, val_datas, val_labels, test_datas, test_labels

    # load Freesound dataset, before call this function, make sure you have construct
    # the mixed data by the function mix_data()
    # Parameters 'nevents' means the number of events
    #            'inUnbalanced' means if load the unbalanced data, which also can be construct by mix_data().
    def __load_data2__(self, nevents, fold, isUnbalanced=False):
        if not isUnbalanced:
            self.data_path = '../aed_data/freesound/mfccs/datas/mfccs_' + str(nevents) + '.pkl'
            self.label_path = '../aed_data/freesound/mfccs/labels/labels_' + str(nevents) + '.pkl'
        else:
            self.data_path = '../aed_data/freesound/mfccs/datas/DA_mfccs_' + str(nevents) + '.pkl'
            self.label_path = '../aed_data/freesound/mfccs/labels/DA_labels_' + str(nevents) + '.pkl'
        with open(self.data_path, 'rb') as f:
            datas = pkl.load(f)

        with open(self.label_path, 'rb') as f:
            labels = pkl.load(f)
        nsamples = len(datas)
        if not isUnbalanced:
            if fold == 1:
                test_datas = datas[:int(nsamples * 0.25)]
                test_labels = labels[:int(nsamples * 0.25)]
                train_datas = datas[int(nsamples * 0.25):]
                train_labels = labels[int(nsamples * 0.25):]

            elif fold == 2:
                test_datas = datas[int(nsamples * 0.25):int(nsamples * 0.5)]
                test_labels = labels[int(nsamples * 0.25):int(nsamples * 0.5)]
                train_datas = datas[:int(nsamples * 0.25)] + datas[int(nsamples * 0.5):]
                train_labels = labels[:int(nsamples * 0.25)] + labels[int(nsamples * 0.5):]

            elif fold == 3:
                test_datas = datas[int(nsamples * 0.5):int(nsamples * 0.75)]
                test_labels = labels[int(nsamples * 0.5):int(nsamples * 0.75)]
                train_datas = datas[:int(nsamples * 0.5)] + datas[int(nsamples * 0.75):]
                train_labels = labels[:int(nsamples * 0.5)] + labels[int(nsamples * 0.75):]

            elif fold == 4:
                test_datas = datas[int(nsamples * 0.75):]
                test_labels = labels[int(nsamples * 0.75):]
                train_datas = datas[:int(nsamples * 0.75)]
                train_labels = labels[:int(nsamples * 0.75)]
            else:
                print('Error fold!')
                return
        else:
            train_datas = datas[:int(nsamples * 0.85)]
            train_labels = labels[:int(nsamples * 0.85)]
            test_datas = datas[int(nsamples * 0.85):]
            test_labels = labels[int(nsamples * 0.85):]

        train = train_datas[0]
        label = train_labels[0]
        for i in range(1, len(train_datas)):
            train = np.concatenate([train, train_datas[i]])
            label = np.concatenate([label, train_labels[i]])
        train = self.normalize(train)
        train_datas = train[:int(len(train) * 0.85)]
        train_labels = label[:int(len(train) * 0.85)]

        val_datas = train[int(len(train) * 0.85):]
        val_labels = label[int(len(train) * 0.85):]

        test = test_datas[0]
        label = test_labels[0]
        for i in range(1, len(test_datas)):
            test = np.concatenate([test, test_datas[i]])
            label = np.concatenate([label, test_labels[i]])
        test_datas = self.normalize(test)
        test_datas = test_datas
        test_labels = label
        return train_datas, train_labels, val_datas, val_labels, test_datas, test_labels

    # this function mixs the sound with single event to mixed sound based on Freesound dataset.
    # choose different 'nevents' can construct 'nsamplers' sound with different number of events.
    # 'isUnbalanced' control whether you want to construct the unbanlanced dataset.
    def mix_data(self, nevents, isUnbalanced=False):
        sr = self.sr
        audio_path = '../aed_data/freesound/audio/'
        audio_names = os.listdir(audio_path)
        # random.seed(33)
        random.shuffle(audio_names)
        if nevents == 5:
            nsamples = 2000
        elif nevents == 10:
            nsamples = 3000
        elif nevents == 15:
            nsamples = 4000
        elif nevents == 20:
            nsamples = 6000
        else:
            nsamples = 0
        audio_names = audio_names[:nevents]
        if isUnbalanced:
            print('The first five events are:', audio_names)
            print('Now the inefficient category is:', audio_names[0])
        datas = []
        labels = []
        count = 0
        pb = ProgBar(nsamples)
        for i in range(nsamples):
            min_len = 1000000
            mixed_data = np.zeros((min_len,))
            class_label = []
            for j in range(nevents):
                if isUnbalanced:
                    if j == 0:  # let first event unbalanced
                        if count >= 25000:
                            tag = 0
                        else:
                            tag = random.randint(0, 1)
                    else:
                        tag = random.randint(0, 1)
                else:
                    tag = random.randint(0, 1)
                audio_files = os.listdir(audio_path + audio_names[j])
                audio_file = random.sample(audio_files, 1)[0]
                wav_data, _ = lrs.load(audio_path + audio_names[j] + '/' + audio_file, sr=sr)
                class_label.append(tag)
                if min_len > len(wav_data):
                    min_len = len(wav_data)
                mixed_data = mixed_data[:min_len] + wav_data[:min_len] * tag
            mixed_data = self.signal_norm(mixed_data)
            noise_files = os.listdir('../aed_data/freesound/noise/')
            noise_file = random.sample(noise_files, 1)[0]
            n_data, _ = lrs.load('../aed_data/freesound/noise/' + noise_file, sr=sr)
            dB = random.sample(list(range(6, 12)), 1)[0]
            if len(n_data) < len(mixed_data):
                mixed_data = mixed_data[:len(n_data)]
            else:
                n_data = n_data[:len(mixed_data)]

            k = self.SNR2K(mixed_data, n_data, dB)

            mixed_data += k*n_data[:len(mixed_data)]
            # lrs.output.write_wav('test.wav', mixed_data, sr=sr)

            s = self.__emphasize__(mixed_data)
            frames, nframe = self.__enframe__(s)
            if class_label[0] == 1:
                count += nframe
            frames = self.__windowing__(frames)
            mfccs_list = []
            labels_list = []
            for frame in frames:
                if np.max(np.abs(frame)) < 0.03:
                    class_label = [0] * nevents
                mfcc = lrs.feature.mfcc(frame, sr=sr, n_fft=256, n_mfcc=24,
                                        n_mels=24, center=False, norm=None)
                mfcc_delt = lrs.feature.delta(mfcc, width=3)
                mfcc_delt2 = lrs.feature.delta(mfcc, order=2, width=3)
                mfccs = np.concatenate([mfcc, mfcc_delt, mfcc_delt2], axis=1)
                mfccs = np.reshape(mfccs, (np.size(mfccs),))
                mfccs_list.append(mfccs)
                labels_list.append(class_label)
            datas.append(self.normalize(np.array(mfccs_list)))
            labels.append(np.array(labels_list))
            pb.update()
        ziped = list(zip(datas, labels))
        random.shuffle(ziped)
        datas[:], labels[:] = zip(*ziped)
        if isUnbalanced:
            with open('../aed_data/freesound/mfccs/datas/DA_mfccs_' + str(nevents) + '.pkl', 'wb') as f:
                pkl.dump(datas, f)
            with open('../aed_data/freesound/mfccs/labels/DA_labels_' + str(nevents) + '.pkl', 'wb') as f:
                pkl.dump(labels, f)

        else:
            with open('../aed_data/freesound/mfccs/datas/mfccs_' + str(nevents) + '.pkl', 'wb') as f:
                pkl.dump(datas, f)
            with open('../aed_data/freesound/mfccs/labels/labels_' + str(nevents) + '.pkl', 'wb') as f:
                pkl.dump(labels, f)
        print()
        print('Mixing audios with {} nevents and {} nsamples DONE.'.format(nevents, nsamples))

    # calculate the weight of noise by SNR
    def SNR2K(self, signal, noise, dB):
        energe_s = np.sum(signal * signal) / len(signal)
        energe_n = np.sum(noise * noise) / len(noise)
        K = np.sqrt(energe_s / energe_n) * (10 ** (-dB / 20))
        return K

    # sequentialize single frame data to sequential data with specific timestep
    def sequentialize_data(self, datas, timestep):
        if timestep <= 1:
            return datas
        newdata = np.empty(np.shape(datas))
        for time in range(timestep):
            if time == 0:
                newdata = datas[time:time - timestep + 1]
            elif time == timestep - 1:
                newdata = np.concatenate((newdata, datas[time:]), axis=1)
            else:
                newdata = np.concatenate((newdata, datas[time:time - timestep + 1]), axis=1)
        return newdata

    # visualize data distribution using t-SNE. Given the output of hidden layers, you can use this function to visualize
    # the event-specific features distribution, mentioned in paper figure 3.
    def visualization(self, datas, name, target_dim=2):
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=target_dim)
        tsne_datas = tsne.fit_transform(datas)
        # tsne_datas = self.normalize(tsne_datas)
        nums_each_ev = int(len(datas) / 6)
        for i in range(6):
            plt.scatter(tsne_datas[nums_each_ev * i:nums_each_ev * (i + 1), 0],
                        tsne_datas[nums_each_ev * i:nums_each_ev * (i + 1), 1],
                        c=plt.cm.Set1((i + 1) / 10.),
                        alpha=0.6, marker='o')

        plt.legend(['brakes squeaking', 'car', 'children',
                    'large vehicle', 'people speaking', 'people walking'], loc='lower right', ncol=2)
        plt.title('Distribution of features learned by {}.'.format(name))
        plt.show()

    # load data according to different datasets.
    def load_data(self, fold, nevents=0, isUnbalanced=False):
        if self.dataset == 'tut_data':
            train_datas, train_labels, val_datas, val_labels, test_datas, test_labels = self.__load_data__(fold=fold)
            return train_datas, train_labels, val_datas, val_labels, test_datas, test_labels

        elif self.dataset == 'freesound':
            train_datas, train_labels, val_datas, val_labels, test_datas, test_labels \
                = self.__load_data2__(nevents, isUnbalanced=isUnbalanced, fold=fold)
            return train_datas, train_labels, val_datas, val_labels, test_datas, test_labels
        else:
            print('No such dataset!')
