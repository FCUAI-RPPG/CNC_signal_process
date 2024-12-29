import gc
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt
from sklearn.utils import shuffle
import tensorflow as tf
from tqdm import tqdm
from config import cfg


def frquency_binned_statistic(s):
    N = len(s)
    T = 1.0 / 10240.0
    yf = np.abs(rfft(s, norm='forward'))
    xf = rfftfreq(N, T)
    cat = pd.cut(xf, np.arange(-1,5121,1), labels=np.arange(0,5121))
    return pd.DataFrame({'yf':yf, 'cat':cat}).groupby('cat')['yf'].sum().values


def MEGenerator2TFDataset(generator):
    inputs = (defaultdict(list), defaultdict(list))
    for (signalss, frequenciess, cutss), (labelss) in tqdm(generator):
        inputs[0]['time_domain_signals'].append(signalss)
        inputs[0]['frequency_domain_signals'].append(frequenciess)
        inputs[0]['cut'].append(cutss)
        inputs[1]['regressor_act_2'].append(labelss)

    inputs[0]['time_domain_signals'] = np.concatenate(inputs[0]['time_domain_signals'])
    inputs[0]['frequency_domain_signals'] = np.concatenate(inputs[0]['frequency_domain_signals'])
    inputs[0]['cut'] = np.concatenate(inputs[0]['cut'])
    inputs[1]['regressor_act_2'] = np.concatenate(inputs[1]['regressor_act_2'])
    return tf.data.Dataset.from_tensor_slices(inputs)


def sampling_signal(data, norm_mean, norm_std, sequence_length, workpiece_length, measure_location, training):
    signals_dict = dict()
    sampling_index = list()
    
    # 設定濾波器
    # fs = 10240.0
    # lowcut = 30.0
    # highcut = 1500.0
    # order = 4
    # nyquist = 0.5 * fs
    # low = lowcut / nyquist
    # high = highcut / nyquist
    # b, a = butter(order, [low, high], btype='band')
    
    for filepath, piece_name in tqdm(data[['filepaths', 'name']].values):
        with open(filepath, 'rb') as f:
            signals = pickle.load(f)

        label_indexs = list()
        for idx, signal in enumerate(signals):
            index_1mm = signal.shape[1] / workpiece_length
            if training:
                label_mm = np.arange(measure_location[0],measure_location[-1],0.1)
            else:
                label_mm = np.array(measure_location)
            
            label_index = np.floor(index_1mm * label_mm).astype(np.int32) - 1
            label_index = np.minimum(signal.shape[1] - sequence_length, label_index)
            label_index = np.maximum(sequence_length, label_index)

            # signal = signal[[1, 2, 3, 4, 5, 6, 7, 8, 9]]
            signal = signal[1:]
            # signal = signal[[1,2,3,4,5,6,7,8,10]]

            '''
            0,1: accelemeter
            2,3,4: rpm
            5,6,7: current
            8: label
            '''

            # signal[[0,1]] = filtfilt(b, a, signal[[0,1]], axis=1)
            # signal[2] = signal[2] * (5900 / 4200)
            signal[[0,1,5,6,7]] = (signal[[0,1,5,6,7]] - norm_mean[[0,1,5,6,7]]) / norm_std[[0,1,5,6,7]]
            signal[[2,3,4]] = (((signal[[2,3,4]] - norm_mean[[2,3,4]]) / (norm_std[[2,3,4]] - norm_mean[[2,3,4]])) * 4) - 2
            signals[idx] = signal

            label_indexs.append(label_index)
        
        label_indexs = np.stack(label_indexs, axis=1)
        sampling_index.append(np.concatenate([np.full(label_indexs.shape[0], piece_name)[:, np.newaxis], label_indexs], axis=1))
        signals_dict[piece_name] = {'signals':signals}

    sampling_index = np.concatenate(sampling_index, axis=0)
    return signals_dict, sampling_index


class MachiningErrorGenerator(tf.keras.utils.Sequence):

    def __init__(self, data, batch_size, sequence_length, workpiece_length, measure_location, training=True):
        self.training = training
        self.batch_size = batch_size
        self.sequence_length = sequence_length // 2
        norm_mean = np.load(cfg.DATASETS.NORM_MEAN)[1][[0,2,4,5,6,7,8,9]]
        norm_std = np.load(cfg.DATASETS.NORM_STD)[1][[0,2,4,5,6,7,8,9]] #內0,1,2,3,4,5,6,7 外0,2,4,5,6,7,8,9
        self.signals_dict, self.sampling_index = sampling_signal(
            data=data,
            norm_mean=norm_mean,
            norm_std=norm_std,
            sequence_length=self.sequence_length,
            workpiece_length=workpiece_length,
            measure_location=measure_location,
            training=training
        )

        if self.training:
            self.sampling_index = shuffle(self.sampling_index)

    def __len__(self):
        return int(np.ceil(self.sampling_index.shape[0] / float(self.batch_size)))

    def __getitem__(self, idx):
        gc.collect()

        batch_index = self.sampling_index[idx * self.batch_size: (idx + 1) * self.batch_size]
        item = self.__generator(batch_index)

        return item

    def on_epoch_end(self):
        if self.training:
            self.sampling_index = shuffle(self.sampling_index)

        gc.collect()

    def __generator(self, batch_index):
        '''
        signals shape [cut, signal_source, signal_length]

        signals channel:
        'spindle_front', 'turret',
        'spindle_rpm', 'motor_x_rpm', 'motor_z_rpm',
        'spindle_electric_current', 'motor_x_electric_current', 'motor_z_electric_current',
        'label'
        '''

        signalss = list()
        frequenciess = list()
        cutss = list()
        labelss = list()

        for batch in batch_index:
            
            piece_name = batch[0]
            signals = self.signals_dict[piece_name]['signals']

            labels = signals[-1][-1, int(batch[-1])]
            signals = np.stack([signal[:-1,int(sequence_idx)-self.sequence_length:int(sequence_idx)+self.sequence_length] for signal, sequence_idx in zip(signals, batch[1:])], axis=0)
            frequencies = np.asarray([[np.abs(rfft(v, norm='forward')) for v in vs] for vs in signals[:,:2]])

            signalss.append(signals)
            frequenciess.append(frequencies)
            cutss.append([0,1])
            labelss.append(labels)

        signalss = np.asarray(signalss, dtype=np.float32).transpose(0,1,3,2)
        frequenciess = np.asarray(frequenciess, dtype=np.float32).transpose(0,1,3,2)
        cutss = np.asarray(cutss)
        labelss = np.asarray(labelss, dtype=np.float32) * 100
        
        return (signalss, frequenciess, cutss), (labelss)