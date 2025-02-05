#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :
import os.path

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

__all__ = ['Format1']


class Format1(Dataset):
    """
    the format according to paper
    "Joint Activity Recognition and Indoor Localization with WiFi Fingerprints"
    """

    def __init__(self, dataset_path, mode):
        self.dataset = sio.loadmat(os.path.join(dataset_path, f'{mode}set.mat'))

        self.data_amplitude = torch.from_numpy(self.dataset['data_amplitude']).type(torch.FloatTensor)
        self.data_phase = torch.from_numpy(self.dataset['data_phase']).type(torch.FloatTensor)

        self.label_activity = self.dataset['label_activity']
        self.label_location = self.dataset['label_location']
        self.label = torch.from_numpy(
            np.concatenate((self.label_activity, self.label_location), 1)
        ).type(torch.LongTensor)

    def __len__(self):
        return len(self.data_amplitude)

    def __getitem__(self, idx):
        # 论文中只使用幅值
        data = self.data_amplitude[idx]
        label = self.label[idx]
        return data, label


class Format2(Dataset):
    """
    the format for regression
    """

    def __init__(self, dataset_path, mode):
        self.dataset = sio.loadmat(os.path.join(dataset_path, f'{mode}set.mat'))

        self.data_amplitude = torch.from_numpy(self.dataset['data_amplitude']).type(torch.FloatTensor)
        self.data_phase = torch.from_numpy(self.dataset['data_phase']).type(torch.FloatTensor)

        self.label_activity = self.dataset['label_activity']
        self.label_location = self.dataset['label_location']
        self.label = torch.from_numpy(
            np.concatenate((self.label_activity, self.label_location), 1)
        ).type(torch.LongTensor)

    def __len__(self):
        return len(self.data_amplitude)

    def __getitem__(self, idx):
        # 论文中只使用幅值
        data = self.data_amplitude[idx]
        label = self.label[idx]
        return data, label


if __name__ == '__main__':
    format_1 = Format1('../dataset', 'train')
    for data, label in format_1:
        print(data)
        print(label)
