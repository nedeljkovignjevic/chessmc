import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class ChessDataset(Dataset):
    def __init__(self):
        data = np.load('../data/processed.npz')
        self.inputs = data['arr_0']
        self.outputs = data['arr_1']
        print(f'Data loaded, {self.inputs.shape}, {self.outputs.shape}')

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(768, 1048)
        self.fc2 = nn.Linear(1048, 500)
        self.fc3 = nn.Linear(500, 50)

        self.last = nn.Linear(50, 1)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x))
        x = self.dropout(x)

        x = self.last(x)
        return torch.tanh(x)
