import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class ChessTrainDataset(Dataset):
    def __init__(self):
        data = np.load('../processed/stockfish_processed15M.npz')
        self.inputs = data['arr_0'][150_000:]
        self.outputs = data['arr_1'][150_000:]
        print(f'Data loaded, {self.inputs.shape}, {self.outputs.shape}')

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]


class ChessValidationDataset(Dataset):
    def __init__(self):
        data = np.load('../processed/stockfish_processed15M.npz')
        self.inputs = data['arr_0'][:150_000]
        self.outputs = data['arr_1'][:150_000]
        print(f'Data loaded, {self.inputs.shape}, {self.outputs.shape}')

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 64)

        self.last = nn.Linear(64, 3)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)

        return self.last(x)
