from typing import Mapping
import torch.nn as nn
import torch.utils.data

from model import Model, ChessDataset
from torch import optim


class TrainerConfig:
    n_epochs = 100
    batch_size = 128
    save_path = '../models/mlp.pth'
    

class Trainer:

    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.config = config

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self, epoch, state_dict, optimizer_dict):
        torch.save({
            'epoch': epoch + 1,
            'state_dict': state_dict,
            'optimizer_dict': optimizer_dict
        }, self.config.save_path)


    def train(self):
        model, data, config, device = self.model, self.data, self.config, self.device

        train_loader = torch.utils.data.DataLoader(data, batch_size=config.batch_size, shuffle=True)
        optimizer = optim.Adam(model.parameters())
        loss_fn = nn.MSELoss()

        model.train()
        for epoch in range(config.n_epochs):
            cumulative_loss = 0
            n_losses = 0
            for batch_idx, (x, y) in enumerate(train_loader):
                input, target = x.to(device).float(), y.to(device).unsqueeze(-1).float()

                optimizer.zero_grad()
                output = model(input)

                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()

                cumulative_loss += loss.item()
                n_losses += 1

            print(f'epoch: {epoch}, loss: {cumulative_loss / n_losses}')
            self.save_checkpoint(epoch, model.state_dict(), optimizer.state_dict())


if __name__ == '__main__':

    model = Model()
    data = ChessDataset()
    config = TrainerConfig()

    trainer = Trainer(model, data, config)
    trainer.train()