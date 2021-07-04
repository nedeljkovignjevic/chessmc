import torch
import torch.nn as nn
import torch.utils.data

from torch import optim
from model import Model, ChessTrainDataset, ChessValidationDataset
from utils import stockfish_treshold


class TrainerConfig:
    n_epochs = 150
    batch_size = 128
    save_path = '../models/mlp-stockfish-new.pth'
    

class Trainer:

    def __init__(self, model, train_data, validation_data, config):
        self.model = model
        self.train_data = train_data
        self.validation_data = validation_data
        self.config = config

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = self.model.to(self.device)

    def save_checkpoint(self, epoch, state_dict, optimizer_dict):
        torch.save({
            'epoch': epoch + 1,
            'state_dict': state_dict,
            'optimizer_dict': optimizer_dict
        }, self.config.save_path)


    def train(self):
        model, train_data, validation_data, config, device = self.model, self.train_data, self.validation_data, self.config, self.device

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=config.batch_size, shuffle=True)

        optimizer = optim.Adam(model.parameters())
        loss_fn = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(config.n_epochs):
            cumulative_loss = 0
            n_losses = 0
            for batch_idx, (x, y) in enumerate(train_loader):
                input, target = x.to(device).float(), y.apply_(stockfish_treshold).to(device).long()

                optimizer.zero_grad()
                output = model(input)

                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()

                cumulative_loss += loss.item()
                n_losses += 1

            print(f'epoch: {epoch}, loss: {cumulative_loss / n_losses}')

            cumulative_loss = 0
            n_losses = 0
            model.eval()
            with torch.no_grad():
                for (x, y) in validation_loader:
                    input, target = x.to(device).float(), y.apply_(stockfish_treshold).to(device).long()

                    output = model(input)
                    loss = loss_fn(output, target)
                    cumulative_loss += loss.item()
                    n_losses += 1

            print(f'epoch: {epoch}, validation loss: {cumulative_loss / n_losses}')
            self.save_checkpoint(epoch, model.state_dict(), optimizer.state_dict())
            model.train()


def evaluate(test_data):
    model = Model()
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True)

    checkpoint = torch.load('../models/mlp-stockfish-new.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            input, target = data
            output = model(input)
            for idx, i in enumerate(output):
                if torch.argmax(i) == target[idx]:
                    correct += 1
                total += 1

    print(f'Accuracy: {round(correct / total, 3)}')


if __name__ == '__main__':

    model = Model()
    train_data = ChessTrainDataset()
    validation_data = ChessValidationDataset()
    config = TrainerConfig()

    trainer = Trainer(model, train_data, validation_data, config)
    trainer.train()
