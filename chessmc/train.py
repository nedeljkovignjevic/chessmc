import torch.nn as nn
import torch.utils.data

from model import Model, ChessDataset
from torch import optim, Tensor

if __name__ == '__main__':
    dataset = ChessDataset()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    model = Model()
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    n_epochs = 100

    model.train()
    for epoch in range(n_epochs):
        cumulative_loss = 0
        n_losses = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            input, target = x.float(), y.unsqueeze(-1).float()

            optimizer.zero_grad()
            output = model(input)

            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            cumulative_loss += loss.item()
            n_losses += 1

        print(f'epoch: {epoch}, loss: {cumulative_loss / n_losses}')
