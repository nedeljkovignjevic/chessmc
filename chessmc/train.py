import torch.nn as nn
import torch.utils.data

from model import Model, ChessDataset
from torch import optim


if __name__ == '__main__':

    device = 'cuda'
    
    dataset = ChessDataset()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    model = Model()
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    n_epochs = 100

    if device == 'cuda':
        model.cuda()

    model.train()
    for epoch in range(n_epochs):
        cumulative_loss = 0
        n_losses = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            input, target = x.to(device).float(), y.unsqueeze(-1).to(device).float()

            optimizer.zero_grad()
            output = model(input)

            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            cumulative_loss += loss.item()
            n_losses += 1

        print(f'epoch: {epoch}, loss: {cumulative_loss / n_losses}')
        torch.save(model.state_dict(), '../models/mlp.pth')