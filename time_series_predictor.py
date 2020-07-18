import torch
from torch import nn as nn


class TimeSeriesPredictor(nn.Module):
    def __init__(self, n_hidden, window, n_layers: int, dropout: float):
        super(TimeSeriesPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=n_hidden, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(in_features=n_hidden * window, out_features=1)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, batch, state=None, target=None):
        out, state = self.lstm(batch.unsqueeze(2), state)
        out = out.reshape([out.size(0), out.size(1) * out.size(2)])
        out = self.linear(out)
        if target is not None:
            loss = self.loss_fn(out, target.unsqueeze(1))
            return out, state, loss
        return out, state