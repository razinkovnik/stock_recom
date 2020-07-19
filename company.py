import datetime as dt
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import yfinance as yf

from args import TrendArgs
from trend_utils import *
from time_series_predictor import TimeSeriesPredictor
from sklearn.preprocessing import MinMaxScaler


class Company:
    scaler: MinMaxScaler
    train_data: torch.Tensor
    test_data: torch.Tensor
    train_sequences: List[Tuple[torch.Tensor, torch.Tensor]]
    test_sequences: List[Tuple[torch.Tensor, torch.Tensor]]
    inputs: torch.Tensor
    targets: torch.Tensor
    data: pd.DataFrame
    model: TimeSeriesPredictor

    def __init__(self, name: str, args: TrendArgs):
        self.args = args
        self.name = name
        self.model = TimeSeriesPredictor(n_hidden=args.n_hidden, window=args.seq_window, n_layers=args.n_layers,
                                         dropout=args.dropout)
        self.model_path = f"{args.model_path}/{name}.pt"
        if args.load_model:
            self.load_model()

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_data_from_yahoo(self) -> pd.DataFrame:
        actual_date = dt.date.today()
        past_date = actual_date - dt.timedelta(days=self.args.days)
        actual_date = actual_date.strftime("%Y-%m-%d")
        past_date = past_date.strftime("%Y-%m-%d")
        data = yf.download(self.name, start=past_date, end=actual_date)
        return pd.DataFrame(data=data)

    def load_data(self):
        data_path = f"{self.args.stock_dataset}/{self.name}.csv"
        if self.args.load_from_yahoo:
            data = self.load_data_from_yahoo()
            data.to_csv(data_path)
        else:
            data = pd.read_csv(data_path)
        self.data = data

    def init_train_test_data(self):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        df = self.data
        data = np.array(df.Close.tolist())
        n = math.ceil(data.shape[0] * self.args.split_n)
        train_data, test_data = data[:n], data[n:]
        self.train_data = norm_data(train_data, self.scaler)
        self.test_data = norm_data(test_data, self.scaler)
        self.train_sequences = create_inout_sequences(self.train_data, self.args.seq_window)
        self.test_sequences = create_inout_sequences(self.test_data, self.args.seq_window)
        inputs = []
        targets = []
        for seq, target in self.train_sequences:
            inputs += [seq]
            targets += target
        self.inputs = torch.stack(inputs)
        self.targets = torch.stack(targets)

    def test(self) -> float:
        self.model.eval()
        self.model.cpu()
        total_loss = 0
        state = None
        for inputs, target in self.test_sequences:
            with torch.no_grad():
                out, state, loss = self.model(inputs.unsqueeze(0), state, target=target)
            total_loss += loss.item()
        return total_loss / len(self.test_data)

    def predict(self, history: np.ndarray, n=1) -> np.ndarray:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        x = scaler.fit_transform(history.reshape(-1, 1)).reshape(1, -1)
        # noinspection PyArgumentList
        x = torch.FloatTensor(x)
        state = None
        for i in range(n):
            with torch.no_grad():
                y, state = self.model(x, state)
            x = torch.cat((x.view((-1, 1)), y)).view(1, -1)
            x = x[:, 1:]
        return scaler.inverse_transform(x.reshape(-1, 1).numpy())[-n:].reshape(-1)
