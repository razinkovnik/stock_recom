from typing import List, Tuple
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch


def split_data(data: List[float], window=10) -> List[Tuple[List[float], float]]:
    dataset = []
    for i in range(len(data) - window):
        values = data[i: i + window + 1]
        dataset += [(values[:-1], values[-1])]
    return dataset


def norm_data(data: np.ndarray, scaler: MinMaxScaler) -> torch.Tensor:
    data = np.array(data)
    data_normalized = scaler.fit_transform(data.reshape(-1, 1))
    # noinspection PyArgumentList
    return torch.FloatTensor(data_normalized).view(-1)


def create_inout_sequences(input_data: torch.Tensor, tw: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    inout_seq = []
    length = len(input_data)
    for i in range(length - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq
