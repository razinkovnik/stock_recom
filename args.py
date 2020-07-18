from dataclasses import dataclass


@dataclass
class TrendArgs:
    n_hidden = 512
    n_layers = 2
    dropout = 0.5
    seq_window = 10
    batch_size = 8
    split_n = 0.7
    lr = 0.002
    days = 366
    num_epochs = 128
    model_path = "models/companies"
    stock_dataset = 'dataset/yahoo'
    load_from_yahoo = False
    do_train = True
    load_model = False

