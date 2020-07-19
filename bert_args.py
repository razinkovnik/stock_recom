from dataclasses import dataclass, field
from torch.utils.tensorboard import SummaryWriter
from typing import Optional


@dataclass
class TrainingArguments:
    output_dir = "models/twitter"
    _writer: Optional[SummaryWriter] = field(default=None)
    train_batch_size = 16
    test_batch_size = 32
    block_size = 128
    learning_rate = 5e-5
    max_grad_norm = 1.0
    save_steps = 500
    device = "cuda"
    model_name = "bert-base-uncased"
    dataset = "dataset/twitter/training.1600000.processed.noemoticon.csv"
    train_dataset = "dataset/twitter/train.csv"
    test_dataset = "dataset/twitter/test.csv"
    num_train_epochs = 1
    load = False

    @property
    def writer(self) -> Optional[SummaryWriter]:
        return self._writer

    @writer.setter
    def writer(self, log_dir: str):
        self._writer = SummaryWriter(log_dir=log_dir)
