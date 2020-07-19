import json
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from typing import List, Dict
from bert_args import TrainingArguments
from bert_utils import *
from torch.utils.data import DataLoader
import pandas as pd


def transform_dataset(args: TrainingArguments):
    df = pd.read_csv(args.dataset, encoding="latin")
    labels = df.iloc[:, 0].to_list()
    labels = [0 if label == 0 else 1 for label in labels]
    texts = df.iloc[:, 5].to_list()
    positive_data = [(text, label) for text, label in zip(texts, labels) if label == 0]
    negative_data = [(text, label) for text, label in zip(texts, labels) if label == 1]
    train_data = positive_data[:-1000] + negative_data[:-1000]
    test_data = positive_data[-1000:] + negative_data[-1000:]
    test_data = pd.DataFrame(test_data, columns=['text', 'label'])
    train_data = pd.DataFrame(train_data, columns=['text', 'label'])
    test_data.to_csv(args.test_dataset, index=False)
    train_data.to_csv(args.train_dataset, index=False)


def collate(data: List[Tuple[str, int]], tokenizer: BertTokenizer, block_size: int) -> Dict:
    texts, labels = list(map(list, zip(*data)))
    input_data = tokenizer.batch_encode_plus(texts, max_length=block_size,
                                             truncation=True, pad_to_max_length=True, return_tensors="pt").to(
        args.device)
    input_data['labels'] = torch.tensor(labels).to(args.device)
    return input_data


def load_data(filename: str, tokenizer: BertTokenizer, batch_size: int, args: TrainingArguments) -> DataLoader:
    data = pd.read_csv(filename)
    data = list(zip(data.text, data.label))
    return build_data_iterator(data, batch_size, lambda data: collate(data, tokenizer, args.block_size))


class TwitterSentimentModel:
    def __init__(self, args: TrainingArguments):
        self.model = BertForSequenceClassification.from_pretrained(args.output_dir)
        self.tokenizer = BertTokenizer.from_pretrained(args.model_name)
        self.model.eval()
        self.device = args.device
        self.model.to(self.device)

    def analyze(self, tweets: List[str]):
        inputs = self.tokenizer.batch_encode_plus(tweets, return_tensors="pt", max_length=256, truncation=True,
                                                  pad_to_max_length=True).to(self.device)
        with torch.no_grad():
            out = torch.softmax(self.model(**inputs)[0], dim=1)
        return out[:, 1].cpu().numpy()


if __name__ == "__main__":
    args = setup()
    tokenizer, model, optimizer = init_model(BertForSequenceClassification, args)
    train(tokenizer, model, optimizer, load_data, args)
