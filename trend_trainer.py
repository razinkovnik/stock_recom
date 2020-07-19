import pandas as pd
from typing import List, Tuple
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR

from args import TrendArgs
from company import Company
from utils import *


class TrendTrainer:
    companies = {}

    def __init__(self, args: TrendArgs):
        self.args = args

    def __getitem__(self, company: str) -> Company:
        return self.companies[company]

    def add_company(self, name: str):
        try:
            company = Company(name, self.args)
            company.load_data()
            company.init_train_test_data()
            self.companies[name] = company
            print(f"{name} -OK")
        except ValueError:
            print(f"{name} -FAIL")

    def train_with_company(self, company: Company):
        model = company.model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        test_loss = 100.
        counter_no_best = 0
        scheduler = MultiStepLR(optimizer, milestones=[32, 64], gamma=0.5)
        for _ in tqdm(range(self.args.num_epochs)):
            model.cuda()
            model.train()
            state = None
            for i in range(company.inputs.size(0) // self.args.batch_size):
                total_loss = 0
                optimizer.zero_grad()
                batch = company.inputs[i * self.args.batch_size: (i + 1) * self.args.batch_size]
                target = company.targets[i * self.args.batch_size: (i + 1) * self.args.batch_size]
                out, state, loss = model(batch.cuda(), state, target=target.cuda())
                state = [s.detach() for s in state]
                loss.backward()
                total_loss += loss.item()
                optimizer.step()
            loss = company.test()
            scheduler.step()
            if test_loss > loss:
                test_loss = loss
                company.save_model()
                counter_no_best = 0
            else:
                counter_no_best += 1
                if counter_no_best == 10:
                    break
        # noinspection PyUnboundLocalVariable
        print(f"train_loss {total_loss / i * self.args.batch_size}")
        print(f"test_loss {test_loss}")
        company.load_model()

    def train(self):
        for i, (company_name, company) in enumerate(self.companies.items()):
            print(f"train with {company_name}")
            self.train_with_company(company)
            print(f"{i + 1} / {len(self.companies)}")
            company.save_model()


if __name__ == "__main__":
    args = TrendArgs()
    trend_trainer = TrendTrainer(args)
    df = pd.read_csv("dataset/companies.csv")
    companies = df.symbol.to_list()
    for i, company in enumerate(companies[280:]):
        trend_trainer.add_company(company)
        print(f"{i} / {len(companies)}")
    if args.do_train:
        trend_trainer.train()
