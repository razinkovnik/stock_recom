import pandas as pd
from typing import Dict, Tuple, List
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

from args import TrendArgs
from company import Company
from utils import *


@dataclass
class Stock:
    company: Company
    price = 0.
    delta = 0.
    predicts: List[float] = field(default_factory=list)
    history: List[float] = field(default_factory=list)


class Portfolio:
    stocks: Dict[str, Tuple[Stock, int]] = {}

    def __init__(self, market: List[Stock], cash: float, size: int, max_test_loss: float, args: TrendArgs):
        self.market = market
        self.cash = cash
        self.size = size
        self.max_test_loss = max_test_loss
        self.args = args

    def add(self, stock: Stock, n: int):
        if stock.company.name in self.stocks.keys():
            self.stocks[stock.company.name] = (stock, self.stocks[stock.company.name][1] + n)
        else:
            self.stocks[stock.company.name] = (stock, n)

    def create(self):
        size = self.size - len(self.stocks)
        if size == 0:
            return
        part_cash = self.cash / size
        for stock in self.market:
            if stock.delta < 0:
                break
            stock.company.init_train_test_data()
            loss = stock.company.test()
            if loss < self.max_test_loss:
                n = int(part_cash // stock.price)
                if n > 0:
                    self.add(stock, n)
                    self.cash -= stock.price * n
                    size -= 1
                    if size == 0:
                        break
                    part_cash = self.cash / size

    def show(self):
        for stock, n in self.stocks.values():
            print(stock.company.name)
            x = np.arange(len(stock.history))
            plt.plot(x, stock.history)
            plt.plot(x[args.seq_window - 1:], stock.history[args.seq_window - 1:args.seq_window] + stock.predicts)
            plt.show()

    def value(self) -> float:
        return round(sum([stock.price * n for stock, n in self.stocks.values()]) + self.cash, 2)

    def update(self, days):
        for company in self.stocks.keys():
            stock, n = self.stocks[company]
            self.stocks[company] = predict_test(stock.company, self.args, days), n

    def sell(self):
        for company in list(self.stocks.keys()):
            stock, n = self.stocks[company]
            if stock.delta < 0:
                self.cash += round(stock.price * n, 2)
                print(f"sell {company}: {stock.price}$ x {n} = {round(stock.price * n, 2)}")
                del self.stocks[company]


def load_companies(args: TrendArgs) -> List[Company]:
    args.load_model = True
    args.load_from_yahoo = False
    df = pd.read_csv("dataset/companies.csv")
    companies_name = df['0'].to_list()
    companies = []

    for i, name in enumerate(companies_name):
        try:
            company = Company(name, args)
            company.load_data()
            companies += [company]
        except FileNotFoundError:
            pass
    return companies


# noinspection PyArgumentList
def predict_test(company: Company, args: TrendArgs, days=10) -> Stock:
    stock = Stock(company)
    stock.history = list(company.data.Close[-days - args.seq_window:])
    inputs = np.array(stock.history[:args.seq_window]).reshape([-1, 1])
    scaler = MinMaxScaler(feature_range=(-1, 1))
    inputs = norm_data(inputs, scaler).unsqueeze(0)
    company.model.eval()
    state = None
    for day in range(days):
        with torch.no_grad():
            out, state = company.model(inputs, state)
        inputs = inputs.tolist()[0] + out.tolist()[0]
        inputs = torch.tensor(inputs[1:]).unsqueeze(0)
    predicts = inputs.numpy()[0].reshape(-1, 1)
    stock.predicts = scaler.inverse_transform(predicts.reshape(-1, 1)).flatten().tolist()[-days:]
    stock.price = stock.history[args.seq_window - 1]
    stock.delta = stock.predicts[0] - stock.price
    return stock


def get_market(companies: List[Company], days: int) -> List[Stock]:
    market = [predict_test(company, args, days=days) for company in companies]
    return sorted(market, key=lambda stock: stock.delta / stock.price, reverse=True)


if __name__ == "__main__":
    args = TrendArgs()
    companies = load_companies(args)
    market = get_market(companies, 10)
    portfolio = Portfolio(market, 1000.0, 4, 0.04, args)
    for i in reversed(range(10)):
        portfolio.create()
        portfolio.update(i)
        print(portfolio.value())
        portfolio.sell()
        for company in portfolio.stocks.keys():
            stock, n = portfolio.stocks[company]
            print(f"{company}: {round(stock.price, 2)}$ x {n} = {round(stock.price*n, 2)}")
        print(f"cash: {portfolio.cash}")
        print(f"{i})" + "="*10)


