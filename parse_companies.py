from urllib.request import urlopen
from lxml import html
import pandas as pd

with urlopen("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies") as fp:
    code = fp.read().decode("utf8")

parsed_body = html.fromstring(code)
table = parsed_body.xpath('//*[@id="constituents"]/tbody')[0]

symbols, names = [], []
for i, row in enumerate(iter(table)):
    if i == 0:
        continue
    row = iter(row)
    col = next(row)
    symbols += [col.xpath('a')[0].text]
    col = next(row)
    names += [col.xpath('a')[0].text]

df = pd.DataFrame({'symbol': symbols, 'name': names})
df.to_csv("dataset/companies.csv")
