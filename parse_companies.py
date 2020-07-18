from urllib.request import urlopen
from lxml import html
import pandas as pd

with urlopen("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies") as fp:
    code = fp.read().decode("utf8")

parsed_body = html.fromstring(code)
table = parsed_body.xpath('//*[@id="constituents"]/tbody')[0]

companies = []
for row in iter(table):
    col = next(iter(row))
    companies += [col.xpath('a')[0].text]
companies = companies[1:]

df = pd.DataFrame(companies)
df.to_csv("dataset/companies.csv")
