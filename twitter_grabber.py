import os
from typing import Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta

import load_env


def grab(query: str, maxtweets: int, date_interval: Optional[Tuple[str, str]] = None) -> pd.DataFrame:
    temp_file = os.environ.get("TMP_CSV")
    grabber_path = os.environ.get("GRABBER")
    args = f'--querysearch "{query}" --lang en --maxtweets {maxtweets} --output {temp_file}'
    if date_interval:
        d1, d2 = date_interval
        args += f' --since {d1} --until {d2}'
    print(args)
    os.system(f'python {grabber_path} {args}')
    return pd.read_csv(temp_file)


def grab_tweets(query: str, days: int, maxtweets=10) -> pd.DataFrame:
    d1 = datetime.today().date() - timedelta(days-1)
    d2 = d1 + timedelta(1)
    data = grab(query, maxtweets, (d1.isoformat(), d2.isoformat()))
    for i in range(2, days+2):
        print(d1.isoformat(), d2.isoformat())
        d1, d2 = d2, d2 + timedelta(1)
        data = pd.concat([data, grab(query, maxtweets, (d1.isoformat(), d2.isoformat()))], ignore_index=True)
    return data
