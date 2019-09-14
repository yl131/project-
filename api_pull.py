import pandas as pd
from pandas import DataFrame
import requests
import simplejson as json
from datetime import datetime, date


def get_data(ticker):
    url = 'https://www.quandl.com/api/v3/datatables/WIKI/PRICES.json'

    params = {'api_key':'pa4L2QVwPtLzfQeoSMq7', 'qopts.columns':'date,close'}
    params['ticker'] = ticker
    
    today = date.today()
    start_date = today.replace(year=today.year if today.month > 1 else today.year - 1, month=today.month - 1 if today.month >1 else 12)
    
    resp = requests.get(url,params)
    json = resp.json()
    datatable = json['datatable']
    data = datatable['data']
    df = DataFrame(data)
    try:
        df = df.set_index(0)
        df.index = pd.to_datetime(df.index)
        df.columns = ['closing']
        return df[start_date:today]
    except Exception as e:
        df = []
        return df
