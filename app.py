from flask import Flask, render_template, request, redirect
import requests
from bokeh.plotting import figure
from bokeh.embed import components
import pandas as pd
from pandas.io.json import json_normalize
import datetime
from calendar import monthrange
import os
# from __future__ import print_function

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def plot_price():
    if request.method == 'GET':
        # default plot when loaded
        STOCK = 'GOOG'
        COLUMNS_TO_PLOT = ['adj. close']
    else:
        STOCK = request.form['ticker']
        if STOCK == '':
            STOCK = 'GOOG'
        # checkbox request form results returns [u'string']
        COLUMNS_TO_PLOT = [str(s) for s in request.form.getlist('features')]

    def minus_one_month(current_date):
        '''return YYYY-MM-DD for current_date minus one month.'''
        last_month = datetime.date(current_date.year, current_date.month, 1) - datetime.timedelta(days=1)
        days_last_month = monthrange(last_month.year, last_month.month)[1]
        if current_date.day <= days_last_month:
            return last_month.replace(day=current_date.day).strftime('%Y-%m-%d')
        else:
            return last_month.replace(day=days_last_month).strftime('%Y-%m-%d')

    # get date range
    current_time = datetime.datetime.today()
    end_date = current_time.strftime('%Y-%m-%d')
    start_date = minus_one_month(current_time)

    api_url = 'https://www.quandl.com/api/v3/datasets/WIKI/{}.json'.format(STOCK) + \
              '?start_date={}&end_date={}&api_key={}'.format(start_date, end_date, os.environ.get('API_KEY', ''))
    session = requests.Session()
    session.mount('http://', requests.adapters.HTTPAdapter(max_retries=3))
    raw_data = session.get(api_url)

    data = raw_data.json()
    # handling Quanle error
    if 'quandl_error' in data:
        error_msg = data['quandl_error']['message']
        error_code = data['quandl_error']['code']
        return render_template('quandl_error.html', error_code=error_code, error_msg=error_msg)

    columns = data['dataset']['column_names']
    # Load tabular price data into df_data
    df_data = json_normalize(data['dataset'], 'data')
    # Change all column names to lowercase
    df_data.columns = [name.lower() for name in columns]
    df_data['date'] = pd.to_datetime(df_data['date'])


    plot = figure(title='Stock Ticker: {}'.format(STOCK),
                  x_axis_label='date',
                  x_axis_type='datetime')
    colors = ['#e69f00', '#56b4e9', '#009e73', '#d55e00']

    for i in range(len(COLUMNS_TO_PLOT)):
        COLUMN = COLUMNS_TO_PLOT[i]
        plot.line(df_data['date'], df_data[COLUMN], line_color=colors[i], line_width=3, alpha=0.5, legend=COLUMN)

    plot.legend.location = "top_center"

    script, div = components(plot)
    return render_template('graph.html', script=script, div=div)


#if __name__ == '__main__':
  #app.run(port=33507)

# Binding PORT for Heroku deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
