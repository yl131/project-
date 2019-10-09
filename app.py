import requests
import os
import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('/lib/models/wine_estimator.pkl','rb'))
    
@app.route('/', methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def predict():
    
    query = {}
    query['description'] = request.form['description']
    query['price'] = request.form['price']
    query['province'] = request.form['province']
    query['variety'] = request.form['variety']
    query['winery'] = request.form['winery']
    
    query_df = pd.DataFrame.from_dict(query)
    prediction = model.predict(query_df)
     
    return render_template('prediction.html', prediction = prediction)


if __name__ == '__main__':
    app.run(port=33507)
