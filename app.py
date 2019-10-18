import requests
import os
import pandas as pd
from flask import Flask, render_template, request
import numpy as np
import dill
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
from sklearn import base
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from model import ColumnSelectTransformer, CorpusTransformer, DictEncoder, EstimatorTransformer

app = Flask(__name__)

dill._dill._reverse_typemap['ClassType'] = type

model = dill.load(open('lib/models/wine_estimator.dill','rb'))

def make_plot (df):
    
    g = sns.regplot(x='points', y='price', data=df, 
                    x_jitter=True, fit_reg=False,
                    marker='o', color='teal', scatter_kws={'s':2})
    g.set_title("Points by Price Distribuition", fontsize=20)
    g.set_xlabel("Points", fontsize= 15)
    g.set_ylabel("Price (USD)", fontsize= 15)

    g.annotate('Your wine',
               xy=(wine_estimator.predict(query_df).item(),query['price'][0]), 
               xytext=(92+0.5, 1800),fontsize= 13,
               arrowprops=dict(arrowstyle="->",
               connectionstyle="angle3,angleA=0,angleB=-90"));
    image = plt.show()
    return image

@app.route('/', methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    
    query = {}
    query['description'] = [request.form['description']]
    query['price'] = [request.form['price']]
    query['province'] = [request.form['province']]
    query['variety'] = [request.form['variety']]
    query['winery'] = [request.form['winery']]
    
    query_df = pd.DataFrame.from_dict(query, orient = 'columns')
    prediction = model.predict(query_df).item()
    
    image = make_plot(query)
    
    return render_template('prediction.html', prediction = prediction, image = image)


if __name__ == '__main__':
    app.run()
