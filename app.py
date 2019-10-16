import requests
import os
import pandas as pd
from flask import Flask, render_template, request
import numpy as np
import dill
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

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

app = Flask(__name__)

dill._dill._reverse_typemap['ClassType'] = type

model = dill.load(open('lib/models/wine_estimator.dill','rb'))

@app.route('/', methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    
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
    app.run()
