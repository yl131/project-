import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas import Series
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
from sklearn import base
from sklearn.pipeline import FeatureUnion
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge

class ColumnSelectTransformer(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, col_names):
        self.col_names = col_names 
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        final_array = X[self.col_names].values
        
        return final_array
        
class CorpusTransformer(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self):
        self
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        final_array = []
        
        for elem in X:
            
            for i in elem:
            
                final_array.append(i)
        
        return final_array

class DictEncoder(base.BaseEstimator, base.TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        l_dict = []
        
        for lst in X:
            
            for elem in lst:
            
                d = defaultdict(int)
            
                d[elem] += 1
            
            l_dict.append(d)
        
        return l_dict

class EstimatorTransformer(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, estimator):
        self.estimator = estimator
    
    def fit(self, X, y):
        self.estimator.fit(X,y)
        return self

    def transform(self, X):
        arr = np.array(self.estimator.predict(X)).reshape(-1,1)
        return arr

description_est = Pipeline([
    ('transformer', ColumnSelectTransformer(['description'])),
    ('corpus', CorpusTransformer()),
    ('vectorizer', TfidfVectorizer()),
    ('Ridge', RandomForestRegressor())
  ])

price_est = Pipeline([
    ('transformer', ColumnSelectTransformer(['price'])),
  ('ridge', RandomForestRegressor())
  ])

province_est = Pipeline([
    ('transformer', ColumnSelectTransformer(['province'])),
    ('encoder', DictEncoder()),
    ('vectorizer',DictVectorizer()),
  ('ridge', RandomForestRegressor())
  ])

variety_est = Pipeline([
    ('transformer', ColumnSelectTransformer(['variety'])),
    ('encoder', DictEncoder()),
    ('vectorizer',DictVectorizer()),
  ('ridge', RandomForestRegressor())
  ])

winery_est = Pipeline([
    ('transformer', ColumnSelectTransformer(['winery'])),
    ('encoder', DictEncoder()),
    ('vectorizer',DictVectorizer()),
  ('ridge', RandomForestRegressor())
  ])

union = FeatureUnion([
    ('description_model', EstimatorTransformer(description_est)),
    ('price_model', EstimatorTransformer(price_est)),
    ('province_model', EstimatorTransformer(province_est)),
    ('variety_model', EstimatorTransformer(variety_est)),
    ('winery_model', EstimatorTransformer(winery_est))
    ])

wine_estimator = Pipeline([
  ("features", union),
  ("ridge", RandomForestRegressor())
  ])



