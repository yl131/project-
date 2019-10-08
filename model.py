#Import ML modules
import os
import numpy as np 
import pandas as pd 
from pandas import Series
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
from sklearn import base
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import pickle

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
