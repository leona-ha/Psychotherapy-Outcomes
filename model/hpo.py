from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
 
from sklearn.preprocessing import scale, normalize
from sklearn.model_selection import cross_val_score 
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def get_acc_status(clf,X,y): 
    acc = cross_val_score(clf, X, y, cv=5, scoring='recall').mean() 

    return {'loss': -acc, 'status': STATUS_OK}

def obj_fnc(params) :  
    model = params.get('model').lower() 
    X_ = X[:]
    if 'normalize' in params:
        if params['normalize'] == 1:
            X_ = normalize(X_)
        del params['normalize'] 
    if 'scale' in params:
        if params['scale'] == 1:
            X_ = scale(X_)
        del params['scale'] 

    del params['model'] 
    if model == "rf":
        clf = RandomForestClassifier(**params) 
    
    return(get_acc_status(clf,X_,y))