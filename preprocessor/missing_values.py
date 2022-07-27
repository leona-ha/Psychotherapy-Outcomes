from config import STANDARDPATH,DATAPATH_IN,DATAPATH_OUT, MODEL_PATH
import os
import pandas as pd
import sys
from pathlib import Path
import copy
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pickle


def prepare_data(numrun, ml_options, X_train, X_test, y_train, y_test):

    ml_options['seed'] = numrun

    """
    "Drop missing values in features 
    """
    if ml_options['missing_values_option'] == 0: # Just leave missing values
        X_train_imputed = copy.deepcopy(X_train)
        X_test_imputed = copy.deepcopy(X_test)

    elif ml_options['missing_values_option']  == 1: # Drop missing values
        #X_train = X_train.replace([999, 888,777], np.NaN)
        X_train_imputed = copy.deepcopy(X_train)
        X_train_imputed.dropna(inplace=True)
        y_train = y_train.drop(y_train.index[~y_train.index.isin(X_train_imputed.index)])
        print("The number of dropped rows in X_train is: ", len(X_train) - len(X_train_imputed))
        #X_test = X_test.replace([999, 888,777], np.NaN)
        X_test_imputed = copy.deepcopy(X_train)
        X_test_imputed.dropna(inplace=True)
        y_test = y_test.drop(y_test.index[~y_test.index.isin(X_test_imputed.index)])
        print("The number of dropped rows in X_test is: ", len(X_test) - len(X_test_imputed))

    elif ml_options['missing_values_option']  == 2: # Fill them with mean/median/mode
        imp_arith = SimpleImputer(missing_values=999, strategy='mean', verbose = 10)
        imp_median = SimpleImputer(missing_values=888, strategy='median')
        imp_mode = SimpleImputer(missing_values=777, strategy='most_frequent')
        imp_arith.fit(X_train)
        imp_median.fit(X_train)
        imp_mode.fit(X_train)

        X_train_imputed = imp_arith.transform(X_train)
        X_train_imputed = imp_median.transform(X_train_imputed)
        X_train_imputed = imp_mode.transform(X_train_imputed)

        X_test_imputed = imp_arith.transform(X_test)
        X_test_imputed = imp_median.transform(X_test_imputed)
        X_test_imputed = imp_mode.transform(X_test_imputed)


    return X_train_imputed, X_test_imputed, y_train, y_test