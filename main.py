# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 2021

@author: hammelrathl, hilbertk
"""
# System  and path configurations
import copy
import sys
import os
import multiprocessing
import time
import pickle
import csv
from pathlib import Path
import json
from importlib import import_module

# Data handling and plotting
import numpy as np
import pandas as pd
import math
from scipy import stats

from preprocessor import main_preprocessing, aggregation_transformation, dataset_split, scaling, onehot
from postprocessor import safe_results
from model import RF, NN, SVM

# Model preparation and selection
from sklearn import preprocessing
from sklearn import set_config
from sklearn.utils import shuffle, resample
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, ParameterSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import RFE, RFECV, SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import log_loss, roc_curve, auc
from sklearn.decomposition import PCA, FastICA

import config
from config import ml_options
from config import STANDARDPATH, DATAPATH_IN, DATAPATH_OUT

if ml_options["model"] == "RF":
    ml_options = config.rf_config(ml_options)
elif ml_options["model"] == "NN":
    ml_options = config.nn_config(ml_options)

def reminder():
    print("Sind die Werte 777, 888, 999 für missings reserviert, und sind alle missings mit diesen Werten kodiert?")
    print("Sind die Standardparameter bzw der Range der Parameter fürs Tuning für das Datenset sinnvoll?")
    input("Press Enter to continue...")
    

if __name__ == '__main__':
    reminder()
    runslist = [i for i in range(ml_options["n_iterations"])]
    
    outcome_list = []
    for numrun in runslist:
        features = onehot.prepare_data(ml_options)
        X_train, X_test, y_train, y_test = dataset_split.prepare_data(ml_options, numrun, features)
        X_train, X_test, y_train, y_test = main_preprocessing.prepare_data(numrun, ml_options, X_train,X_test, y_train,y_test)
        X_train, X_test, y_train, y_test = aggregation_transformation.prepare_data(ml_options,X_train, X_test, y_train, y_test)
        X_train, X_test, y_train, y_test = scaling.prepare_data(numrun, ml_options, X_train,X_test, y_train,y_test)

        clf = import_module(f"{ml_options["model_architecture"]}")

        clf = clf.build_model(ml_options, X_train, X_test, y_train, y_test)
        clf = clf.fit_model(X_train, y_train, clf)
        outcome_results = clf.predict(X_test, y_test, clf, ml_options)
        
        outcome_list.append(outcome_results)
    safe_results.aggregate_metrics(ml_options, outcome_list, X_train)

    if ml_options["baseline"] == 1:
        base = import_module(f"{ml_options["baseline_model"]}")
        outcome_baseline = base.run(ml_options, X_train, X_test, y_train, y_test)
        


    if ml_options["save_config_option"] == 1:
        savepath = os.path.join(STANDARDPATH, f'{ml_options["model_architecture"]}_configurations.csv')
        

    if not os.path.exists(savepath):
        header = [*ml_options.keys()]
        with open(savepath, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(header)
            # write outcome rows
            writer.writerow(list(ml_options.values()))
    else:
        with open(savepath, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(list(ml_options.values()))



    #print("elapsed_time = {}", time.time() - start_time)