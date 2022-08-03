# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 2021

@author: hammelrathl, hilbertk
"""
# System  and path configurations

import os
import config
from config import ml_options
from config import STANDARDPATH, OUTCOME_PATH, ROUND_PATH, DATAPATH_IN
from importlib import import_module
import csv
from tqdm import tqdm

# Data handling and plotting
import math
import numpy as np
import pandas as pd
from preprocessor import missing_values, aggregation_transformation, dataset_split, scaling, onehot
from postprocessor import safe_results, compare_models
from model import RF, NN, SVM

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use(matplotlib.get_data_path()+'/stylelib/apa.mplstyle') # selecting the style sheet
#import seaborn as sns
#sns.set_theme(style="darkgrid")
##sns.set(rc={'figure.figsize':(15,8.)})
#sns.set(font_scale=1.5)
#sns.set()

IMG_SAFEPATH = os.path.join(OUTCOME_PATH, "plots")

model = ml_options["model_architecture"]
baseline = ml_options["baseline_model"]

if model == "RF":
    ml_options = config.rf_config(ml_options)
elif model == "NN":
    ml_options = config.nn_config(ml_options)

def reminder():
    print("Sind die Werte 777, 888, 999 für missings reserviert, und sind alle missings mit diesen Werten kodiert?")
    print("Sind die Standardparameter bzw der Range der Parameter fürs Tuning für das Datenset sinnvoll?")
    input("Press Enter to continue...")
    

if __name__ == '__main__':
    reminder()
    runslist = [i for i in range(ml_options["n_iterations"])]

    #features_in = pd.read_csv(DATAPATH_IN, sep=";", low_memory=False)
    #features_in = features_in[ml_options["feature_columns"]]

    outcome_list = []
    baseline_list = []
    
    fig = plt.figure(figsize=(6.4,4.8))

    for numrun in tqdm(runslist, total=ml_options["n_iterations"]):
        features_out = onehot.prepare_data(ml_options)
        X_train, X_test, y_train, y_test = dataset_split.prepare_data(ml_options, numrun, features_out)
        X_train, X_test, y_train, y_test = missing_values.prepare_data(numrun, ml_options, X_train,X_test, y_train,y_test)
        X_train, X_test, y_train, y_test = aggregation_transformation.prepare_data(ml_options,X_train, X_test, y_train, y_test)
        X_train, X_test, y_train, y_test = scaling.prepare_data(numrun, ml_options, X_train,X_test, y_train,y_test)

        clf_mod = import_module(f"model.{model}")

        clf = clf_mod.build_model(ml_options, X_train, X_test, y_train, y_test)
        clf = clf_mod.fit_model(X_train, y_train, clf)
        outcome_results = clf_mod.predict(X_train,X_test, y_test, clf, ml_options)
        
        outcome_list.append(outcome_results)

        
        fpr = outcome_results[9]
        tpr = outcome_results[10]
        roc_auc = outcome_results[12]

        print("fpr:",fpr)
        print("tpr:",tpr)
        print("auc:",roc_auc)

    
        plt.plot(fpr, tpr)
        #plt.plot(fpr, tpr, label="{}, AUC={:.3f}".format(roc_auc))
        features_out = None
        features = None




        if ml_options["baseline"] == 1:
            base = import_module(f"model.{baseline}")
            outcome_baseline = base.run(ml_options, X_train, X_test, y_train, y_test)
            baseline_list.append(outcome_baseline)
        
    plt.plot([0,1], [0,1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate")

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate")

    plt.title('ROC Curve Analysis')#, fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')

    img_safepath = os.path.join(IMG_SAFEPATH, f'{ml_options["model_name"]}_roc_curves.png')
    plt.savefig(img_safepath, dpi=300)
        
        
    model_flatlists = safe_results.aggregate_metrics(ml_options, outcome_list, X_train, X_test)
    if ml_options["baseline"] == 1:
        compare_models.run(ml_options, baseline_list, X_train, X_test, model_flatlists)
        
    if ml_options["save_config_option"] == 1:
        savepath = os.path.join(STANDARDPATH, f'{ml_options["model_architecture"]}_configurations.csv')
        

    if not os.path.exists(savepath):
        header = [*ml_options.keys()]
        with open(savepath, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(list(ml_options.values()))  # write outcome rows

    else:
        with open(savepath, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(list(ml_options.values()))



    #print("elapsed_time = {}", time.time() - start_time)