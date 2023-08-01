# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 2021

@author: hammelrathl, hilbertk
"""
# System  and path configurations
# show for Github
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
from postprocessor import safe_results, compare_models, aggregate_benchmark
from model import RF
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use(matplotlib.get_data_path()+'/stylelib/apa.mplstyle') # selecting the style sheet


IMG_SAFEPATH = os.path.join(OUTCOME_PATH, "plots")

model = ml_options["model_architecture"]
baseline = ml_options["baseline_model"]
baseline_extra = ml_options["baseline_model_extra"]


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
    baseline_list_extra = []

    
    for numrun in tqdm(runslist, total=ml_options["n_iterations"]):
        features_out = onehot.prepare_data(ml_options)
        X_train, X_test, y_train, y_test = dataset_split.prepare_data(ml_options, numrun, features_out)
        X_train, X_test, y_train, y_test = missing_values.prepare_data(numrun, ml_options, X_train,X_test, y_train,y_test)
        X_train, X_test, y_train, y_test = aggregation_transformation.prepare_data(ml_options,X_train, X_test, y_train, y_test)
        X_train, X_test, y_train, y_test = scaling.prepare_data(numrun, ml_options, X_train,X_test, y_train,y_test)

        # Baseline model
        if ml_options["baseline"] == 1:
            base = import_module(f"model.{baseline}")
            outcome_baseline = base.run(ml_options, X_train, X_test, y_train, y_test)
            baseline_list.append(outcome_baseline)
        
        if ml_options["baseline_extra"] == 1:
            base_extra = import_module(f"model.{baseline_extra}")
            outcome_baseline_extra = base_extra.run(ml_options, X_train, X_test, y_train, y_test)
            baseline_list_extra.append(outcome_baseline_extra)
        
            # Drop early change variable for baseline models
        if ml_options["include_early_change"] == 0:
            X_train.drop("phq_early_change", axis=1,inplace=True)
            X_test.drop("phq_early_change", axis=1,inplace=True)
        print(X_train.columns)
        
        # Main model
        clf_mod = import_module(f"model.{model}")

        clf = clf_mod.build_model(ml_options, X_train, X_test, y_train, y_test)
        clf = clf_mod.fit_model(X_train, y_train, clf)
        outcome_results = clf_mod.predict(X_train,X_test, y_test, clf, ml_options)
        
        outcome_list.append(outcome_results)


# Compare to baseline model
        
    model_flatlists = safe_results.aggregate_metrics(ml_options, outcome_list, X_train, X_test)
    if ml_options["baseline"] == 1:
        output_name = ml_options["baseline_model"]
        compare_models.run(ml_options, baseline_list, X_train, X_test, model_flatlists, output_name)
        
    if ml_options["baseline_extra"] == 1:
        output_name = ml_options["baseline_model_extra"]
        compare_models.run(ml_options, baseline_list_extra, X_train, X_test, model_flatlists, output_name)
        baseline_list_agg = aggregate_benchmark.aggregate_metrics(ml_options, baseline_list_extra, X_train, X_test)

# ROC curve

    fpr = model_flatlists[7]
    tpr = model_flatlists[8]
    roc_auc = model_flatlists[6]
    tprs = model_flatlists[9]

    fig = plt.figure(figsize=(6.4,4.8))

    for i in range(len(fpr)):
        if i == 0:
            plt.plot(fpr[i], tpr[i], lw=1, color = 'grey', label = 'Individual Iterations')
        else:
            plt.plot(fpr[i], tpr[i], lw=1, color = 'grey')

    mean_fpr = np.linspace(0, 1, 100)
    tprs[-1][0] = 0.0
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[0] = 0
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(roc_auc)

    if ml_options["baseline_extra"] == 1:
        fpr_ec = baseline_list_agg[1]
        tpr_ec = baseline_list_agg[2]
        tprs_ec = baseline_list_agg[3]
        roc_auc_ec = baseline_list_agg[0]
        tprs_ec[-1][0] = 0.0
        mean_fpr_ec = np.linspace(0, 1, 100)
        mean_tpr_ec = np.mean(tprs_ec, axis=0)
        mean_tpr_ec[0] = 0
        mean_tpr_ec[-1] = 1.0
        mean_auc_ec = auc(mean_fpr_ec, mean_tpr_ec)
        std_auc_ec = np.std(roc_auc_ec)

        
    plt.plot(mean_fpr, mean_tpr, color='k', label=r'Mean ROC RF', lw=2) #plus minus: $\pm$
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='w', label='AUC RF = %0.2f, SD RF = %0.2f' % (mean_auc, std_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='k', label='Chance')
    if ml_options["baseline_extra"] == 1:
        fpr_ec = baseline_list_agg[1]
        tpr_ec = baseline_list_agg[2]
        mean_fpr_ec = np.linspace(0, 1, 100)
        mean_tpr_ec = np.mean(tprs_ec, axis=0)
        mean_tpr_ec[0] = 0
        mean_tpr_ec[-1] = 1.0
        plt.plot(mean_fpr_ec, mean_tpr_ec, color='k', label=r'Mean ROC Benchmark', lw=2) #plus minus: $\pm$
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='w', label='AUC Benchmark = %0.2f, SD Benchmark = %0.2f' % (mean_auc_ec, std_auc_ec))

    plt.legend(prop={'size':9}, loc='lower right')
    plt.xlabel("False Positive Rate")
    plt.xlim([-0.005, 1.005])

    plt.ylabel("True Positive Rate")
    plt.ylim([-0.005, 1.005])

    img_safepath_1 = os.path.join(IMG_SAFEPATH, f'{ml_options["model_name"]}_roc_curves.png')
    img_safepath_2 = os.path.join(IMG_SAFEPATH, f'{ml_options["model_name"]}_roc_curves.eps')


    plt.savefig(img_safepath_1, dpi=300)
    plt.savefig(img_safepath_2, dpi=1000)

# ROC curve benchmark with early change 
    fpr_ec = baseline_list_agg[1]
    tpr_ec = baseline_list_agg[2]
    roc_auc_ec = baseline_list_agg[0]
    tprs_ec = baseline_list_agg[3]
    tprs_ec[-1][0] = 0.0

    fig = plt.figure(figsize=(6.4,4.8))

    for i in range(len(fpr)):
        if i == 0:
            plt.plot(fpr_ec[i], tpr_ec[i], lw=1, color = 'grey', label = 'Individual Iterations')
        else:
            plt.plot(fpr_ec[i], tpr_ec[i], lw=1, color = 'grey')

    mean_fpr_ec = np.linspace(0, 1, 100)
    tprs_ec[-1][0] = 0.0
    mean_tpr_ec = np.mean(tprs_ec, axis=0)
    mean_tpr_ec[0] = 0
    mean_tpr_ec[-1] = 1.0
    mean_auc_ec = auc(mean_fpr_ec, mean_tpr_ec)
    std_auc_ec = np.std(roc_auc_ec)

        
    plt.plot(mean_fpr_ec, mean_tpr_ec, color='k', label=r'Mean ROC Benchmark', lw=2) #plus minus: $\pm$
    plt.plot(mean_fpr, mean_tpr, color='k', label=r'Mean ROC RF Model', lw=2, linestyle='-.') #plus minus: $\pm$

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='w', label='AUC = %0.2f, SD = %0.2f' % (mean_auc_ec, std_auc_ec))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Chance')

    plt.legend(prop={'size':9}, loc='lower right')
    plt.xlabel("False Positive Rate")
    plt.xlim([-0.005, 1.005])

    plt.ylabel("True Positive Rate")
    plt.ylim([-0.005, 1.005])

    img_safepath_A = os.path.join(IMG_SAFEPATH, f'{ml_options["model_name"]}_ec-benchmark_roc_curves.png')
    img_safepath_B = os.path.join(IMG_SAFEPATH, f'{ml_options["model_name"]}_ec-benchmark_roc_curves_ec-benchmark.eps')


    plt.savefig(img_safepath_A, dpi=300)
    plt.savefig(img_safepath_B, dpi=1000)

        
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