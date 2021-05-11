# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 12:21:54 2020

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
import tkinter as tk
from tkinter import filedialog

# Data handling and plotting
import numpy as np
import pandas as pd
import math
from scipy import stats

# Model preparation and selection
from sklearn import preprocessing
from sklearn import set_config
from sklearn.utils import shuffle, resample
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, ParameterSampler
from sklearn.impute import IterativeImputer, SimpleImputer,
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import RFE, RFECV, SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import log_loss, roc_curve, auc
from sklearn.decomposition import PCA, FastICA

from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE

start_time = time.time()
timestr = time.strftime("%Y%m%d-%H%M%S")
standardpath = os.environ.get("PSY_PATH")
root = tk.Tk()
root.withdraw()


""" Create dictionary with parameters and options """

ml_options = {}
ml_options["model_name"] = "RF_" + timestr
ml_options["n_iterations"] = int(input("Choose number of iterations:"))
ml_options["feature_data"] = filedialog.askopenfilename()
ml_options["label_data"] =  filedialog.askopenfilename()


"Overall Options"

ml_options['balanced_split_option'] = int(input("""Choose an option for balanced train/test splits:
                                    \n- 0: no balancing
                                    \n- 1: sample balancing for train only
                                    \n- 2: sample balancing for train and test
                                    \n- 3: balanced class_weights
                                    \n- 4: balanced random forest estimator
                                    \n- 5: class weighting with custom dict
                                    """))

if ml_options["balanced_split_option"] == 0 or 3 or 4 or 5:
    ml_options['stratify_option'] = int(input("Would you like to stratify? (0 = no, 1=yes):"))

if ml_options['balanced_split_option'] == 5:
    ml_options['rf_classes_class_weight'] = dict({1:1, 0:20})

ml_options['test_size_option'] = float(input("Set proportion of data points in test set (e.g. 0.33):"))
ml_options['outcome_option'] = input("What is the name of your outcome column?")

"Data preprocessing"

ml_options['missing_values_option'] = int(input("""Chose how to deal with NA:
                                    \n- 0: no Imputation/ just leave them
                                    \n- 1: delete rows with missing values
                                    \n- 2: replace NA with mode/median/most most frequent
                                    \n- 3: MICE Imputation"""))

ml_options['data_scaling_option'] = int(input("""Would you like to scale your data?:
                                    \n- 0: Scaling
                                    \n- 1: Centering and Standardisation"""

ml_options['feature_selection_option'] = int(input("""Choose feature selection strategy:
                                    \n- 0: no feature selection
                                    \n- 1: remove features with low variance (< 80%)
                                    \n- 3: recursive
                                    \n- 4: recursive cross validated
                                    \n- 5: Elastic-Net (only with standardized data)
                                """))
if ml_options['feature_selection_option'] == 3 or 4:
    ml_options['number_features_recursive'] = int(input("Choose max. number of features (default is 10):"))
    ml_options['step_reduction_recursive'] = int(input("Choose n of features to remove per step (default is 3):"))
    ml_options['scoring_recursive'] = 'balanced_accuracy'

if ml_options['feature_selection_option'] == 5:
    ml_options['treshold_option'] = 'mean'


ml_options['hyperparameter_tuning_option'] = int(input("""Choose hyperparameter tuning option:
                                    \n- 0: no hyperparameter tuning
                                    \n- 1: hyperparameter tuning per scikit-learn RandomizeSearch
                                    \n- 2: hyperparameter tuning per oob error"""))

ml_options['hyperparameter_dict'] = {'n_estimators': [500, 2000, 10000],
                                    'criterion_hyper': ['gini', 'entropy',
                                    'max_features_hyper': ['sqrt', 'log2'],
                                    'max_depth': [2, 3, 4, 5],
                                    'min_samples_split_hyper': [2,4,6,8,10],
                                    'min_samples_leaf_hyper': [1, 2, 3, 4, 5],
                                    'bootstrap_hyper': [True]} # [True, False] verfügbar, aber OOB-Estimate nur für True verfügbar
ml_options['n_iter_hyper_randsearch'] = 100 # Anzahl Durchgänge mit zufälligen Hyperparameter - Kombinationen; so hoch wie möglich
ml_options['cvs_hyper_randsearch'] = 5 # default-cvs bei Hyperparameter - Kombinationen; Höhere Anzahl weniger Overfitting


ml_options['permutation_option'] = int(input("Would you like a permutation test? (0 = no, 1=yes):"))
if ml_options['permutation_option'] == 1:
    ml_options['n_permutations_option'] = int(input("Choose number of permutations (default is 5000):"))
ml_options['save_clf_option'] = int(input("Would you like to safe the classifier? (0 = no, 1=yes):"))


def prepare_data(numrun):

    global standardpath, ml_options

    random_state_option = numrun


    """
    "Import Data und Labels
    """

    
    features_import = pd.read_csv(ml_options["feature_data"], sep=";", low_memory=False)
    labels_import = pd.read_csv(ml_options["label_data"], sep=";", low_memory=False)

    n_classes = len(labels_import.unique())
    if n_classes > 2:
        response = input("You have more than 2 outcome classes. Have you checked/adjusted the skript? (y/n)")
        if response == "y":
            pass
        else:
            sys.exit("Please check or adjust the skript for more than 2 classes")

    """
    "Zuweisung features / labels
    "Split train / test sets
    """

    if ml_options['sampling'] == 0 or 3 or 4 or 5:
        if ml_options['stratify_option'] == 0:
            strat_option = None
        else:
            strat_option = labels_import
        X_train, X_test, y_train, y_test = train_test_split(
                features_import, labels_import, stratify=strat_option,
                test_size=ml_options['test_size_option'], random_state=random_state_option)

    elif ml_options['sampling'] == 1:

        X_train, X_test, y_train, y_test = train_test_split(
                    features_import, labels_import, stratify=strat_option, test_size=ml_options['test_size_option'], random_state=random_state_option)
        min_number = min(y_train.value_counts())
        sample_for_balancing = pd.concat([X_train, y_train], axis=1, sort=False)
        reduced_sample = sample_for_balancing.groupby(sample_for_balancing[ml_options['outcome_option']],
                sort=False).apply(lambda frame: frame.sample(int(min_number),random_state=random_state_option))
        mask = reduced_sample.index.get_level_values(-1)
        X_train = X_train.loc[mask]


    elif ml_options['sampling'] == 2:
        min_number = min(labels_import.value_counts())
        sample_for_balancing = pd.concat([features_import, labels_import], axis=1, sort=False)
        reduced_sample = sample_for_balancing.groupby(sample_for_balancing[ml_options['outcome_option']],
                    sort=False).apply(lambda frame: frame.sample(min_number, random_state=random_state_option))
        mask = reduced_sample.index.get_level_values(-1)

        X_reduced_sample = features_import.loc[mask]
        y_reduced_sample = labels_import.loc[mask]
        X_train, X_test, y_train, y_test = train_test_split(
            X_reduced_sample, y_reduced_sample, stratify=y_reduced_sample, test_size=ml_options['test_size_option'], random_state=random_state_option)


### Change to np.array for random forest models ###

    y_train= np.array(y_train)
    y_test= np.array(y_test)
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    if ml_options['balanced_split_option'] == 6 or 7:
         yyy=np.zeros((len(y_train),1))
         yyy[:,0]= np.array(y_train)
         y_train=yyy

### Save Datasets for Documentation ###

    save_features_train = Path(standardpath, 'features_train', ml_options['name_model'] + '_save_cv_fold_')
    save_features_test =  Path(standardpath, 'features_test', ml_options['name_model'] + '_save_cv_fold_')
    save_labels_train =  Path(standardpath, 'labels_train', ml_options['name_model'] + '_save_cv_fold_')
    save_labels_test =  Path(standardpath, 'labels_test', ml_options['name_model'] + '_save_cv_fold_')

    full_path_X_train = save_features_train + str(random_state_option) + '_features_train.txt'
    with open(full_path_X_train, 'w', newline='') as file:
        csvsave = csv.writer(file, delimiter=' ')
        csvsave.writerows(X_train)

    full_path_y_train = save_labels_train + str(random_state_option) + '_labels_train.txt'
    with open(full_path_y_train, 'w', newline='') as file:
        csvsave = csv.writer(file, delimiter=' ')
        csvsave.writerows(y_train)

    full_path_X_test = save_features_test + str(random_state_option) + '_features_test.txt'
    with open(full_path_X_test, 'w', newline='') as file:
        csvsave = csv.writer(file, delimiter=' ')
        csvsave.writerows(X_test)

    full_path_y_test = save_labels_test + str(random_state_option) + '_labels_test.txt'
    with open(full_path_y_test, 'w', newline='') as file:
        csvsave = csv.writer(file, delimiter=' ')
        csvsave.writerows(y_test)


def process_and_run(numrun):

    global standardpath, ml_options

    random_state_option = numrun

    """
    "Import Data und Labels
    """

    X_train = pd.read_csv(full_path_X_train, sep="\s", header=None, engine='python')
    X_test = pd.read_csv(full_path_X_test, sep="\s", header=None, engine='python')
    y_train = pd.read_csv(full_path_y_train, sep="\s", header=None, engine='python')
    y_test = pd.read_csv(full_path_y_test, sep="\s", header=None, engine='python')

"""""""""""""""""
"
"
" Training-Set
"
"
"""""""""""""""""


"""
"Imputation missing values
"""

    if ml_options['missing_values_option'] == 0: # Just leave missing values
        X_train_imputed = copy.deepcopy(X_train)
    elif ml_options['missing_values_option']  == 1: # Drop missing values
        X_train = X_train.replace([999, 888,777], np.NaN)
        X_train_imputed = copy.deepcopy(X_train)
        X_train_imputed.dropna(inplace=True)
        y_train = y_train.drop(y_train.index[~y_train.index.isin(X_train_imputed.index)])
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
    elif ml_options['missing_values_option']  == 3: # MICE imputation
        scaffolding_arith = np.zeros((X_train.shape[0],X_train.shape[1]))
        scaffolding_median = np.zeros((X_train.shape[0],X_train.shape[1]))
        scaffolding_mode = np.zeros((X_train.shape[0],X_train.shape[1]))

        scaffolding_arith[X_train==999] = 1
        scaffolding_median[X_train==888] = 1
        scaffolding_mode[X_train==777] = 1

        X_train_arith = X_train.replace([777,888], 999)
        X_train_median = X_train.replace([777,999], 888)
        X_train_mode = X_train.replace([888,999], 777)

        imp_arith_mice = IterativeImputer(estimator=BayesianRidge(), missing_values=999, sample_posterior=True,
                        max_iter=10, initial_strategy="mean", random_state=random_state_option)
        imp_median_mice = IterativeImputer(estimator=BayesianRidge(), missing_values=888, sample_posterior=True,
                        max_iter=10, initial_strategy="median", random_state=random_state_option)
        imp_mode_mice = IterativeImputer(estimator=BayesianRidge(), missing_values=777, sample_posterior=True,
                        max_iter=10, initial_strategy="most_frequent", min_value=0, max_value=1, random_state=random_state_option)

        imp_arith_mice.fit(X_train_arith)
        imp_median_mice.fit(X_train_median)
        imp_mode_mice.fit(X_train_mode)
        X_train_arith_imputed = imp_arith_mice.transform(X_train_arith)
        X_train_median_imputed = imp_median_mice.transform(X_train_median)
        X_train_mode_imputed = imp_mode_mice.transform(X_train_mode)

        X_train_imputed = copy.deepcopy(X_train)

        for imputed_values_x in range(scaffolding_arith.shape[0]):
            for imputed_values_y in range(scaffolding_arith.shape[1]):
                if scaffolding_arith[imputed_values_x,imputed_values_y] == 1:
                    X_train_imputed.iloc[imputed_values_x,imputed_values_y] = X_train_arith_imputed[imputed_values_x,imputed_values_y]
                elif scaffolding_median[imputed_values_x,imputed_values_y] == 1:
                    X_train_imputed.iloc[imputed_values_x,imputed_values_y] = round(X_train_median_imputed[imputed_values_x,imputed_values_y])
                elif scaffolding_mode[imputed_values_x,imputed_values_y] == 1:
                    X_train_imputed.iloc[imputed_values_x,imputed_values_y] = round(X_train_mode_imputed[imputed_values_x,imputed_values_y])


"""
"Scaling
"""

    if ml_options['data_scaling_option'] == 0:
        X_train_imputed_scaled = copy.deepcopy(X_train_imputed)
    elif ml_options['data_scaling_option']  == 1:
        scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train_imputed)
        X_train_imputed_scaled = scaler.transform(X_train_imputed)


"""
"Feature Selection
"""

    y_train=np.ravel(y_train) # flattens y train


    if ml_options['feature_selection_option'] == 0:
        X_train_imputed_scaled_selected = copy.deepcopy(X_train_imputed_scaled)
    elif ml_options["feature_selection_option"] == 1:
        sel = VarianceThreshold(threshold=(.8 * (1 - .8))) # < 80% variance
        sel.fit(X_train_imputed_scaled)
        X_train_imputed_scaled_selected = sel.transform(X_train_imputed_scaled)
    elif ml_options["feature_selection_option"] == 3:
        clf_rfelim = RandomForestClassifier(n_estimators=1000, random_state=random_state_option)
        rfe = RFE(estimator=clf_rfelim, n_features_to_select=ml_options['number_features_recursive'],
                    step=ml_options['step_reduction_recursive'],random_state=random_state_option)
        rfe.fit(X_train_imputed_scaled, y_train)
        X_train_imputed_scaled_selected = copy.deepcopy(X_train_imputed_scaled)
        X_train_imputed_scaled_selected = X_train_imputed_scaled_selected[:,rfe.support_]
    elif ml_options['feature_selection_option'] == 4:
        X_train_scaled_imputed_rf = copy.deepcopy(X_train_imputed_scaled)
        clf_rfe = RandomForestClassifier(n_estimators=1000, random_state=random_state_option)
        rfe = RFECV(estimator=clf_rfe, step=ml_options['step_reduction_recursive'], cv=5,
                scoring=ml_options['scoring_recursive'], verbose = 1)
        rfe.fit(X_train_scaled_imputed_rf, y_train)
        X_train_imputed_scaled_selected = copy.deepcopy(X_train_scaled_imputed_rf)
        X_train_imputed_scaled_selected = X_train_imputed_scaled_selected[:,rfe.support_]

    elif ml_options['feature_selection_option'] == 5: #https://stats.stackexchange.com/questions/276865/interpreting-the-outcomes-of-elastic-net-regression-for-binary-classification
        if ml_options['data_scaling_option'] == 1:
            clf_elastic_logregression_features = SGDClassifier(loss='log', penalty='elasticnet', l1_ratio=0.5, fit_intercept=False, tol=0.0001, max_iter=1000, random_state=random_state_option)
            sfm = SelectFromModel(clf_elastic_logregression_features, threshold=ml_options['threshold_option'])
            sfm.fit(X_train_imputed_scaled, y_train)
            X_train_imputed_scaled_selected = sfm.transform(X_train_imputed_scaled)
        else:
            print('Please change data scaling option to perform Elstic Net‚')
            sys.exit("Execution therefore stopped")

"""
"Hyperparameter Tuning
"""

    if ml_options['hyperparameter_tuning_option'] == 0:
        standard_parameter = {'n_estimators': 1000,
                       'criterion': 'gini',
                       'max_features': 'auto',
                       'max_depth': None,
                       'min_samples_split': 2,
                       'min_samples_leaf': 1,
                       'bootstrap': True}
        best_parameter = standard_parameter
    elif ml_options['hyperparameter_tuning_option'] == 1:
        random_parameter = ml_options['hyperparameter_dict']

        clf_hyper_tuning = RandomForestClassifier(random_state=random_state_option)

        random_hyper_tuning = RandomizedSearchCV(estimator = clf_hyper_tuning, param_distributions = random_parameter,
                                n_iter = ml_options['n_iter_hyper_randsearch'], cv = ml_options['cvs_hyper_randsearch'],
                                verbose=0, random_state=random_state_option)
        random_hyper_tuning.fit(X_train_imputed_scaled_selected, y_train)
        best_parameter = random_hyper_tuning.best_params_

    elif ml_options['hyperparameter_tuning_option'] == 2:
        random_parameter = ml_options['hyperparameter_dict']

        param_list = list(ParameterSampler(random_parameter, n_iter=ml_options['n_iter_hyper_randsearch'],
                        random_state=random_state_option))
        oob_accuracy_hyper_tuning = np.zeros((ml_options['n_iter_hyper_randsearch']))
        counter_hyper_tuning = 0

        for current_parameter_setting in param_list:
            print("hyperparameter tuning iteration: {}".format(counter_hyper_tuning))
            clf_hyper_tuning = RandomForestClassifier(n_estimators= current_parameter_setting["n_estimators"],
                            criterion = current_parameter_setting["criterion"], max_features= current_parameter_setting["max_features"],
                            max_depth= current_parameter_setting["max_depth"], min_samples_split= current_parameter_setting["min_samples_split"],
                            min_samples_leaf= current_parameter_setting["min_samples_leaf"], bootstrap= current_parameter_setting["bootstrap"],
                            oob_score=True, random_state=random_state_option)
            clf_hyper_tuning = clf_hyper_tuning.fit(X_train_imputed_scaled_selected, y_train)
            oob_accuracy_hyper_tuning[counter_hyper_tuning] = clf_hyper_tuning.oob_score_
            counter_hyper_tuning = counter_hyper_tuning +1

        best_parameter = param_list[np.argmax(oob_accuracy_hyper_tuning)]

"""
"Random Forest Analyse
"""

    if ml_options['balanced_split_option'] == 0 or 1 or 2:
        clf = RandomForestClassifier(n_estimators= best_parameter["n_estimators"], criterion = best_parameter["criterion"],
            max_features= best_parameter["max_features"], max_depth= best_parameter["max_depth"],
            min_samples_split= best_parameter["min_samples_split"], min_samples_leaf= best_parameter["min_samples_leaf"],
            bootstrap= best_parameter["bootstrap"], oob_score=True, random_state=random_state_option)

    elif ml_options['balanced_split_option']  == 3:
        clf = RandomForestClassifier(n_estimators= best_parameter["n_estimators"], criterion = best_parameter["criterion"],
            max_features= best_parameter["max_features"], max_depth= best_parameter["max_depth"],
        min_samples_split= best_parameter["min_samples_split"], min_samples_leaf= best_parameter["min_samples_leaf"],
            bootstrap= best_parameter["bootstrap"], oob_score=True, class_weight='balanced', random_state=random_state_option)

    elif ml_options['balanced_split_option']  == 4:
        clf = BalancedRandomForestClassifier(n_estimators= best_parameter["n_estimators"], criterion = best_parameter["criterion"],
            max_features= best_parameter["max_features"], max_depth= best_parameter["max_depth"],
            min_samples_split= best_parameter["min_samples_split"], min_samples_leaf= best_parameter["min_samples_leaf"],
            bootstrap= best_parameter["bootstrap"], oob_score=True, random_state=random_state_option)

    elif ml_options['balanced_split_option']  == 5:
        clf = RandomForestClassifier(n_estimators= best_parameter["n_estimators"], criterion = best_parameter["criterion"],
            max_features= best_parameter["max_features"], max_depth= best_parameter["max_depth"],
            min_samples_split= best_parameter["min_samples_split"], min_samples_leaf= best_parameter["min_samples_leaf"],
            bootstrap= best_parameter["bootstrap"], oob_score=True, class_weight=ml_options['rf_classes_class_weight'],
            random_state=random_state_option)

    clf = clf.fit(X_train_imputed_scaled_selected, y_train)

    """""""""""""""""
    "
    "
    " Test-Set
    "
    "
    """""""""""""""""


    """
    "Imputation missing values - Test Set
    """

    if ml_options['missing_values_option'] == 0: # Just leave missing values
        X_test_imputed = copy.deepcopy(X_test)

    elif ml_options['missing_values_option']  == 1: # Drop missing values
        X_test = X_test.replace([999, 888,777], np.NaN)
        X_test_imputed = copy.deepcopy(X_train)
        X_test_imputed.dropna(inplace=True)
        y_test = y_test.drop(y_test.index[~y_test.index.isin(X_test_imputed.index)])

    elif ml_options['missing_values_option']  == 2: # Fill them with mean/median/mode
        X_test_imputed = imp_arith.transform(X_test)
        X_test_imputed = imp_median.transform(X_test_imputed)
        X_test_imputed = imp_mode.transform(X_test_imputed)

    elif ml_options['missing_values_option']  == 3: # MICE imputation
        scaffolding_arith_test = np.zeros((X_test.shape[0],X_test.shape[1]))
        scaffolding_median_test = np.zeros((X_test.shape[0],X_test.shape[1]))
        scaffolding_mode_test = np.zeros((X_test.shape[0],X_test.shape[1]))

        scaffolding_arith_test[X_test==999] = 1
        scaffolding_median_test[X_test==888] = 1
        scaffolding_mode_test[X_test==777] = 1

        X_test_arith = X_test.replace(777, 999)
        X_test_arith = X_test_arith.replace(888, 999)
        X_test_median = X_test.replace(777, 888)
        X_test_median = X_test_median.replace(999, 888)
        X_test_mode = X_test.replace(888, 777)
        X_test_mode = X_test_mode.replace(999, 777)

    X_test_arith_imputed = imp_arith_mice.transform(X_test_arith)
    X_test_median_imputed = imp_median_mice.transform(X_test_median)
    X_test_mode_imputed = imp_mode_mice.transform(X_test_mode)

    X_test_imputed = copy.deepcopy(X_test)

    for imputed_values_x in range(scaffolding_arith_test.shape[0]):
        for imputed_values_y in range(scaffolding_arith_test.shape[1]):
            if scaffolding_arith_test[imputed_values_x,imputed_values_y] == 1:
                X_test_imputed.iloc[imputed_values_x,imputed_values_y] = round(X_test_arith_imputed[imputed_values_x,imputed_values_y])
            elif scaffolding_median_test[imputed_values_x,imputed_values_y] == 1:
                X_test_imputed.iloc[imputed_values_x,imputed_values_y] = round(X_test_median_imputed[imputed_values_x,imputed_values_y])
            elif scaffolding_mode_test[imputed_values_x,imputed_values_y] == 1:
                X_test_imputed.iloc[imputed_values_x,imputed_values_y] = round(X_test_mode_imputed[imputed_values_x,imputed_values_y])

"""
"Scaling - Test Set
"""


    if ml_options['data_scaling_option']  == 0:
        X_test_imputed_scaled = copy.deepcopy(X_test_imputed)
    elif ml_optionsn['data_scaling_option']  == 1:
        X_test_imputed_scaled = scaler.transform(X_test_imputed)

    """
    "Feature Selection - Test Set
    """

    y_test=np.ravel(y_test)

    if ml_options['feature_selection_option'] == 0:
        X_test_scaled_imputed_selected = copy.deepcopy(X_test_imputed_scaled)
    elif ml_options['feature_selection_option'] == 1:
        X_test_scaled_imputed_selected = sel.transform(X_test_imputed_scaled)
    elif ml_options['feature_selection_option'] == 3 or 4:
        X_test_scaled_imputed_selected = copy.deepcopy(X_test_imputed_scaled)
        X_test_scaled_imputed_selected = X_test_scaled_imputed_selected[:,rfe.support_]
    elif ml_options['feature_selection_option'] == 5:
        X_test_scaled_imputed_selected = sfm.transform(X_test_imputed_scaled)

    else:
        print("Not working yet")
        sys.exit("Stop Stop Stop")

    """
    "Prediction im Test Set
    """

    y_prediction = np.zeros((len(y_test), 3))

    y_prediction[:,0] = clf.predict(X_test_scaled_imputed_selected)

    y_prediction[:,1] = y_test[:]

    counter_class1_correct = 0
    counter_class2_correct = 0
    counter_class1_incorrect = 0
    counter_class2_incorrect = 0

    for i in range(len(y_test)):
        if y_prediction[i,0] == y_prediction[i,1]:
            y_prediction[i,2] = 1
            if y_prediction[i,1] == 1:
                counter_class1_correct += 1
            else:
                counter_class2_correct += 1
        else:
            y_prediction[i,2] = 0
            if y_prediction[i,1] == 1:
                counter_class1_incorrect += 1
            else:
                counter_class2_incorrect += 1

""" Calculate accuracy scores """

    accuracy = y_prediction.mean(axis=0)[2]
    accuracy_class1 = counter_class1_correct / (counter_class1_correct + counter_class1_incorrect)
    accuracy_class2 = counter_class2_correct / (counter_class2_correct + counter_class2_incorrect)
    balanced_accuracy = (accuracy_class1 + accuracy_class2) / 2
    oob_accuracy = clf.oob_score_
    log_loss_value = log_loss(y_test, clf.predict_proba(X_test_scaled_imputed_selected), normalize=True)

""" Calculate feature importances """

    if ml_options['feature_selection_option'] == 0:
        feature_importances = clf.feature_importances_
    elif ml_options['feature_selection_option'] == 1:
        feature_importances = np.zeros((len(sel.get_support())))
        counter_features_selected = 0
        for number_features in range(len(sel.get_support())):
            if sel.get_support()[number_features] == True:
                feature_importances[number_features] = clf.feature_importances_[counter_features_selected]
                counter_features_selected += 1
            else:
                feature_importances[number_features] = 0

    elif ml_options['feature_selection_option'] == 3 or 4:
        feature_importances = np.zeros((len(rfe.support_)))
        counter_features_selected = 0
        for number_features in range(len(rfe.support_)):
            if rfe.support_[number_features] == True:
                feature_importances[number_features] = clf.feature_importances_[counter_features_selected]
                counter_features_selected += 1
            else:
                feature_importances[number_features] = 0

    elif ml_options['feature_selection_option'] == 5:
        feature_importances = np.zeros((len(sfm.get_support())))
        counter_features_selected = 0
        for number_features in range(len(sfm.get_support())):
            if sfm.get_support()[number_features] == True:
                feature_importances[number_features] = clf.feature_importances_[counter_features_selected]
                counter_features_selected += 1
            else:
                feature_importances[number_features] = 0


    fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test_scaled_imputed_selected)[:,1])
    mean_fpr = np.linspace(0, 1, 100)
    tprs = np.interp(mean_fpr, fpr, tpr)
    roc_auc = auc(fpr, tpr)
    fraction_positives, mean_predicted_value = calibration_curve(y_test, clf.predict_proba(X_test_scaled_imputed_selected)[:,1], n_bins=10)

"""""""""""""""""
"Permutationstest
"""""""""""""""""

save_option = Path(standardpath, 'individual_rounds', ml_options['name_model'], '_clf_round_', str(random_state_option))
if ml_options['save_clf_option'] == 1:
    with open(save_option, 'wb') as AutoPickleFile:
            pickle.dump((clf, y_test, y_train, X_train_imputed_scaled_selected, X_test_scaled_imputed_selected), AutoPickleFile)
    else:
        print("Clf wird nicht gespeichert")

    pvalue = 1
if ml_options['permutation_option']  == 1:
    counter_random = 0
    for j in range(ml_options['n_permutations_option']):
        print("\n Permutationstest: der aktuelle Durchgang ist %s" % (j))

        """
        "Permutierung Labels - Permutationstest
        """

        y_test_random = copy.deepcopy(y_test)
        y_test_random = np.ravel(y_test_random)
        y_test_random = shuffle(y_test_random, random_state=j)

        y_train_random = copy.deepcopy(y_train)
        y_train_random = np.ravel(y_train_random)
        y_train_random = shuffle(y_train_random, random_state=j)

        """
        "Random Forest Analyse - Permutationstest
        """

        if ml_options['balanced_split_option'] == 0 or 1 or 2:
            clf_perm = RandomForestClassifier(n_estimators= best_parameter["n_estimators"], criterion = best_parameter["criterion"], max_features= best_parameter["max_features"], max_depth= best_parameter["max_depth"], min_samples_split= best_parameter["min_samples_split"], min_samples_leaf= best_parameter["min_samples_leaf"], bootstrap= best_parameter["bootstrap"], oob_score=True, random_state=random_state_option)
        elif ml_options['balanced_split_option']  == 3:
            clf_perm = RandomForestClassifier(n_estimators= best_parameter["n_estimators"], criterion = best_parameter["criterion"], max_features= best_parameter["max_features"], max_depth= best_parameter["max_depth"], min_samples_split= best_parameter["min_samples_split"], min_samples_leaf= best_parameter["min_samples_leaf"], bootstrap= best_parameter["bootstrap"], oob_score=True, class_weight='balanced', random_state=random_state_option)
        elif ml_options['balanced_split_option']  == 4:
            clf_perm = BalancedRandomForestClassifier(n_estimators= best_parameter["n_estimators"], criterion = best_parameter["criterion"], max_features= best_parameter["max_features"], max_depth= best_parameter["max_depth"], min_samples_split= best_parameter["min_samples_split"], min_samples_leaf= best_parameter["min_samples_leaf"], bootstrap= best_parameter["bootstrap"], oob_score=True, random_state=random_state_option)
        elif ml_options['balanced_split_option']  == 5:
            clf_perm = RandomForestClassifier(n_estimators= best_parameter["n_estimators"], criterion = best_parameter["criterion"], max_features= best_parameter["max_features"], max_depth= best_parameter["max_depth"], min_samples_split= best_parameter["min_samples_split"], min_samples_leaf= best_parameter["min_samples_leaf"], bootstrap= best_parameter["bootstrap"], oob_score=True, class_weight=ml_options['rf_classes_class_weight'], random_state=random_state_option)
  
        y_prediction_perm = np.zeros((len(y_test_random), 3))
        y_prediction_perm[:,0] = clf_perm.predict(X_test_scaled_imputed_selected)
        y_prediction_perm[:,1] = y_test_random[:]

        counter_class1_correct_perm = 0
        counter_class2_correct_perm = 0
        counter_class1_incorrect_perm = 0
        counter_class2_incorrect_perm = 0

        for i in range(len(y_test_random)):
            if y_prediction_perm[i,0] == y_prediction_perm[i,1]:
                y_prediction_perm[i,2] = 1
                if y_prediction_perm[i,1] == 1:
                    counter_class1_correct_perm += 1
                else:
                    counter_class2_correct_perm += 1
            else:
                y_prediction_perm[i,2] = 0
                if y_prediction_perm[i,1] == 1:
                    counter_class1_incorrect_perm += 1
                else:
                    counter_class2_incorrect_perm += 1

        """
        "Check Significance - Permutationstest
        """
        accuracy_class1_perm = counter_class1_correct_perm / (counter_class1_correct_perm + counter_class1_incorrect_perm)
        accuracy_class2_perm = counter_class2_correct_perm / (counter_class2_correct_perm + counter_class2_incorrect_perm)
        balanced_accuracy_perm = (accuracy_class1_perm + accuracy_class2_perm) / 2

        if balanced_accuracy < balanced_accuracy_perm:
            counter_random += 1

    pvalue = (counter_random + 1)/(j + 1 + 1)


return accuracy, accuracy_class1, accuracy_class2, balanced_accuracy, oob_accuracy, log_loss_value, feature_importances, fpr, tpr, tprs, roc_auc, fraction_positives, mean_predicted_value, pvalue

"""
"Safe metrics per round in files
"""

def save_in_txt(accuracy, accuracy_class1, accuracy_class2, balanced_accuracy, oob_accuracy, log_loss_value, feature_importances, fpr, tpr, tprs, roc_auc, fraction_positives, mean_predicted_value, pvalue):
 
    global standardpath, ml_options

    save_option_prebuild = Path(standardpath, 'accuracy')
    
    save_option = Path(save_option_prebuild, ml_options['name_model'] + '_per_round_accuracy.txt')   
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(accuracy)

    save_option = Path(save_option_prebuild, ml_options['name_model'] + '_per_round_accuracy_class1.txt')   
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(accuracy_class1)
            
    save_option = Path(save_option_prebuild, ml_options['name_model'] + '_per_round_accuracy_class2.txt') 
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(accuracy_class2)
            
    save_option = Path(save_option_prebuild, ml_options['name_model'] + '_per_round_balanced_accuracy.txt')   
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(balanced_accuracy)
            
    save_option = Path(save_option_prebuild, ml_options['name_model'] + '_per_round_oob_accuracy.txt')   
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(oob_accuracy)
            
    save_option = Path(save_option_prebuild, ml_options['name_model'] + '_per_round_log_loss.txt')   
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(log_loss_value)           
               
    save_option = Path(save_option_prebuild, ml_options['name_model'] + '_per_round_feature_importances.txt')   
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerows(feature_importances)
            
    save_option = Path(save_option_prebuild, ml_options['name_model'] + '_per_round_fpr.txt')   
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(fpr)
            
    save_option = Path(save_option_prebuild, ml_options['name_model'] + '_per_round_tpr.txt')   
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(tpr)
            
    save_option = Path(save_option_prebuild, ml_options['name_model'] + '_per_round_tprs.txt')   
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(tprs)
            
    save_option = Path(save_option_prebuild, ml_options['name_model'] + '_per_round_roc_auc.txt')   
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(roc_auc)
            
    save_option = Path(save_option_prebuild, ml_options['name_model'] + '_per_round_fraction_positives.txt')   
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(fraction_positives)
            
    save_option = Path(save_option_prebuild, ml_options['name_model'] + '_per_round_predicted_value.txt')   
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(mean_predicted_value)
            
    save_option = Path(save_option_prebuild, ml_options['name_model'] + '_per_round_p_value.txt')   
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(pvalue)


def list_to_flatlist(input_data):
    
    accuracy_flat = []
    accuracy_class1_flat = []
    accuracy_class2_flat = []
    balanced_accuracy_flat = []
    oob_accuracy_flat = []
    log_loss_value_flat = []
    feature_importances_flat = np.zeros((len(input_data),len(outcomes[0][6])))
    fpr_flat = []
    tpr_flat = []
    tprs_flat = []
    roc_auc_flat = []
    fraction_positives_flat = []
    mean_predicted_value_flat = []
    pvalue_flat = []
        
    for counter, sublist in enumerate(input_data):
        for itemnumber in range(len(sublist)):
            if itemnumber == 0:
                accuracy_flat.append(sublist[itemnumber])
            elif itemnumber == 1:
                accuracy_class1_flat.append(sublist[itemnumber])
            elif itemnumber == 2:
                accuracy_class2_flat.append(sublist[itemnumber])
            elif itemnumber == 3:
                balanced_accuracy_flat.append(sublist[itemnumber])
            elif itemnumber == 4:
                oob_accuracy_flat.append(sublist[itemnumber])
            elif itemnumber == 5:
                log_loss_value_flat.append(sublist[itemnumber])
            elif itemnumber == 6:
                feature_importances_flat[counter,:] = sublist[itemnumber]
            elif itemnumber == 7:
                fpr_flat.append(sublist[itemnumber])
            elif itemnumber == 8:
                tpr_flat.append(sublist[itemnumber])
            elif itemnumber == 9:
                tprs_flat.append(sublist[itemnumber])
            elif itemnumber == 10:
                roc_auc_flat.append(sublist[itemnumber])
            elif itemnumber == 11:
                fraction_positives_flat.append(sublist[itemnumber])
            elif itemnumber == 12:
                mean_predicted_value_flat.append(sublist[itemnumber])
            elif itemnumber == 13:
                pvalue_flat.append(sublist[itemnumber])

        
    return accuracy_flat, accuracy_class1_flat, accuracy_class2_flat, balanced_accuracy_flat, oob_accuracy_flat, log_loss_value_flat, feature_importances_flat, fpr_flat, tpr_flat, tprs_flat, roc_auc_flat, fraction_positives_flat, mean_predicted_value_flat, pvalue_flat

def aggregate_this(accuracy, accuracy_class1, accuracy_class2, balanced_accuracy, oob_accuracy, log_loss_value, feature_importances, fpr, tpr, tprs, roc_auc, fraction_positives, mean_predicted_value, pvalue):
    
    global standardpath, ml_options

    accuracy_min = min(accuracy)
    accuracy_max = max(accuracy)
    accuracy_mean = np.mean(accuracy)
    accuracy_std = np.std(accuracy)
    accuracy_class1_min = min(accuracy_class1)
    accuracy_class1_max = max(accuracy_class1)
    accuracy_class1_mean = np.mean(accuracy_class1)
    accuracy_class1_std = np.std(accuracy_class1)
    accuracy_class2_min = min(accuracy_class2)
    accuracy_class2_max = max(accuracy_class2)
    accuracy_class2_mean = np.mean(accuracy_class2)
    accuracy_class2_std = np.std(accuracy_class2)
    balanced_accuracy_min = min(balanced_accuracy)
    balanced_accuracy_max = max(balanced_accuracy)
    balanced_accuracy_mean = np.mean(balanced_accuracy)
    balanced_accuracy_std = np.std(balanced_accuracy)
    oob_accuracy_min = min(oob_accuracy)
    oob_accuracy_max = max(oob_accuracy)
    oob_accuracy_mean = np.mean(oob_accuracy)
    oob_accuracy_std = np.std(oob_accuracy)
    log_loss_value_min = min(log_loss_value)
    log_loss_value_max = max(log_loss_value)
    log_loss_value_mean = np.mean(log_loss_value)
    log_loss_value_std = np.std(log_loss_value)
    feature_importances_min = feature_importances.min(axis=0).reshape(1,feature_importances.shape[1])
    feature_importances_max = feature_importances.max(axis=0).reshape(1,feature_importances.shape[1])
    feature_importances_mean = feature_importances.mean(axis=0).reshape(1,feature_importances.shape[1])
    feature_importances_std = feature_importances.std(axis=0).reshape(1,feature_importances.shape[1])
    pvalue_min = min(pvalue)
    pvalue_max = max(pvalue)
    pvalue_mean = np.mean(pvalue)
    pvalue_std = np.std(pvalue)
    
    number_rounds = len(accuracy)
    
    
    savepath_option = Path(standardpath, 'accuracy', ml_options['name_model'] + '.txt')  

    with open(safepath_option, 'w') as f
        f.write('Number of Rounds: ' + str(number_rounds) + 
             '\nMin Accuracy: ' + str(accuracy_min) + '\nMax Accuracy: ' + str(accuracy_max) + '\nMean Accuracy: ' + str(accuracy_mean) + '\nStd Accuracy: ' + str(accuracy_std) +
             '\nMin Accuracy_class_1: ' + str(accuracy_class1_min) + '\nMax Accuracy_class_1: ' + str(accuracy_class1_max) + '\nMean Accuracy_class_1: ' + str(accuracy_class1_mean) + '\nStd Accuracy_class_1: ' + str(accuracy_class1_std) +
             '\nMin Accuracy_class_2: ' + str(accuracy_class2_min) + '\nMax Accuracy_class_2: ' + str(accuracy_class2_max) + '\nMean Accuracy_class_2: ' + str(accuracy_class2_mean) + '\nStd Accuracy_class_2: ' + str(accuracy_class2_std) +
             '\nMin Balanced_Accuracy: ' + str(balanced_accuracy_min) + '\nMax Balanced_Accuracy: ' + str(balanced_accuracy_max) + '\nMean Balanced_Accuracy: ' + str(balanced_accuracy_mean) + '\nStd Balanced_Accuracy: ' + str(balanced_accuracy_std) +
             '\nMin OOB_Accuracy: ' + str(oob_accuracy_min) + '\nMax OOB_Accuracy: ' + str(oob_accuracy_max) + '\nMean OOB_Accuracy: ' + str(oob_accuracy_mean) + '\nStd OOB_Accuracy: ' + str(oob_accuracy_std) +
             '\nMin Log-Loss: ' + str(log_loss_value_min) + '\nMax Log-Loss: ' + str(log_loss_value_max) + '\nMean Log-Loss: ' + str(log_loss_value_mean) + '\nStd Log-Loss: ' + str(log_loss_value_std) +
             '\nMin p: ' + str(pvalue_min) + '\nMax p: ' + str(pvalue_max) + '\nMean p: ' + str(pvalue_mean) + '\nStd p: ' + str(pvalue_std) +
             '\nComparator p: ' + str(p_comparator) + '\nComparator t: ' + str(t_comparator) + '\nComparator df: ' + str(df_comparator) + '\nComparator name: ' + str(name_comparator) +
             '\nMin feature_importances: ' + str(feature_importances_min) + '\nMax feature_importances: ' + str(feature_importances_max) + '\nMean feature_importances: ' + str(feature_importances_mean) + '\nStd feature_importances: ' + str(feature_importances_std))

    print('Number of Rounds: ' + str(number_rounds) + 
             '\nMin Accuracy: ' + str(accuracy_min) + '\nMax Accuracy: ' + str(accuracy_max) + '\nMean Accuracy: ' + str(accuracy_mean) + '\nStd Accuracy: ' + str(accuracy_std) +
             '\nMin Accuracy_class_1: ' + str(accuracy_class1_min) + '\nMax Accuracy_class_1: ' + str(accuracy_class1_max) + '\nMean Accuracy_class_1: ' + str(accuracy_class1_mean) + '\nStd Accuracy_class_1: ' + str(accuracy_class1_std) +
             '\nMin Accuracy_class_2: ' + str(accuracy_class2_min) + '\nMax Accuracy_class_2: ' + str(accuracy_class2_max) + '\nMean Accuracy_class_2: ' + str(accuracy_class2_mean) + '\nStd Accuracy_class_2: ' + str(accuracy_class2_std) +
             '\nMin Balanced_Accuracy: ' + str(balanced_accuracy_min) + '\nMax Balanced_Accuracy: ' + str(balanced_accuracy_max) + '\nMean Balanced_Accuracy: ' + str(balanced_accuracy_mean) + '\nStd Balanced_Accuracy: ' + str(balanced_accuracy_std) +
             '\nMin OOB_Accuracy: ' + str(oob_accuracy_min) + '\nMax OOB_Accuracy: ' + str(oob_accuracy_max) + '\nMean OOB_Accuracy: ' + str(oob_accuracy_mean) + '\nStd OOB_Accuracy: ' + str(oob_accuracy_std) +
             '\nMin Log-Loss: ' + str(log_loss_value_min) + '\nMax Log-Loss: ' + str(log_loss_value_max) + '\nMean Log-Loss: ' + str(log_loss_value_mean) + '\nStd Log-Loss: ' + str(log_loss_value_std) +
             '\nMin p: ' + str(pvalue_min) + '\nMax p: ' + str(pvalue_max) + '\nMean p: ' + str(pvalue_mean) + '\nStd p: ' + str(pvalue_std) +
             '\nComparator p: ' + str(p_comparator) + '\nComparator t: ' + str(t_comparator) + '\nComparator df: ' + str(df_comparator) + '\nComparator name: ' + str(name_comparator))
 
      
    


def reminder():
    print("Sind die Daten als tab-separierter Text gespeichert?")
    print("Sind Kommas in Punkte verwandelt?")
    print("Sind die Werte 777, 888, 999 für missings reserviert, und sind alle missings mit diesen Werten kodiert?")
    print("Falls ein unabhängiges Test-Set genutzt werden soll: sind die Daten entfernt?")
    print("Sind die Standardparameter bzw der Range der Parameter fürs Tuning für das Datenset sinnvoll?")
    input("Press Enter to continue...")
    


if __name__ == '__main__':
    reminder()
    runs_list = [i for i in range(ml_options["n_iterations"])]
    outcomes = []
    
    outcomes[:] = map(process_and_run,runs_list)

    accuracy_flat, accuracy_class1_flat, accuracy_class2_flat, balanced_accuracy_flat, oob_accuracy_flat, log_loss_value_flat, feature_importances_flat, fpr_flat, tpr_flat, tprs_flat, roc_auc_flat, fraction_positives_flat, mean_predicted_value_flat, pvalue_flat = list_to_flatlist(outcomes)
    save_this_stuff(accuracy_flat, accuracy_class1_flat, accuracy_class2_flat, balanced_accuracy_flat, oob_accuracy_flat, log_loss_value_flat, feature_importances_flat, fpr_flat, tpr_flat, tprs_flat, roc_auc_flat, fraction_positives_flat, mean_predicted_value_flat, pvalue_flat)
    aggregate_this(accuracy_flat, accuracy_class1_flat, accuracy_class2_flat, balanced_accuracy_flat, oob_accuracy_flat, log_loss_value_flat, feature_importances_flat, fpr_flat, tpr_flat, tprs_flat, roc_auc_flat, fraction_positives_flat, mean_predicted_value_flat, pvalue_flat)

    print("elapsed_time = {}", time.time() - start_time)