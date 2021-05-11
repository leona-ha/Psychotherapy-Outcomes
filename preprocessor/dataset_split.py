from config import STANDARDPATH, DATAPATH_IN, DATAPATH_OUT, DATAPATH_INTERIM
from sklearn.model_selection import train_test_split
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from . import coach_dicts
import csv


def prepare_data(ml_options, numrun, features=None):

    ml_options['seed'] = numrun
    target_id = ml_options["target_id"]
    target_columns_post = ml_options["target_columns_post"]
    target_columns_pre = ml_options["target_columns_pre"]

    if features is not None:
        features = features
    else:
        features = pd.read_csv(DATAPATH_IN, sep=";", low_memory=False)
        features = features[ml_options["feature_columns"]]
    
    if ml_options['missing_outcome_option'] == 1:
        print("The number of dropped rows is: ", len(features[features[target_columns_post].isnull().all(axis=1)]))
        features = features[features[target_columns_post].notnull().all(axis=1)]
            
    """
    "Create sum score for post phq 
    """
    features[target_columns_post] = features[target_columns_post].apply(pd.to_numeric, errors='coerce').astype('Int64')
    features["outcome_sum_post"] = features[target_columns_post].sum(axis=1)
    features.drop(target_columns_post, axis=1, inplace=True)

    """
    "Create sum score for pre phq 
    """
    features[target_columns_pre] = features[target_columns_pre].apply(pd.to_numeric, errors='coerce').astype('Int64')
    features["outcome_sum_pre"] = features[target_columns_pre].sum(axis=1)
            

    """
    "Create RCI variable
    """
    features[target_id] = features["outcome_sum_pre"] - features["outcome_sum_post"]
    features[target_id] = features[target_id].apply(lambda x:0 if x >= 5 else 1)
    features.drop("outcome_sum_post", axis=1, inplace=True)
        
    labels = features[ml_options["target_id"]]
    features.drop(ml_options["target_id"],axis=1, inplace=True)


    if ml_options["sampling"] in (0,4,5,6):
        if ml_options['stratify_option'] == 0:
            strat_option = None
        else:
            strat_option = labels

        X_train, X_test, y_train, y_test = train_test_split(
                features, labels, stratify=strat_option,
                test_size=ml_options['test_size_option'], random_state=ml_options['seed'])

    elif ml_options['sampling'] in (1,2):
        X_train, X_test, y_train, y_test = train_test_split(
                    features, labels, test_size=ml_options['test_size_option'], random_state=ml_options['seed'])

        if ml_options['sampling'] == 1:
            min_number = min(y_train.value_counts())
            sample_for_balancing = pd.concat([X_train, y_train], axis=1, sort=False)
            reduced_sample = sample_for_balancing.groupby(sample_for_balancing[ml_options['target_id']],
                sort=False).apply(lambda frame: frame.sample(int(min_number),random_state=ml_options['seed']))
            mask = reduced_sample.index.get_level_values(-1)
            X_train = X_train.loc[mask]
            y_train = y_train.loc[mask]
            
        elif ml_options['sampling'] == 2:
            min_number = min(y_test.value_counts())
            sample_for_balancing = pd.concat([X_test, y_test], axis=1, sort=False)
            reduced_sample = sample_for_balancing.groupby(sample_for_balancing[ml_options['target_id']],
                sort=False).apply(lambda frame: frame.sample(int(min_number),random_state=ml_options['seed']))
            mask = reduced_sample.index.get_level_values(-1)
            X_test = X_test.loc[mask]
            y_test = y_test.loc[mask]


    elif ml_options['sampling'] == 3:
        min_number = min(labels.value_counts())
        sample_for_balancing = pd.concat([features, labels], axis=1, sort=False)
        reduced_sample = sample_for_balancing.groupby(sample_for_balancing[ml_options["target_id"]],
                    sort=False).apply(lambda frame: frame.sample(min_number, random_state=ml_options['seed']))
        mask = reduced_sample.index.get_level_values(-1)

        X_reduced_sample = features.loc[mask]
        y_reduced_sample = labels.loc[mask]
        X_train, X_test, y_train, y_test = train_test_split(
            X_reduced_sample, y_reduced_sample, stratify=y_reduced_sample, test_size=ml_options['test_size_option'], random_state=ml_options['seed'])

    if ml_options["save_split_option"] == 1:
        save_model_split = Path(DATAPATH_INTERIM, ml_options['model_name'], str(ml_options["seed"]))
        if not os.path.exists(save_model_split):
            os.makedirs(save_model_split)
    
        full_path_X_train = os.path.join(save_model_split,'_features_train.csv')
        full_path_X_test = os.path.join(save_model_split,'_features_test.csv')
        full_path_y_train = os.path.join(save_model_split,'_labels_train.csv')
        full_path_y_test = os.path.join(save_model_split,'_labels_test.csv')

        y_test.to_csv(full_path_y_test, sep=";", encoding="utf-8", header=True)
        y_train.to_csv(full_path_y_train, sep=";", encoding="utf-8", header=True)
        X_test.to_csv(full_path_X_test, sep=";", encoding="utf-8", header=True)
        X_train.to_csv(full_path_X_train, sep=";", encoding="utf-8", header=True)


    return X_train, X_test, y_train, y_test
