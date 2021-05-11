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
    "Scale features
    """
    if ml_options['data_scaling_option'] ==1:
        if ml_options['feature_selection_option'] == 5:
            features = X_train.columns.tolist()
        else:
            features = ml_options['scaling_columns']

        X_train_scaled = copy.deepcopy(X_train)
        X_test_scaled = copy.deepcopy(X_test)

        for feature in features: 
            scalerpath = os.path.join(MODEL_PATH, f'std_scaler_{feature}_{ml_options["model_name"]}.pkl')
            if not os.path.isdir(scalerpath):
                print(f'Fit new scaler to feature {feature}')
                scaler = StandardScaler()
                scaler = scaler.fit(X_train[[feature]])
                X_train_scaled[feature] = scaler.transform(X_train_scaled[[feature]])
                X_test_scaled[feature] = scaler.transform(X_test_scaled[[feature]])
                pickle.dump(scaler, open(scalerpath, 'wb'))
            else:
                print(f'Load existing scaler fot feature {feature}')
                scaler = pickle.load(open(scalerpath, 'rb'))
                X_train_scaled[feature] = scaler.transform(X_train_scaled[[feature]])
                X_test_scaled[feature] = scaler.transform(X_test_scaled[[feature]])
    else: 
        X_train_scaled = copy.deepcopy(X_train)
        X_test_scaled = copy.deepcopy(X_test)
    
    save_model_split = Path(DATAPATH_OUT, ml_options['model_name'], str(ml_options["seed"]))
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


    return X_train_scaled, X_test_scaled, y_train, y_test