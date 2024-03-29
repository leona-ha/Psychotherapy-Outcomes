import pandas as pd
import numpy as np
from config import DATAPATH_IN


def prepare_data(ml_options, features_in=None):

    if features_in is not None:
        features = features_in
    else:
        features = pd.read_csv(DATAPATH_IN, sep=";", low_memory=False)
        features = features[ml_options["feature_columns"]]
    
    features["registration_dt"] = pd.to_datetime(features["registration"])
    features["registration"] = features["registration_dt"].dt.year

    features["age"] = features["registration_dt"].dt.year - features['PRE_birth']

    features["corona_train"] = np.where((features["registration_dt"] >= pd.Timestamp(2020, 1, 15)),1,0)

    features.drop(["registration_dt",  "PRE_birth"], axis=1, inplace= True)
    
    features["PRE_residence"].replace(5,4, inplace=True) # Gehöft/Alleinstehendes Haus --> Dorf
    features["PRE_household"].replace(3,4, inplace=True) # nur mit Kindern --> mit anderen Personen
    features["PRE_work"].replace(3,4, inplace=True) # Student --> Schüler/Auszubildend
    features["PRE_work"].replace(5,6, inplace=True) # Rentner --> Derzeit arbeitslos
    features["PRE_work"].replace(1,7, inplace=True) # Selbstständig --> Sonstige
    features["PRE_relation"].replace(3,0, inplace=True) # Verwitwet --> Ledig
    features ["PRE_education"].replace(0,1, inplace=True) # 
    
    if ml_options["categorical_encoding"] == 1:
        
        encoders_list = ["PRE_work", "PRE_household", "PRE_residence", "PRE_relation", 'PRE_sickleave',"registration", 'studyVariant'] 

        features = pd.get_dummies(features, columns=encoders_list)
   
    
    return features