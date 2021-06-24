import pandas as pd
from config import DATAPATH_IN
from . import coach_dicts

def prepare_data(ml_options, features=None):

    if features is not None:
        features = features
    else:
        features = pd.read_csv(DATAPATH_IN, sep=";", low_memory=False)
        features = features[ml_options["feature_columns"]]
    
    coach_dict = coach_dicts.coach_dict
    anonym_dict = coach_dicts.anonym_dict
    features["coach_gender"] = features["coach"]
    features.replace({"coach_gender": coach_dict}, inplace=True)
    features.replace({"coach": anonym_dict}, inplace=True)
    features["coach"] = features["coach"]

    features['studyVariant'] = pd.get_dummies(features['studyVariant'])

    features["registration_dt"] = pd.to_datetime(features["registration"])
    features["registration"] = features["registration_dt"].dt.year

    features["age"] = features["registration_dt"].dt.year - features['PRE_birth']
    features.drop(["registration_dt",  "PRE_birth"], axis=1, inplace= True)

    if ml_options["categorical_encoding"] == 1:
        encoders_list = ["PRE_work", "PRE_household", "PRE_residence", "PRE_relation", 'PRE_sickleave',"registration", "coach"] #"TI_rekrut", 
        features = pd.get_dummies(features, columns=encoders_list)
   
    
    return features