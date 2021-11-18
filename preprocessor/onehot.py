import pandas as pd
import numpy as np
from config import DATAPATH_IN
from collections import Counter

def cumulatively_categorise(column,threshold=0.80):
    threshold_value=int(threshold*len(column))
    #Initialise an empty list for our new minimised categories
    categories_list=[]
    #Initialise a variable to calculate the sum of frequencies
    s=0
    #Create a counter dictionary of the form unique_value: frequency
    counts=Counter(column)

    #Loop through the category name and its corresponding frequency after sorting the categories by descending order of frequency
    for i,j in counts.most_common():
        #Add the frequency to the global sum
        s+=dict(counts)[i]
        #Append the category name to the list
        categories_list.append(i)
        #Check if the global sum has reached the threshold value, if so break the loop
        if s>=threshold_value:
            break
        #Append the category Other to the list
    categories_list.append('Other')

    #Replace all instances not in our new categories by Other  
    new_column=column.apply(lambda x: x if x in categories_list else 'Other')

    return new_column


def prepare_data(ml_options, features=None):

    if features is not None:
        features = features
    else:
        features = pd.read_csv(DATAPATH_IN, sep=";", low_memory=False)
        features = features[ml_options["feature_columns"]]
    
    #coach_dict = coach_dicts.coach_dict
    #features["coach_gender"] = features["coach"]
    #features.replace({"coach_gender": coach_dict}, inplace=True)

    #if ml_options["include_coach"] == 1:
    #    transformed_column = cumulatively_categorise(features['coach'])
    #    features['coach'] = transformed_column
    
    #elif ml_options["include_coach"] == 0:
    #    features.drop(["coach"], axis=1, inplace=True)


    features["registration_dt"] = pd.to_datetime(features["registration"])
    features["registration"] = features["registration_dt"].dt.year

    features["age"] = features["registration_dt"].dt.year - features['PRE_birth']

    features["corona_train"] = np.where((features["registration_dt"] >= pd.Timestamp(2020, 1, 15)),1,0)

    features.drop(["registration_dt",  "PRE_birth"], axis=1, inplace= True)

    features["PRE_residence"].replace(5,4)
    features["PRE_household"].replace(3,4)
    features["PRE_work"].replace(3,4)
    features["PRE_work"].replace(5,6)
    features["PRE_work"].replace(1,7)
    features["PRE_relation"].replace(3,0)

    if ml_options["categorical_encoding"] == 1:
        #if ml_options["include_coach"] == 1:
         #   encoders_list = ["PRE_work", "PRE_household", "PRE_residence", "PRE_relation", 'PRE_sickleave',"registration",'studyVariant', "coach"] 
        encoders_list = ["PRE_work", "PRE_household", "PRE_residence", "PRE_relation", 'PRE_sickleave',"registration", 'studyVariant'] 

        features = pd.get_dummies(features, columns=encoders_list)
   
    
    return features