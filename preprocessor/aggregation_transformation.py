from config import STANDARDPATH, DATAPATH_IN, DATAPATH_OUT
from sklearn.model_selection import train_test_split
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
import csv
import copy

def prepare_data(ml_options, X_train, X_test, y_train, y_test):
    #data = pd.read_csv(DATAPATH_IN, sep=";", low_memory=False)
    #data = data[ml_options["feature_columns"]]

    dflist = [X_train, X_test]

    for data in dflist:
        

            """
            "Aggregate and transform questionnaire features
            """


            diagnose_cols = ['TI_bip1','TI_bip2','TI_MDE','TI_dyst','TI_MDE_vr','TI_MDE_tr','TI_HYP_vr',
                            'TI_MAN_vr','TI_medik']
            data["n_diagnoses"] = data[diagnose_cols].sum(axis=1)

            pre_bdi_columns = ['PRE_bdi1','PRE_bdi2','PRE_bdi3',
                    'PRE_bdi4','PRE_bdi5','PRE_bdi6','PRE_bdi7','PRE_bdi8','PRE_bdi9','PRE_bdi10','PRE_bdi11','PRE_bdi12',
                    'PRE_bdi13','PRE_bdi14','PRE_bdi15','PRE_bdi16','PRE_bdi17','PRE_bdi18','PRE_bdi19','PRE_bdi20',
                    'PRE_bdi21']
            data[pre_bdi_columns] = data[pre_bdi_columns].apply(pd.to_numeric, errors='coerce').astype('Int64')
            data["PRE_bdi_sum"] = data[pre_bdi_columns].sum(axis=1).astype('Int64')


            if ml_options["include_early_change"] == 1:
                m1_phq_columns = ['M1_phqD1','M1_phqD2','M1_phqD3','M1_phqD4','M1_phqD5','M1_phqD6','M1_phqD7','M1_phqD8','M1_phqD9']
                data[m1_phq_columns] = data[m1_phq_columns].apply(pd.to_numeric, errors='coerce').astype('Int64')
                data["phq_m1_sum"] = data[m1_phq_columns].sum(axis=1)
                data["phq_early_change"] = data["outcome_sum_pre"] - data["phq_m1_sum"]
                data.drop(m1_phq_columns, axis=1, inplace=True)
                data.drop("phq_m1_sum", axis=1, inplace=True)

            phq_s_columns = ['PRE_phqS1','PRE_phqS2','PRE_phqS3','PRE_phqS4','PRE_phqS5','PRE_phqS6','PRE_phqS7',
                    'PRE_phqS8','PRE_phqS9', 'PRE_phqS10']

            data[phq_s_columns] = data[phq_s_columns].apply(pd.to_numeric, errors='coerce').astype('Int64')
            data["phq_s_sum"] = data[phq_s_columns].sum(axis=1)
            
            eurohis_columns = ['PRE_eurohis1','PRE_eurohis2','PRE_eurohis3','PRE_eurohis4','PRE_eurohis5','PRE_eurohis6',
                        'PRE_eurohis7','PRE_eurohis8']
            data[eurohis_columns] = data[eurohis_columns].apply(pd.to_numeric, errors='coerce').astype('Int64')
            data["eurohis_sum"] = data[eurohis_columns].sum(axis=1)

            imet_columns = ['PRE_imet1','PRE_imet2','PRE_imet3','PRE_imet4','PRE_imet5','PRE_imet6','PRE_imet7',
                        'PRE_imet8','PRE_imet9','PRE_imet10']
            data[imet_columns] = data[imet_columns].apply(pd.to_numeric, errors='coerce')
            data['PRE_imet10'] = data['PRE_imet10'].map({0:10, 1:9, 2:8, 3:7, 4:6, 5:5, 6:4, 7:3, 8:2, 9:1, 10:0})
            data["imet_sum"] = data[imet_columns].sum(axis=1)

            gad_columns= ['PRE_gad1','PRE_gad2','PRE_gad3','PRE_gad4','PRE_gad5','PRE_gad6','PRE_gad7']
            data[gad_columns] = data[gad_columns].apply(pd.to_numeric, errors='coerce').astype('Int64')
            data["gad_sum"] = data[gad_columns].sum(axis=1)

            costa_columns = ['PRE_costa1', 'PRE_costa2', 'PRE_costa3','PRE_costa4', 'PRE_costa5', 'PRE_costa6', 'PRE_costa7', 
                        'PRE_costa8', 'PRE_costa9', 'PRE_costa10', 'PRE_costa11','PRE_costa12', 'PRE_costa13', 'PRE_costa14',
                        'PRE_costa15', 'PRE_costa16', 'PRE_costa17', 'PRE_costa18','PRE_costa19', 'PRE_costa20', 'PRE_costa21']
            
            data[costa_columns] = data[costa_columns].apply(pd.to_numeric, errors='coerce').astype('Int64')
            data["costa_sum"] = data[costa_columns].sum(axis=1).astype('Int64')

            pathev_columns = ['PRE_pathev1', 'PRE_pathev2', 'PRE_pathev3', 'PRE_pathev4','PRE_pathev5', 'PRE_pathev6', 
                        'PRE_pathev7', 'PRE_pathev8', 'PRE_pathev9', 'PRE_pathev10']
            data[pathev_columns] = data[pathev_columns].apply(pd.to_numeric, errors='coerce').astype('Int64')
            invert_columns = ['PRE_pathev1', 'PRE_pathev3','PRE_pathev5','PRE_pathev7','PRE_pathev9', 'PRE_pathev10']

            for column in invert_columns:
                data[column] = data[column].map({1:5, 2:4, 3:3, 4:2, 5:1})
            
            data["pathev_sum"] = data[pathev_columns].sum(axis=1).astype('Int64')

            euheals_columns = ['PRE_euheals1','PRE_euheals2','PRE_euheals3']
            data[euheals_columns] = data[euheals_columns].apply(pd.to_numeric, errors='coerce').astype('Int64')
            data["euheals_sum"] = data[euheals_columns].sum(axis=1).astype('Int64')

            ipqr_columns = ['PRE_ipqr1','PRE_ipqr2','PRE_ipqr3','PRE_ipqr4','PRE_ipqr5',
                    'PRE_ipqr6','PRE_ipqr7','PRE_ipqr8','PRE_ipqr9','PRE_ipqr10','PRE_ipqr11','PRE_ipqr12','PRE_ipqr13',
                    'PRE_ipqr14','PRE_ipqr15','PRE_ipqr16','PRE_ipqr17','PRE_ipqr18']
                    
            data[ipqr_columns] = data[ipqr_columns].apply(pd.to_numeric, errors='coerce')
            data['PRE_ipqr8'] = data['PRE_ipqr8'].map({1:5, 2:4, 3:3, 4:2, 5:1})
            data["ipqr_sum"] = data[ipqr_columns].sum(axis=1).astype('Int64')

            gpse_columns = ['PRE_gpse1','PRE_gpse2','PRE_gpse3','PRE_gpse4','PRE_gpse5','PRE_gpse6',
                    'PRE_gpse7','PRE_gpse8','PRE_gpse9','PRE_gpse10']
            
            data[gpse_columns] = data[gpse_columns].apply(pd.to_numeric, errors='coerce')
            data["gpse_sum"] = data[gpse_columns].sum(axis=1).astype('Int64')

            bsss_columns = ['PRE_bsss1','PRE_bsss2','PRE_bsss3','PRE_bsss4','PRE_bsss5','PRE_bsss6','PRE_bsss7','PRE_bsss8',
                        'PRE_bsss9','PRE_bsss10','PRE_bsss11','PRE_bsss12','PRE_bsss13']
            data[bsss_columns] = data[bsss_columns].apply(pd.to_numeric, errors='coerce').astype('Int64')
            data["bsss_perceived"] = data[['PRE_bsss1','PRE_bsss2','PRE_bsss3','PRE_bsss4','PRE_bsss5','PRE_bsss6','PRE_bsss7',
                                'PRE_bsss8']].sum(axis=1).astype('Int64')
            data["bsss_suppseek"] = data[['PRE_bsss9','PRE_bsss10','PRE_bsss11','PRE_bsss12','PRE_bsss13']].sum(axis=1).astype('Int64')

            pvq_columns = ['PRE_pvq1','PRE_pvq2','PRE_pvq3','PRE_pvq4','PRE_pvq5','PRE_pvq6','PRE_pvq7','PRE_pvq8','PRE_pvq9',
                    'PRE_pvq10','PRE_pvq11','PRE_pvq12','PRE_pvq13','PRE_pvq14','PRE_pvq15','PRE_pvq16','PRE_pvq17',
                    'PRE_pvq18','PRE_pvq19','PRE_pvq20','PRE_pvq21']
            data[pvq_columns] = data[pvq_columns].apply(pd.to_numeric, errors ='coerce').astype('Int64')


            data['PRE_height'] = data['PRE_height']/100
            data["bmi_score"] = data['PRE_weight']/ (data['PRE_height']*data['PRE_height']) 
            data.drop(['PRE_weight', 'PRE_height'], axis=1, inplace=True)


    return X_train, X_test, y_train, y_test