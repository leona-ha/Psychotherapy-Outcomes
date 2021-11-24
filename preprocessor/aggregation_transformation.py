
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

    X_train_cp = X_train.copy()
    X_test_cp = X_test.copy()
    dflist = [X_train_cp, X_test_cp]

    for data in dflist:
        
            """
            "Aggregate and transform questionnaire features
            """

            pre_bdi_columns = ['PRE_bdi1','PRE_bdi2','PRE_bdi3',
                    'PRE_bdi4','PRE_bdi5','PRE_bdi6','PRE_bdi7','PRE_bdi8','PRE_bdi9','PRE_bdi10','PRE_bdi11','PRE_bdi12',
                    'PRE_bdi13','PRE_bdi14','PRE_bdi15','PRE_bdi16','PRE_bdi17','PRE_bdi18','PRE_bdi19','PRE_bdi20',
                    'PRE_bdi21']
            data[pre_bdi_columns] = data[pre_bdi_columns].apply(pd.to_numeric, errors='coerce').astype('Int64')
            data["PRE_bdi_sum"] = data[pre_bdi_columns].sum(axis=1).astype('Int64')


            if ml_options["include_early_change"] in (1,2,3):
                ec = ml_options["include_early_change"]
                early_phq_columns = [f'M{ec}_phqD1',f'M{ec}_phqD2',f'M{ec}_phqD3',f'M{ec}_phqD4',f'M{ec}_phqD5',f'M{ec}_phqD6',f'M{ec}_phqD7',f'M{ec}_phqD8',f'M{ec}_phqD9']
                data[early_phq_columns] = data[early_phq_columns].apply(pd.to_numeric, errors='coerce').astype('Int64')
                data["phq_early_sum"] = data[early_phq_columns].sum(axis=1)
                data["phq_early_change"] = data["outcome_sum_pre"] - data["phq_early_sum"]
                #data.drop(['M3_phqD1', 'M3_phqD2', 'M3_phqD3', 'M3_phqD4', 'M3_phqD5', 'M3_phqD6',
               #'M3_phqD7', 'M3_phqD8', 'M3_phqD9', 'M4_phqD1','M4_phqD2','M4_phqD3','M4_phqD4','M4_phqD5','M4_phqD6',
               #'M4_phqD7','M4_phqD8','M4_phqD9'], axis=1, inplace=True)
                #data.drop("phq_early_sum", axis=1, inplace=True)

                if ml_options["include_costa_sewip"] ==1:
                    M3_sewip_cols = ['M3_sewip1','M3_sewip2','M3_sewip3','M3_sewip4','M3_sewip5','M3_sewip6','M3_sewip7','M3_sewip8',
                    'M3_sewip9','M3_sewip10','M3_sewip11','M3_sewip12','M3_sewip13','M3_sewip14','M3_sewip15','M3_sewip16','M3_sewip17',
                    'M3_sewip18','M3_sewip19','M3_sewip20','M3_sewip21']

                    M3_costa_cols = ['M3_costa1','M3_costa2','M3_costa5','M3_costa6','M3_costa8',
                    'M3_costa10','M3_costa11','M3_costa12','M3_costa13','M3_costa14','M3_costa15',
                    'M3_costa18']

                    data[M3_sewip_cols] = data[M3_sewip_cols].apply(pd.to_numeric, errors='coerce').astype('Int64')
                    data[M3_costa_cols] = data[M3_costa_cols].apply(pd.to_numeric, errors='coerce').astype('Int64')

                    data["M3_costa_sum"] = data[M3_costa_cols].sum(axis=1).astype('Int64')
                    data["M3_sewip_sum"] = data[M3_sewip_cols].sum(axis=1).astype('Int64')

                    ##data["sewip_emo"] = data[['M3_sewip1', 'M3_sewip8', 'M3_sewip15']].sum(axis=1).astype('Int64')
                    #data["sewip_prob"] = data[['M3_sewip2', 'M3_sewip9', 'M3_sewip16']].sum(axis=1).astype('Int64')
                    #data["sewip_res"] = data[['M3_sewip3', 'M3_sewip10', 'M3_sewip17']].sum(axis=1).astype('Int64')
                    #data["sewip_mean"] = data[['M3_sewip4', 'M3_sewip11', 'M3_sewip18']].sum(axis=1).astype('Int64')
                    #data["sewip_collab"] = data[['M3_sewip5', 'M3_sewip6', 'M3_sewip12', 'M3_sewip13', 'M3_sewip19', 'M3_sewip20']].sum(axis=1).astype('Int64')
                    #data["sewip_mast"] = data[['M3_sewip7', 'M3_sewip14', 'M3_sewip21']].sum(axis=1).astype('Int64')

                    #data.drop(M3_sewip_cols, axis=1, inplace=True)



            elif ml_options["include_early_change"] == 0:
                data.drop(['M3_phqD1', 'M3_phqD2', 'M3_phqD3', 'M3_phqD4', 'M3_phqD5', 'M3_phqD6',
               'M3_phqD7', 'M3_phqD8', 'M3_phqD9', 
               'M3_sewip1','M3_sewip2','M3_sewip3','M3_sewip4','M3_sewip5','M3_sewip6','M3_sewip7','M3_sewip8',
                    'M3_sewip9','M3_sewip10','M3_sewip11','M3_sewip12','M3_sewip13','M3_sewip14','M3_sewip15','M3_sewip16','M3_sewip17',
                    'M3_sewip18','M3_sewip19','M3_sewip20','M3_sewip21',
                'M3_costa1','M3_costa2','M3_costa5','M3_costa6','M3_costa8',
                    'M3_costa10','M3_costa11','M3_costa12','M3_costa13','M3_costa14','M3_costa15',
                    'M3_costa18'], axis=1, inplace=True)
        

            phq_s_columns = ['PRE_phqS1','PRE_phqS2','PRE_phqS3','PRE_phqS4','PRE_phqS5','PRE_phqS6','PRE_phqS7',
                    'PRE_phqS8','PRE_phqS9', 'PRE_phqS10']

            data[phq_s_columns] = data[phq_s_columns].apply(pd.to_numeric, errors='coerce').astype('Int64')
            data["phq_s_sum"] = data[phq_s_columns].sum(axis=1)
            
            eurohis_columns = ['PRE_eurohis1','PRE_eurohis2','PRE_eurohis3','PRE_eurohis4','PRE_eurohis5','PRE_eurohis6',
                        'PRE_eurohis7','PRE_eurohis8']
            data[eurohis_columns] = data[eurohis_columns].apply(pd.to_numeric, errors='coerce').astype('Int64')
            data["eurohis_sum"] = data[eurohis_columns].sum(axis=1)

            imet_columns = ['PRE_imet1','PRE_imet2','PRE_imet3','PRE_imet4','PRE_imet5','PRE_imet6','PRE_imet7',
                        'PRE_imet8','PRE_imet10']
            data[imet_columns] = data[imet_columns].apply(pd.to_numeric, errors='coerce')
            data["imet_sum"] = data[imet_columns].sum(axis=1)

            gad_columns= ['PRE_gad1','PRE_gad2','PRE_gad3','PRE_gad4','PRE_gad5','PRE_gad6','PRE_gad7']
            data[gad_columns] = data[gad_columns].apply(pd.to_numeric, errors='coerce').astype('Int64')
            data["gad_sum"] = data[gad_columns].sum(axis=1)

            costa_columns = ['PRE_costa1', 'PRE_costa2', 'PRE_costa5', 'PRE_costa6',
                        'PRE_costa8', 'PRE_costa10', 'PRE_costa11','PRE_costa12', 'PRE_costa13', 'PRE_costa14',
                        'PRE_costa15', 'PRE_costa18']
            
            data[costa_columns] = data[costa_columns].apply(pd.to_numeric, errors='coerce').astype('Int64')
            data["costa_sum"] = data[costa_columns].sum(axis=1).astype('Int64')

            pathev_columns = ['PRE_pathev1', 'PRE_pathev2', 'PRE_pathev3', 'PRE_pathev4','PRE_pathev5', 'PRE_pathev6', 
                        'PRE_pathev7', 'PRE_pathev8', 'PRE_pathev9', 'PRE_pathev10']
            data[pathev_columns] = data[pathev_columns].apply(pd.to_numeric, errors='coerce').astype('Int64')
            invert_columns = ['PRE_pathev1', 'PRE_pathev5','PRE_pathev9', 'PRE_pathev10']

            for column in invert_columns:
                data[column] = data[column].map({1:5, 2:4, 3:3, 4:2, 5:1})
            
            data["pathev_zuv"] = data[['PRE_pathev1','PRE_pathev4', 'PRE_pathev5', 'PRE_pathev9']].sum(axis=1).astype('Int64')
            data["pathev_fur"] = data[['PRE_pathev3', 'PRE_pathev7']].sum(axis=1).astype('Int64')
            data["pathev_pas"] = data[['PRE_pathev2', 'PRE_pathev6', 'PRE_pathev8', 'PRE_pathev10']].sum(axis=1).astype('Int64')


            ipqr_columns = ['PRE_ipqr1','PRE_ipqr2','PRE_ipqr3','PRE_ipqr4','PRE_ipqr5',
                    'PRE_ipqr6','PRE_ipqr7','PRE_ipqr8','PRE_ipqr9','PRE_ipqr10','PRE_ipqr11','PRE_ipqr12','PRE_ipqr13',
                    'PRE_ipqr14','PRE_ipqr15','PRE_ipqr16','PRE_ipqr17','PRE_ipqr18']
                    
            data[ipqr_columns] = data[ipqr_columns].apply(pd.to_numeric, errors='coerce')
            data['PRE_ipqr7'] = data['PRE_ipqr7'].map({1:5, 2:4, 3:3, 4:2, 5:1})
            data['PRE_ipqr9'] = data['PRE_ipqr9'].map({1:5, 2:4, 3:3, 4:2, 5:1})
            data['PRE_ipqr10'] = data['PRE_ipqr10'].map({1:5, 2:4, 3:3, 4:2, 5:1})
            data['PRE_ipqr11'] = data['PRE_ipqr11'].map({1:5, 2:4, 3:3, 4:2, 5:1})
            data['PRE_ipqr12'] = data['PRE_ipqr12'].map({1:5, 2:4, 3:3, 4:2, 5:1})

            data["ipqr_timeline"] = data[['PRE_ipqr1','PRE_ipqr2','PRE_ipqr3']].sum(axis=1).astype('Int64')
            data["ipqr_cons"] = data[['PRE_ipqr4','PRE_ipqr5','PRE_ipqr6']].sum(axis=1).astype('Int64')
            data["ipqr_perscont"] = data[['PRE_ipqr7','PRE_ipqr8','PRE_ipqr9']].sum(axis=1).astype('Int64')
            data["ipqr_illcoher"] = data[['PRE_ipqr10','PRE_ipqr11','PRE_ipqr12']].sum(axis=1).astype('Int64')
            data["ipqr_timecycl"] = data[['PRE_ipqr13','PRE_ipqr14','PRE_ipqr15']].sum(axis=1).astype('Int64')
            data["ipqr_emotrep"] = data[['PRE_ipqr16','PRE_ipqr17','PRE_ipqr18']].sum(axis=1).astype('Int64')


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

            


    X_train = X_train_cp
    X_test = X_test_cp
    return X_train, X_test, y_train, y_test
