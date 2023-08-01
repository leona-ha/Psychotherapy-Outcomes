import csv
import os
from config import OUTCOME_PATH, DATAPATH_OUT


"""
Aggregate benchmark metrics for ROC plots
"""
def aggregate_metrics(ml_options, input_list, X_train=None, X_test=None):

    fpr_flat = []
    tpr_flat = []
    tprs_flat = []
    roc_auc_flat = []

    for counter, sublist in enumerate(input_list):
        for itemnumber in range(len(sublist)):
                    
            if itemnumber == 6:
                roc_auc_flat.append(sublist[itemnumber])
            if itemnumber == 7:
                fpr_flat.append(sublist[itemnumber])
            if itemnumber == 8:
                tpr_flat.append(sublist[itemnumber])
            if itemnumber == 9:
                tprs_flat.append(sublist[itemnumber])
                

        
       
    model_flatlists = [roc_auc_flat, fpr_flat, tpr_flat, tprs_flat]
    return model_flatlists

