import csv
import os
import numpy as np
from config import STANDARDPATH, OUTCOME_PATH
import matplotlib.pyplot as plt
from scipy.stats import t
from math import sqrt
from statistics import stdev

"""
"Safe aggregated baseline metrics and compare models
"""


def corrected_dependent_ttest(data1, data2, X_train, X_test, alpha=0.05):
    n = len(data1)
    differences = [(data1[i]-data2[i]) for i in range(n)]
    sd = stdev(differences)
    divisor = 1 / n * sum(differences)
    test_training_ratio = len(X_test) / len(X_train) 
    denominator = sqrt(1 / n + test_training_ratio) * sd
    t_stat = divisor / denominator
    # degrees of freedom
    df = n - 1
    # calculate the critical value
    cv = t.ppf(1.0 - alpha, df)
    # calculate the p-value
    p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
    # return everything
    return t_stat, df, cv, p

def run(ml_options, input_list, X_train, X_test, model_flatlists):
    
    accuracy_flat = []
    accuracy_class1_flat = []
    accuracy_class0_flat = []
    precision_flat = []
    f1_score_flat = []
    balanced_accuracy_flat = []
    

    for counter, sublist in enumerate(input_list):
        for itemnumber in range(len(sublist)):
            if itemnumber == 0:
                accuracy_flat.append(sublist[itemnumber])
            elif itemnumber == 1:
                accuracy_class1_flat.append(sublist[itemnumber])
            elif itemnumber == 2:
                accuracy_class0_flat.append(sublist[itemnumber])
            elif itemnumber == 3:
                precision_flat.append(sublist[itemnumber])
            elif itemnumber == 4:
                f1_score_flat.append(sublist[itemnumber])
            elif itemnumber == 5:
                balanced_accuracy_flat.append(sublist[itemnumber])
           

    accuracy_min = min(accuracy_flat)
    accuracy_max = max(accuracy_flat)
    accuracy_mean = np.mean(accuracy_flat)
    accuracy_std = np.std(accuracy_flat)
    accuracy_class0_min = min(accuracy_class0_flat)
    accuracy_class0_max = max(accuracy_class0_flat)
    accuracy_class0_mean = np.mean(accuracy_class0_flat)
    accuracy_class0_std = np.std(accuracy_class0_flat)
    accuracy_class1_min = min(accuracy_class1_flat)
    accuracy_class1_max = max(accuracy_class1_flat)
    accuracy_class1_mean = np.mean(accuracy_class1_flat)
    accuracy_class1_std = np.std(accuracy_class1_flat)
    precision_min = min(precision_flat)
    precision_max = max(precision_flat)
    precision_mean = np.mean(precision_flat)
    precision_std = np.std(precision_flat)
    f1_score_min = min(f1_score_flat)
    f1_score_max = max(f1_score_flat)
    f1_score_mean = np.mean(f1_score_flat)
    f1_score_std = np.std(f1_score_flat)
    balanced_accuracy_min = min(balanced_accuracy_flat)
    balanced_accuracy_max = max(balanced_accuracy_flat)
    balanced_accuracy_mean = np.mean(balanced_accuracy_flat)
    balanced_accuracy_std = np.std(balanced_accuracy_flat)

    _,_,_, p_accuracy = corrected_dependent_ttest(model_flatlists[0], accuracy_flat, X_train, X_test)

    _,_,_,p_accuracy_class1 = corrected_dependent_ttest(model_flatlists[1], accuracy_class1_flat, X_train, X_test)

    _,_,_,p_accuracy_class0 = corrected_dependent_ttest(model_flatlists[2], accuracy_class0_flat, X_train, X_test)

    _,_,_,p_precision = corrected_dependent_ttest(model_flatlists[3], precision_flat, X_train, X_test)

    _,_,_,p_f1_score = corrected_dependent_ttest(model_flatlists[4], f1_score_flat, X_train, X_test)

    _,_,_,p_balanced_acc = corrected_dependent_ttest(model_flatlists[5], balanced_accuracy_flat, X_train, X_test)
       
         
    savepath = os.path.join(OUTCOME_PATH, 'model_comparison.csv')
    number_rounds = len(accuracy_flat)
    if not os.path.exists(savepath):
        header = ['model_to_compare', 'n_iterations','p_accuracy', 'p_accuracy_class0', 'p_accuracy_class1',
                'p_precision', 'p_f1_score', 'p_balanced_acc',
             'accuracy_mean', 'accuracy_std', 'accuracy_class0_mean', 'accuracy_class0_std', 'accuracy_class1_mean','accuracy_class1_std', \
                'precision_mean', 'precision_std','f1_score_mean', 'f1_score_std', \
                 'balanced_accuracy_mean', 'balanced_accuracy_std']

        with open(savepath, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(header)
            # write outcome rows
            writer.writerow([ml_options["model_name"],number_rounds, p_accuracy, p_accuracy_class0, p_accuracy_class1, p_precision, p_f1_score, p_balanced_acc, \
                accuracy_mean, accuracy_std, accuracy_class0_mean, accuracy_class0_std, accuracy_class1_mean, \
                    accuracy_class1_std, precision_mean, precision_std,f1_score_mean, f1_score_std,  balanced_accuracy_mean, \
                        balanced_accuracy_std])
    else:
        with open(savepath, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ml_options["model_name"],number_rounds, \
                p_accuracy, p_accuracy_class0, p_accuracy_class1, p_precision, p_f1_score, p_balanced_acc, \
                accuracy_mean, accuracy_std, accuracy_class0_mean, accuracy_class0_std, accuracy_class1_mean, \
                    accuracy_class1_std, precision_mean, precision_std,f1_score_mean, f1_score_std,  balanced_accuracy_mean, \
                        balanced_accuracy_std])


#