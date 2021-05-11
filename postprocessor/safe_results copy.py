import csv
import os
import numpy as np
from config import STANDARDPATH
import matplotlib.pyplot as plt

"""
"Safe aggregated metrics 
"""
def aggregate_metrics(ml_options, input_list, X_train=None):
    
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
        
        
        savepath = os.path.join(STANDARDPATH, 'outcomes_aggregated_rf.csv')
        number_rounds = len(accuracy_flat)
        if not os.path.exists(savepath):
            header = ['model', 'n_iterations', 'accuracy_min', 'accuracy_max', 'accuracy_mean', 'accuracy_std', 'accuracy_class0_min', \
                'accuracy_class0_max', 'accuracy_class0_mean', 'accuracy_class0_std', 'accuracy_class1_min','accuracy_class1_max', \
                'accuracy_class1_mean', 'accuracy_class1_std', 'precision_min', 'precision_max', 'precision_mean', 'precision_std', \
                    'f1_score_min', 'f1_score_max', 'f1_score_mean', 'f1_score_std', \
                        'balanced_accuracy_min', 'balanced_accuracy_max', 'balanced_accuracy_mean', \
                'balanced_accuracy_std', 'oob_accuracy_min', 'oob_accuracy_max', 'oob_accuracy_mean', 'oob_accuracy_std', \
                'log_loss_value_min', 'log_loss_value_max','log_loss_value_mean', 'log_loss_value_std', 'feature_importances_min', 
                'feature_importances_max', 'feature_importances_mean', 'feature_importances_std']

            with open(savepath, 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                # write the header
                writer.writerow(header)
                # write outcome rows
                writer.writerow([ml_options["model_name"],number_rounds, accuracy_min, accuracy_max, accuracy_mean, accuracy_std, accuracy_class0_min, \
                accuracy_class0_max, accuracy_class0_mean, accuracy_class0_std, accuracy_class1_min, accuracy_class1_max, accuracy_class1_mean, \
                    accuracy_class1_std, precision_min, precision_max,precision_mean, precision_std, \
                    f1_score_min, f1_score_max, f1_score_mean, f1_score_std, balanced_accuracy_min, balanced_accuracy_max, balanced_accuracy_mean, \
                        balanced_accuracy_std, oob_accuracy_min, \
                    oob_accuracy_max, oob_accuracy_mean, oob_accuracy_std, log_loss_value_min, log_loss_value_max, log_loss_value_mean, log_loss_value_std, \
                    feature_importances_min, feature_importances_max, feature_importances_mean, feature_importances_std])
        else:
            with open(savepath, 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([ml_options["model_name"],number_rounds, accuracy_min, accuracy_max, accuracy_mean, accuracy_std, accuracy_class0_min, \
                accuracy_class0_max, accuracy_class0_mean, accuracy_class0_std, accuracy_class1_min, accuracy_class1_max, accuracy_class1_mean, \
                    accuracy_class1_std, precision_min, precision_max,precision_mean, precision_std, \
                    f1_score_min, f1_score_max, f1_score_mean, f1_score_std, balanced_accuracy_min, balanced_accuracy_max, balanced_accuracy_mean, \
                        balanced_accuracy_std, oob_accuracy_min, \
                    oob_accuracy_max, oob_accuracy_mean, oob_accuracy_std, log_loss_value_min, log_loss_value_max, log_loss_value_mean, log_loss_value_std, \
                    feature_importances_min, feature_importances_max, feature_importances_mean, feature_importances_std])
        
        print('Number of Rounds: ' + str(number_rounds) + 
                '\nMin Accuracy: ' + str(accuracy_min) + '\nMax Accuracy: ' + str(accuracy_max) + '\nMean Accuracy: ' + str(accuracy_mean) + '\nStd Accuracy: ' + str(accuracy_std) +
                '\nMin Accuracy_class_0: ' + str(accuracy_class0_min) + '\nMax Accuracy_class_0: ' + str(accuracy_class0_max) + '\nMean Accuracy_class_0: ' + str(accuracy_class0_mean) + '\nStd Accuracy_class_0: ' + str(accuracy_class0_std) +
                '\nMin Accuracy_class_1: ' + str(accuracy_class1_min) + '\nMax Accuracy_class_1: ' + str(accuracy_class1_max) + '\nMean Accuracy_class_1: ' + str(accuracy_class1_mean) + '\nStd Accuracy_class_1: ' + str(accuracy_class1_std) +
                '\nMin Balanced_Accuracy: ' + str(balanced_accuracy_min) + '\nMax Balanced_Accuracy: ' + str(balanced_accuracy_max) + '\nMean Balanced_Accuracy: ' + str(balanced_accuracy_mean) + '\nStd Balanced_Accuracy: ' + str(balanced_accuracy_std) +
                '\nMin OOB_Accuracy: ' + str(oob_accuracy_min) + '\nMax OOB_Accuracy: ' + str(oob_accuracy_max) + '\nMean OOB_Accuracy: ' + str(oob_accuracy_mean) + '\nStd OOB_Accuracy: ' + str(oob_accuracy_std) +
                '\nMin Log-Loss: ' + str(log_loss_value_min) + '\nMax Log-Loss: ' + str(log_loss_value_max) + '\nMean Log-Loss: ' + str(log_loss_value_mean) + '\nStd Log-Loss: ' + str(log_loss_value_std)+
                '\nMean Precision:' + str(precision_mean)+ '\nMean F1:' + str(f1_score_mean))


#