import csv
import os
import numpy as np
from config import STANDARDPATH, OUTCOME_PATH
import matplotlib.pyplot as plt

IMG_SAFEPATH = os.path.join(OUTCOME_PATH, "plots")
if not os.path.exists(IMG_SAFEPATH):
    os.makedirs(IMG_SAFEPATH)

"""
"Safe aggregated metrics 
"""
def aggregate_metrics(ml_options, input_list, X_train=None, X_test=None):

    savepath = os.path.join(OUTCOME_PATH, f'outcomes_aggregated_{ml_options["model_architecture"]}.csv')

    accuracy_flat = []
    accuracy_class1_flat = []
    accuracy_class0_flat = []
    precision_flat = []
    f1_score_flat = []
    balanced_accuracy_flat = []
    oob_accuracy_flat = []
    log_loss_value_flat = []
    fpr_flat = []
    tpr_flat = []
    tprs_flat = []
    roc_auc_flat = []
    fraction_positives_flat = []
    mean_predicted_value_flat = []
    counter_features_selected_flat = []

    if ml_options["model_architecture"] == "RF": 
        feature_importances_flat = np.zeros((len(input_list),len(input_list[0][8])))
        feature_importances_count_flat = np.zeros((len(input_list),len(input_list[0][8])))
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
                elif itemnumber == 6:
                    oob_accuracy_flat.append(sublist[itemnumber])
                elif itemnumber == 7:
                    log_loss_value_flat.append(sublist[itemnumber])
                elif itemnumber == 8:
                    feature_importances_flat[counter,:] = sublist[itemnumber]
                elif itemnumber == 9:
                    fpr_flat.append(sublist[itemnumber])
                elif itemnumber == 10:
                    tpr_flat.append(sublist[itemnumber])
                elif itemnumber == 11:
                    tprs_flat.append(sublist[itemnumber])
                elif itemnumber == 12:
                    roc_auc_flat.append(sublist[itemnumber])
                elif itemnumber == 13:
                    fraction_positives_flat.append(sublist[itemnumber])
                elif itemnumber == 14:
                    mean_predicted_value_flat.append(sublist[itemnumber])
                elif itemnumber == 15:
                    counter_features_selected_flat.append(sublist[itemnumber])
                elif itemnumber == 16:
                    feature_importances_count_flat[counter,:] = sublist[itemnumber]
                

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
        log_loss_value_min = min(log_loss_value_flat)
        log_loss_value_max = max(log_loss_value_flat)
        log_loss_value_mean = np.mean(log_loss_value_flat)
        log_loss_value_std = np.std(log_loss_value_flat)
        oob_accuracy_min = min(oob_accuracy_flat)
        oob_accuracy_max = max(oob_accuracy_flat)
        oob_accuracy_mean = np.mean(oob_accuracy_flat)
        oob_accuracy_std = np.std(oob_accuracy_flat)
        feature_importances_min = feature_importances_flat.min(axis=0).reshape(1,len(input_list[0][8]))
        feature_importances_max = feature_importances_flat.max(axis=0).reshape(1,len(input_list[0][8]))
        feature_importances_mean = feature_importances_flat.mean(axis=0).reshape(1,len(input_list[0][8]))
        feature_importances_std = feature_importances_flat.std(axis=0).reshape(1,len(input_list[0][8]))
        counter_features_selected_min = min(counter_features_selected_flat)
        counter_features_selected_max = max(counter_features_selected_flat)
        counter_features_selected_mean = np.mean(counter_features_selected_flat)
        counter_features_selected_std = np.std(counter_features_selected_flat)
        roc_auc_min = min(roc_auc_flat)
        roc_auc_max = max(roc_auc_flat)
        roc_auc_mean = np.mean(roc_auc_flat)
        roc_auc_std = np.std(roc_auc_flat)
        feature_importances_count_sum = feature_importances_count_flat.sum(axis=0).reshape(1,len(input_list[0][8]))
    

        
        number_rounds = len(accuracy_flat)
        if not os.path.exists(savepath):
            header = ['model', 'n_iterations', 'counter_features_selected_mean', 'counter_features_selected_std','counter_features_selected_min','counter_features_selected_max', \
            'accuracy_min', 'accuracy_max', 'accuracy_mean', 'accuracy_std', 'accuracy_class0_min', \
                'accuracy_class0_max', 'accuracy_class0_mean', 'accuracy_class0_std', 'accuracy_class1_min','accuracy_class1_max', \
                'accuracy_class1_mean', 'accuracy_class1_std', 'precision_min', 'precision_max', 'precision_mean', 'precision_std', \
                    'f1_score_min', 'f1_score_max', 'f1_score_mean', 'f1_score_std', \
                        'balanced_accuracy_min', 'balanced_accuracy_max', 'balanced_accuracy_mean', \
                'balanced_accuracy_std', 'oob_accuracy_min', 'oob_accuracy_max', 'oob_accuracy_mean', 'oob_accuracy_std', \
                'log_loss_value_min', 'log_loss_value_max','log_loss_value_mean', 'log_loss_value_std', 'feature_importances_min', 
                'feature_importances_max', 'feature_importances_mean', 'feature_importances_std',
                'roc_auc_min', 'roc_auc_max', 'roc_auc_mean', 'roc_auc_std']

        
            
            with open(savepath, 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerow([ml_options["model_name"],number_rounds, 
                counter_features_selected_mean, counter_features_selected_std,counter_features_selected_min,counter_features_selected_max, \
                accuracy_min, accuracy_max, accuracy_mean, accuracy_std, accuracy_class0_min, \
                accuracy_class0_max, accuracy_class0_mean, accuracy_class0_std, accuracy_class1_min, accuracy_class1_max, accuracy_class1_mean, \
                    accuracy_class1_std, precision_min, precision_max,precision_mean, precision_std, \
                    f1_score_min, f1_score_max, f1_score_mean, f1_score_std, balanced_accuracy_min, balanced_accuracy_max, balanced_accuracy_mean, \
                        balanced_accuracy_std, oob_accuracy_min, \
                    oob_accuracy_max, oob_accuracy_mean, oob_accuracy_std, log_loss_value_min, log_loss_value_max, log_loss_value_mean, log_loss_value_std, \
                    feature_importances_min, feature_importances_max, feature_importances_mean, feature_importances_std, \
                     roc_auc_min, roc_auc_max, roc_auc_mean, roc_auc_std])

        else:  
            with open(savepath, 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([ml_options["model_name"],number_rounds, \
                counter_features_selected_mean, counter_features_selected_std,counter_features_selected_min,counter_features_selected_max, \
                accuracy_min, accuracy_max, accuracy_mean, accuracy_std, accuracy_class0_min, \
                accuracy_class0_max, accuracy_class0_mean, accuracy_class0_std, accuracy_class1_min, accuracy_class1_max, accuracy_class1_mean, \
                    accuracy_class1_std, precision_min, precision_max,precision_mean, precision_std, \
                    f1_score_min, f1_score_max, f1_score_mean, f1_score_std, balanced_accuracy_min, balanced_accuracy_max, balanced_accuracy_mean, \
                        balanced_accuracy_std, oob_accuracy_min, \
                    oob_accuracy_max, oob_accuracy_mean, oob_accuracy_std, log_loss_value_min, log_loss_value_max, log_loss_value_mean, log_loss_value_std, \
                    feature_importances_min, feature_importances_max, feature_importances_mean, feature_importances_std, \
                    roc_auc_min, roc_auc_max, roc_auc_mean, roc_auc_std])
        
        print('Number of Rounds: ' + str(number_rounds) + 'Mean Nr of Selected Features' + str(counter_features_selected_mean) + 
                '\nMin Accuracy: ' + str(accuracy_min) + '\nMax Accuracy: ' + str(accuracy_max) + '\nMean Accuracy: ' + str(accuracy_mean) + '\nStd Accuracy: ' + str(accuracy_std) +
                '\nMin Accuracy_class_0: ' + str(accuracy_class0_min) + '\nMax Accuracy_class_0: ' + str(accuracy_class0_max) + '\nMean Accuracy_class_0: ' + str(accuracy_class0_mean) + '\nStd Accuracy_class_0: ' + str(accuracy_class0_std) +
                '\nMin Accuracy_class_1: ' + str(accuracy_class1_min) + '\nMax Accuracy_class_1: ' + str(accuracy_class1_max) + '\nMean Accuracy_class_1: ' + str(accuracy_class1_mean) + '\nStd Accuracy_class_1: ' + str(accuracy_class1_std) +
                '\nMin Balanced_Accuracy: ' + str(balanced_accuracy_min) + '\nMax Balanced_Accuracy: ' + str(balanced_accuracy_max) + '\nMean Balanced_Accuracy: ' + str(balanced_accuracy_mean) + '\nStd Balanced_Accuracy: ' + str(balanced_accuracy_std) +
                '\nMin OOB_Accuracy: ' + str(oob_accuracy_min) + '\nMax OOB_Accuracy: ' + str(oob_accuracy_max) + '\nMean OOB_Accuracy: ' + str(oob_accuracy_mean) + '\nStd OOB_Accuracy: ' + str(oob_accuracy_std) +
                '\nMin Log-Loss: ' + str(log_loss_value_min) + '\nMax Log-Loss: ' + str(log_loss_value_max) + '\nMean Log-Loss: ' + str(log_loss_value_mean) + '\nStd Log-Loss: ' + str(log_loss_value_std)+
                '\nMean Precision:' + str(precision_mean)+ '\nMean F1:' + str(f1_score_mean))


        importances = feature_importances_mean.reshape((feature_importances_mean.shape[1],))
        indices = np.argsort(importances)[::-1]
        feature_names = list(X_train.columns)
        names = [feature_names[i] for i in indices]
        names = names[:20]
        plt.figure(figsize=(20,15))
        plt.title("Feature Importance")
        plt.bar(range(X_train.shape[1])[:20], importances[indices][:20])
        plt.xticks(range(20), names, rotation=90)
        img_safepath = os.path.join(IMG_SAFEPATH, f'{ml_options["model_name"]}_feature_importances.png')
        plt.savefig(img_safepath)

        importance_counts = feature_importances_count_sum.reshape((feature_importances_count_sum.shape[1],))
        indices = np.argsort(importance_counts)[::-1]
        feature_names = list(X_train.columns)
        names = [feature_names[i] for i in indices]
        names = names[:20]
        plt.figure(figsize=(20,15))
        plt.title("Feature Counts")
        plt.bar(range(X_train.shape[1])[:20], importance_counts[indices][:20])
        plt.xticks(range(20), names, rotation=90)
        plt.ylim(0, 100)
        img_safepath = os.path.join(IMG_SAFEPATH, f'{ml_options["model_name"]}_feature_counts.png')
        plt.savefig(img_safepath)

### Add input list!!

    elif ml_options["model_architecture"] == "SVM":
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
        balanced_accuracy_min = min(balanced_accuracy_flat)
        balanced_accuracy_max = max(balanced_accuracy_flat)
        balanced_accuracy_mean = np.mean(balanced_accuracy_flat)
        balanced_accuracy_std = np.std(balanced_accuracy_flat)
        feature_importances_min = feature_importances_flat.min(axis=0).reshape(1,len(input_list[0][6]))
        feature_importances_max = feature_importances_flat.max(axis=0).reshape(1,len(input_list[0][6]))
        feature_importances_mean = feature_importances_flat.mean(axis=0).reshape(1,len(input_list[0][6]))
        feature_importances_std = feature_importances_flat.std(axis=0).reshape(1,len(input_list[0][6]))
    
        number_rounds = len(accuracy_flat)

        if not os.path.exists(savepath):
            header = ['model', 'n_iterations', 'accuracy_min', 'accuracy_max', 'accuracy_mean', 'accuracy_std', 'accuracy_class0_min', \
                'accuracy_class0_max', 'accuracy_class0_mean', 'accuracy_class0_std', 'accuracy_class1_min','accuracy_class1_max', \
                'accuracy_class1_mean', 'accuracy_class1_std', 'balanced_accuracy_min', 'balanced_accuracy_max', 'balanced_accuracy_mean', \
                'balanced_accuracy_std', 'feature_importances_min', 'feature_importances_max', 'feature_importances_mean', 'feature_importances_std']


            with open(savepath, 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerow([ml_options["model_name"],number_rounds, accuracy_min, accuracy_max, accuracy_mean, accuracy_std, accuracy_class0_min, \
                accuracy_class0_max, accuracy_class0_mean, accuracy_class0_std, accuracy_class1_min, accuracy_class1_max, accuracy_class1_mean, \
                    accuracy_class1_std, balanced_accuracy_min, balanced_accuracy_max, balanced_accuracy_mean, balanced_accuracy_std,  \
                    feature_importances_min, feature_importances_max, feature_importances_mean, feature_importances_std])
        
        else:  
            with open(savepath, 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([ml_options["model_name"],number_rounds, accuracy_min, accuracy_max, accuracy_mean, accuracy_std, accuracy_class0_min, \
                accuracy_class0_max, accuracy_class0_mean, accuracy_class0_std, accuracy_class1_min, accuracy_class1_max, accuracy_class1_mean, \
                    accuracy_class1_std, balanced_accuracy_min, balanced_accuracy_max, balanced_accuracy_mean, balanced_accuracy_std,  \
                    feature_importances_min, feature_importances_max, feature_importances_mean, feature_importances_std])

        
        print('Number of Rounds: ' + str(number_rounds) + 
                '\nMin Accuracy: ' + str(accuracy_min) + '\nMax Accuracy: ' + str(accuracy_max) + '\nMean Accuracy: ' + str(accuracy_mean) + '\nStd Accuracy: ' + str(accuracy_std) +
                '\nMin Accuracy_class_0: ' + str(accuracy_class0_min) + '\nMax Accuracy_class_0: ' + str(accuracy_class0_max) + '\nMean Accuracy_class_0: ' + str(accuracy_class0_mean) + '\nStd Accuracy_class_0: ' + str(accuracy_class0_std) +
                '\nMin Accuracy_class_1: ' + str(accuracy_class1_min) + '\nMax Accuracy_class_1: ' + str(accuracy_class1_max) + '\nMean Accuracy_class_1: ' + str(accuracy_class1_mean) + '\nStd Accuracy_class_1: ' + str(accuracy_class1_std) +
                '\nMin Balanced_Accuracy: ' + str(balanced_accuracy_min) + '\nMax Balanced_Accuracy: ' + str(balanced_accuracy_max) + '\nMean Balanced_Accuracy: ' + str(balanced_accuracy_mean) + '\nStd Balanced_Accuracy: ' + str(balanced_accuracy_std))

        importances = feature_importances_mean.reshape((feature_importances_mean.shape[1],))
        top_positive_coefficients = np.argsort(importances)[-10:]
        top_negative_coefficients = np.argsort(importances)[:10]
        top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
        plt.figure(figsize=(18, 7))
        colors = ['green' if c < 0 else 'blue' for c in importances[top_coefficients]]
        plt.bar(np.arange(2 * 10), importances[top_coefficients], color=colors)
        feature_names = list(X_train.columns)
        feature_names = np.array(feature_names)
        plt.xticks(np.arange(1 + 2 * 10), feature_names[top_coefficients], rotation=45, ha='right')
        plt.savefig(f'{ml_options["model_name"]}_feature_importances.png')
    

    elif ml_options["model_architecture"] == "NN":
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
                elif itemnumber == 6:
                    log_loss_value_flat.append(sublist[itemnumber])
                
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
        balanced_accuracy_min = min(balanced_accuracy_flat)
        balanced_accuracy_max = max(balanced_accuracy_flat)
        balanced_accuracy_mean = np.mean(balanced_accuracy_flat)
        balanced_accuracy_std = np.std(balanced_accuracy_flat)
        precision_min = min(precision_flat)
        precision_max = max(precision_flat)
        precision_mean = np.mean(precision_flat)
        precision_std = np.std(precision_flat)
        f1_score_min = min(f1_score_flat)
        f1_score_max = max(f1_score_flat)
        f1_score_mean = np.mean(f1_score_flat)
        f1_score_std = np.std(f1_score_flat)
        log_loss_value_min = min(log_loss_value_flat)
        log_loss_value_max = max(log_loss_value_flat)
        log_loss_value_mean = np.mean(log_loss_value_flat)
        log_loss_value_std = np.std(log_loss_value_flat)
                  
        number_rounds = len(accuracy_flat)

        if not os.path.exists(savepath):
            with open(savepath, 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerow([ml_options["model_name"],number_rounds, accuracy_min, accuracy_max, accuracy_mean, accuracy_std, accuracy_class0_min, \
                accuracy_class0_max, accuracy_class0_mean, accuracy_class0_std, accuracy_class1_min, accuracy_class1_max, accuracy_class1_mean, \
                    accuracy_class1_std, precision_min, precision_max,precision_mean, precision_std, \
                    f1_score_min, f1_score_max, f1_score_mean, f1_score_std, balanced_accuracy_min, balanced_accuracy_max, balanced_accuracy_mean, 
                    balanced_accuracy_std,log_loss_value_min, log_loss_value_max, log_loss_value_mean, log_loss_value_std])
        
        print('Number of Rounds: ' + str(number_rounds) + 'Length X_train: ' + str(len(X_train)) + 'Length X_test: ' + str(len(X_test)) + '\nMin Accuracy: ' + str(accuracy_min) + '\nMax Accuracy: ' + str(accuracy_max) + '\nMean Accuracy: ' + str(accuracy_mean) + '\nStd Accuracy: ' + str(accuracy_std) +
                '\nMin Accuracy_class_0: ' + str(accuracy_class0_min) + '\nMax Accuracy_class_0: ' + str(accuracy_class0_max) + '\nMean Accuracy_class_0: ' + str(accuracy_class0_mean) + '\nStd Accuracy_class_0: ' + str(accuracy_class0_std) +
                '\nMin Accuracy_class_1: ' + str(accuracy_class1_min) + '\nMax Accuracy_class_1: ' + str(accuracy_class1_max) + '\nMean Accuracy_class_1: ' + str(accuracy_class1_mean) + '\nStd Accuracy_class_1: ' + str(accuracy_class1_std) +
                '\nMin Balanced_Accuracy: ' + str(balanced_accuracy_min) + '\nMax Balanced_Accuracy: ' + str(balanced_accuracy_max) + '\nMean Balanced_Accuracy: ' + str(balanced_accuracy_mean) + '\nStd Balanced_Accuracy: ' + str(balanced_accuracy_std) +
                '\nMean Precision:' + str(precision_mean) + '\nMean F1:' + str(f1_score_mean) + '\nMean Log-Loss: ' + str(log_loss_value_mean))
    
    model_flatlists = [accuracy_flat, accuracy_class1_flat, accuracy_class0_flat, precision_flat, f1_score_flat, balanced_accuracy_flat, roc_auc_flat]
    
    return model_flatlists


 