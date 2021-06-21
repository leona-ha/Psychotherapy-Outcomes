from config import MODEL_PATH, STANDARDPATH

import numpy as np
import sys
import os
import csv   
import pickle
import copy

from sklearn import preprocessing
from sklearn import set_config
from sklearn.utils import shuffle, resample
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, ParameterSampler
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import RFE, RFECV, SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import log_loss, roc_curve, auc
from sklearn.calibration import calibration_curve


def run(ml_options, X_train,X_test, y_train,y_test):

    X_train = X_train[["outcome_sum_pre"]]
    X_test = X_test[["outcome_sum_pre"]]

    log_model = LogisticRegression(C=1.0)
    log_model.fit(X_train, y_train)   

    y_prediction = np.zeros((len(y_test), 3))
    y_prediction[:,0] = log_model.predict(X_test)
    y_prediction[:,1] = y_test[:]

    counter_class1_correct = 0
    counter_class0_correct = 0
    counter_class1_incorrect = 0
    counter_class0_incorrect = 0

    for i in range(len(y_test)):
        if y_prediction[i,0] == y_prediction[i,1]:
            y_prediction[i,2] = 1
            if y_prediction[i,1] == 1:
                counter_class1_correct += 1
            else:
                counter_class0_correct += 1
        else:
            y_prediction[i,2] = 0
            if y_prediction[i,1] == 1:
                counter_class1_incorrect += 1
            else:
                counter_class0_incorrect += 1
                
    #print(counter_class1_correct, counter_class0_correct, counter_class1_incorrect,counter_class0_incorrect)

    """ Calculate accuracy scores """

    accuracy = y_prediction.mean(axis=0)[2]
    accuracy_class1 = counter_class1_correct / (counter_class1_correct + counter_class1_incorrect) # Recall
    accuracy_class0 = counter_class0_correct / (counter_class0_correct + counter_class0_incorrect)
    balanced_accuracy = (accuracy_class1 + accuracy_class0) / 2
    precision = counter_class1_correct / (counter_class1_correct + counter_class0_incorrect)
    f1_score = 2 * ((accuracy_class1 * precision)/(accuracy_class1+precision))
    #log_loss_value = log_loss(y_test, clf.predict_proba(X_test), normalize=True)
    outcome_list = [accuracy, accuracy_class1, accuracy_class0, precision, f1_score, balanced_accuracy]
    return outcome_list
