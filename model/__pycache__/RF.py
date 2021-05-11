import numpy as np
import sys
import os
import csv   
from config import MODEL_PATH, STANDARDPATH
import pickle

from sklearn import preprocessing
from sklearn import set_config
from sklearn.utils import shuffle, resample
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, ParameterSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import RFE, RFECV, SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import log_loss, roc_curve, auc
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA, FastICA

from imblearn.ensemble import BalancedRandomForestClassifier
import copy

def build_model(ml_options, X_train,X_test, y_train,y_test):

    y_train= np.array(y_train)
    y_test= np.array(y_test)
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    """
    "Feature Selection
    """

    y_train=np.ravel(y_train) # flattens y train
    y_test=np.ravel(y_test)

    if ml_options["feature_selection_option"] == 1:
        sel = VarianceThreshold(threshold=(.8 * (1 - .8))) # < 80% variance
        sel = sel.fit(X_train)
        X_train = sel.transform(X_train)
        X_test = sel.transform(X_test)
        selectorpath = os.path.join(MODEL_PATH, f'feature_selector.pkl')
        pickle.dump(sel, open(selectorpath, 'wb'))

    elif ml_options["feature_selection_option"] == 3:
        clf_rfelim = RandomForestClassifier(n_estimators=1000, random_state=ml_options["seed"])
        rfe = RFE(estimator=clf_rfelim, n_features_to_select=ml_options['number_features_recursive'],
                    step=ml_options['step_reduction_recursive'])
        rfe.fit(X_train, y_train)
        rfepath = os.path.join(MODEL_PATH, f'rfe.pkl')
        pickle.dump(rfe, open(rfepath, 'wb'))

        X_train= X_train[:,rfe.support_]
        X_test = X_test[:,rfe.support_]
        print(f"Selected columns are {X_train}")

    elif ml_options['feature_selection_option'] == 4:
        clf_rfe = RandomForestClassifier(n_estimators=1000, random_state=ml_options["seed"])
        rfe = RFECV(estimator=clf_rfe, step=ml_options['step_reduction_recursive'], cv=5,
                scoring=ml_options['scoring_recursive'], verbose = 1)
        rfe.fit(X_train, y_train)
        rfepath = os.path.join(MODEL_PATH, f'rfe.pkl')
        pickle.dump(rfe, open(rfepath, 'wb'))

        X_train = X_train[:,rfe.support_]
        X_test = X_test[:,rfe.support_]


    elif ml_options['feature_selection_option'] == 5: #https://stats.stackexchange.com/questions/276865/interpreting-the-outcomes-of-elastic-net-regression-for-binary-classification
        if ml_options['data_scaling_option'] == 1:
            clf_elastic_logregression_features = SGDClassifier(loss='log', penalty='elasticnet', l1_ratio=0.5, fit_intercept=False, tol=0.0001, max_iter=1000, random_state=ml_options["seed"])
            sfm = SelectFromModel(clf_elastic_logregression_features, threshold=ml_options['threshold_option'])
            sfm.fit(X_train, y_train)
            sfmpath = os.path.join(MODEL_PATH, f'sfm.pkl')
            pickle.dump(sfm, open(sfmpath, 'wb'))

            X_train = sfm.transform(X_train)
            X_test = sfm.transform(X_test)

        else:
            print('Please change data scaling option to perform Elstic Net‚')
            sys.exit("Execution therefore stopped")

    """
    "Hyperparameter Tuning
    """

    if ml_options['hyperparameter_tuning_option'] == 0:
        standard_parameter = {'n_estimators': 1000,
                       'criterion': 'gini',
                       'max_features': 'auto',
                       'max_depth': 8,
                       'min_samples_split': 2,
                       'min_samples_leaf': 1,
                       'bootstrap': True}
        best_parameter = standard_parameter

    
    elif ml_options['hyperparameter_tuning_option'] == 1:
        random_parameter = ml_options['hyperparameter_dict']

        clf_hyper_tuning = RandomForestClassifier(random_state=ml_options["seed"])

        random_hyper_tuning = RandomizedSearchCV(estimator = clf_hyper_tuning, param_distributions = random_parameter,
                                n_iter = ml_options['n_iter_hyper_randsearch'], cv = ml_options['cvs_hyper_randsearch'],
                                verbose=0, random_state=ml_options["seed"])
        random_hyper_tuning.fit(X_train, y_train)
        best_parameter = random_hyper_tuning.best_params_
    
    elif ml_options['hyperparameter_tuning_option'] == 2:
        random_parameter = ml_options['hyperparameter_dict']

        param_list = list(ParameterSampler(random_parameter, n_iter=ml_options['n_iter_hyper_randsearch'],
                        random_state=ml_options["seed"]))
        oob_accuracy_hyper_tuning = np.zeros((ml_options['n_iter_hyper_randsearch']))
        counter_hyper_tuning = 0

        for current_parameter_setting in param_list:
            print("hyperparameter tuning iteration: {}".format(counter_hyper_tuning))
            clf_hyper_tuning = RandomForestClassifier(n_estimators= current_parameter_setting["n_estimators"],
                            criterion = current_parameter_setting["criterion"], max_features= current_parameter_setting["max_features"],
                            max_depth= current_parameter_setting["max_depth"], min_samples_split= current_parameter_setting["min_samples_split"],
                            min_samples_leaf= current_parameter_setting["min_samples_leaf"], bootstrap= current_parameter_setting["bootstrap"],
                            oob_score=True, random_state=ml_options["seed"])
            clf_hyper_tuning = clf_hyper_tuning.fit(X_train, y_train)
            oob_accuracy_hyper_tuning[counter_hyper_tuning] = clf_hyper_tuning.oob_score_
            counter_hyper_tuning = counter_hyper_tuning +1

        best_parameter = param_list[np.argmax(oob_accuracy_hyper_tuning)]


    if ml_options['sampling'] in (0, 1, 2, 3):
        clf = RandomForestClassifier(n_estimators= best_parameter["n_estimators"], criterion = best_parameter["criterion"],
            max_features= best_parameter["max_features"], max_depth= best_parameter["max_depth"],
            min_samples_split= best_parameter["min_samples_split"], min_samples_leaf= best_parameter["min_samples_leaf"],
            bootstrap= best_parameter["bootstrap"], oob_score=True, random_state=ml_options["seed"])

    elif ml_options['sampling']  == 4:
        clf = RandomForestClassifier(n_estimators= best_parameter["n_estimators"], criterion = best_parameter["criterion"],
            max_features= best_parameter["max_features"], max_depth= best_parameter["max_depth"],
        min_samples_split= best_parameter["min_samples_split"], min_samples_leaf= best_parameter["min_samples_leaf"],
            bootstrap= best_parameter["bootstrap"], oob_score=True, class_weight='balanced', random_state=ml_options["seed"])

    elif ml_options['sampling']  == 5:
        clf = BalancedRandomForestClassifier(n_estimators= best_parameter["n_estimators"], criterion = best_parameter["criterion"],
            max_features= best_parameter["max_features"], max_depth= best_parameter["max_depth"],
            min_samples_split= best_parameter["min_samples_split"], min_samples_leaf= best_parameter["min_samples_leaf"],
            bootstrap= best_parameter["bootstrap"], oob_score=True, random_state=ml_options["seed"])

    elif ml_options['sampling']  == 6:
        clf = RandomForestClassifier(n_estimators= best_parameter["n_estimators"], criterion = best_parameter["criterion"],
            max_features= best_parameter["max_features"], max_depth= best_parameter["max_depth"],
            min_samples_split= best_parameter["min_samples_split"], min_samples_leaf= best_parameter["min_samples_leaf"],
            bootstrap= best_parameter["bootstrap"], oob_score=True, class_weight=ml_options['rf_classes_class_weight'],
            random_state=ml_options["seed"])
    
    return clf

def fit_model(X_train, y_train, clf):
        
    clf = clf.fit(X_train, y_train)
    return clf

def predict(X_test, y_test, clf, ml_options):
    y_prediction = np.zeros((len(y_test), 3))
    y_prediction[:,0] = clf.predict(X_test)
    y_prediction[:,1] = y_test[:]

    counter_class1_correct = 0
    counter_class2_correct = 0
    counter_class1_incorrect = 0
    counter_class2_incorrect = 0

    for i in range(len(y_test)):
        if y_prediction[i,0] == y_prediction[i,1]:
            y_prediction[i,2] = 1
            if y_prediction[i,1] == 1:
                counter_class1_correct += 1
            else:
                counter_class2_correct += 1
        else:
            y_prediction[i,2] = 0
            if y_prediction[i,1] == 1:
                counter_class1_incorrect += 1
            else:
                counter_class2_incorrect += 1

    """ Calculate accuracy scores """

    accuracy = y_prediction.mean(axis=0)[2]
    accuracy_class1 = counter_class1_correct / (counter_class1_correct + counter_class1_incorrect)
    accuracy_class2 = counter_class2_correct / (counter_class2_correct + counter_class2_incorrect)
    balanced_accuracy = (accuracy_class1 + accuracy_class2) / 2
    oob_accuracy = clf.oob_score_
    log_loss_value = log_loss(y_test, clf.predict_proba(X_test), normalize=True)

    """ Calculate feature importances """

    if ml_options['feature_selection_option'] == 0:
        feature_importances = clf.feature_importances_

    elif ml_options['feature_selection_option'] == 1:
        selectorpath = os.path.join(MODEL_PATH, f'feature_selector.pkl')
        sel = pickle.load(open(selectorpath, 'rb'))
        feature_importances = np.zeros((len(sel.get_support())))
        counter_features_selected = 0
        for number_features in range(len(sel.get_support())):
            if sel.get_support()[number_features] == True:
                feature_importances[number_features] = clf.feature_importances_[counter_features_selected]
                counter_features_selected += 1
            else:
                feature_importances[number_features] = 0

    elif ml_options['feature_selection_option'] in (3,4):
        rfepath = os.path.join(MODEL_PATH, f'rfe.pkl')
        rfe = pickle.load(open(rfepath, 'rb'))
        feature_importances = np.zeros((len(rfe.support_)))
        counter_features_selected = 0
        for number_features in range(len(rfe.support_)):
            if rfe.support_[number_features] == True:
                feature_importances[number_features] = clf.feature_importances_[counter_features_selected]
                counter_features_selected += 1
            else:
                feature_importances[number_features] = 0

    elif ml_options['feature_selection_option'] == 5:
        sfmpath = os.path.join(MODEL_PATH, 'sfm.pkl')
        sfm = pickle.load(open(sfmpath, 'rb'))
        feature_importances = np.zeros((len(sfm.get_support())))
        counter_features_selected = 0
        for number_features in range(len(sfm.get_support())):
            if sfm.get_support()[number_features] == True:
                feature_importances[number_features] = clf.feature_importances_[counter_features_selected]
                counter_features_selected += 1
            else:
                feature_importances[number_features] = 0
    
    fpr, tpr, __ = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
    mean_fpr = np.linspace(0, 1, 100)
    tprs = np.interp(mean_fpr, fpr, tpr)
    roc_auc = auc(fpr, tpr)
    fraction_positives, mean_predicted_value = calibration_curve(y_test, clf.predict_proba(X_test)[:,1], n_bins=10)
    
    print('Round Number: ', str(ml_options["seed"]), '\nAccuracy: ', str(accuracy), '\nAccuracy_class0: ', str(accuracy_class1), '\nAccuracy_class1: ', 
        str(accuracy_class2), '\nBalanced Accuracy: ', str(balanced_accuracy), '\nOOB Accuracy ', str(oob_accuracy), '\nLog Loss value: ', str(log_loss_value))

    savepath = os.path.join(STANDARDPATH, 'outcomes.csv')
    if not os.path.exists(savepath):
        header = ['model', 'seed/run', 'accuracy', 'accuracy_class0', 'accuracy_class1', 'balanced_accuracy', 'oob_accuracy', 'log_loss_value', 'fpr', 'tpr', 'tprs', 'roc_auc', 'fraction_positives', 'mean_predicted_value']
        with open(savepath, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(header)
            # write outcome rows
            writer.writerow([ml_options["model_name"],ml_options["seed"], accuracy, accuracy_class1, accuracy_class2, balanced_accuracy, oob_accuracy, log_loss_value, fpr, tpr, tprs, roc_auc, fraction_positives, mean_predicted_value])
    else:
        with open(savepath, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ml_options["model_name"],ml_options["seed"], accuracy, accuracy_class1, accuracy_class2, balanced_accuracy, oob_accuracy, log_loss_value, fpr, tpr, tprs, roc_auc, fraction_positives, mean_predicted_value])

    outcome_list = [accuracy, accuracy_class1, accuracy_class2, balanced_accuracy, oob_accuracy, log_loss_value, feature_importances, fpr, tpr, tprs, roc_auc, fraction_positives, mean_predicted_value]
    return outcome_list
    




