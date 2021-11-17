import numpy as np
import sys
import os
import csv   
from config import MODEL_PATH, ROUND_PATH
import pickle
#from hpo import get_acc_status, obj_fnc
from . import hpo
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK,SparkTrials, space_eval
from sklearn.preprocessing import scale, normalize
from sklearn.model_selection import cross_val_score 
import pyspark 
#from hyperopt.early_stop import no_progress_loss

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

    elif ml_options['feature_selection_option'] == 4:
        clf_rfe = RandomForestClassifier(n_estimators=1000, random_state=ml_options["seed"])
        rfe = RFECV(estimator=clf_rfe, step=ml_options['step_reduction_recursive'], cv=5,
                scoring=ml_options["main_metric"], verbose = 1)
        rfe.fit(X_train, y_train)
        print("Optimal number of features : %d" % rfe.n_features_)

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
            print('Please change data scaling option to perform Elastic Net‚')
            sys.exit("Execution therefore stopped")

    """
    "Hyperparameter Tuning
    """

    if ml_options['hyperparameter_tuning_option'] == 0:
        standard_parameter = {'n_estimators': 1000,
                       'criterion': 'gini',
                       'max_features': 'auto',
                       'max_depth': 9,
                       'min_samples_split': 2,
                       'min_samples_leaf': 1,
                       'bootstrap': True}
        best_parameter = standard_parameter

    
    elif ml_options['hyperparameter_tuning_option'] == 1:
        random_parameter = ml_options['hyperparameter_dict']

        clf_hyper_tuning = RandomForestClassifier(random_state=ml_options["seed"])

        random_hyper_tuning = RandomizedSearchCV(estimator = clf_hyper_tuning, param_distributions = random_parameter,
                                n_iter = ml_options['n_iter_hyper_randsearch'], cv = ml_options['cvs_hyper_randsearch'],
                                verbose=0, random_state=ml_options["seed"], scoring="recall", n_jobs=4)
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
            clf_hyper_tuning = RandomForestClassifier(n_jobs=4,n_estimators= current_parameter_setting["n_estimators"],
                            criterion = current_parameter_setting["criterion"], max_features= current_parameter_setting["max_features"],
                            max_depth= current_parameter_setting["max_depth"], min_samples_split= current_parameter_setting["min_samples_split"],
                            min_samples_leaf= current_parameter_setting["min_samples_leaf"], bootstrap= current_parameter_setting["bootstrap"],
                            oob_score=True, random_state=ml_options["seed"])
            clf_hyper_tuning = clf_hyper_tuning.fit(X_train, y_train)
            oob_accuracy_hyper_tuning[counter_hyper_tuning] = clf_hyper_tuning.oob_score_
            counter_hyper_tuning = counter_hyper_tuning +1

        best_parameter = param_list[np.argmax(oob_accuracy_hyper_tuning)]
    
    elif ml_options["hyperparameter_tuning_option"] == 3: ## Hyperopt 

        space = { 'max_depth': hp.choice('max_depth', [5,6,7,8,9,10,11,12,13,14,15]), 
           'max_features': hp.choice('max_features', ['sqrt', 'log2']), 
           'n_estimators': hp.choice('n_estimators',[10,20,30,40,50]), 
           'min_samples_split':hp.choice('min_samples_split', [2,3,4,5,6,7,8]),
           'min_samples_leaf':hp.choice('min_samples_leaf', [1,2,3,4,5]),
           'criterion': hp.choice('criterion', ["gini", "entropy"]), 
           'n_estimators': hp.choice('n_estimators', [500,1000,1500,5000]),
           'model': "RF"}
        
        X = X_train
        y = y_train

        def get_acc_status(clf,X,y): 
            losses = cross_val_score(clf, X, y, cv=5, scoring='recall')
            acc = np.mean(losses)
            var= np.var(losses, ddof=1) 

            return {'loss': -acc, 'loss_variance':var,'status': STATUS_OK}
        def obj_fnc(params) :  
            model = params.get('model').lower() 
            X_ = X[:]
            if 'normalize' in params:
                if params['normalize'] == 1:
                    X_ = normalize(X_)
                del params['normalize'] 
            if 'scale' in params:
                if params['scale'] == 1:
                    X_ = scale(X_)
                del params['scale'] 

            del params['model'] 
            if model == "rf":
                clf = RandomForestClassifier(**params) 
    
            return(get_acc_status(clf,X_,y))

        def stop(trial, count=0):
             return count+1 >= 100, [count+1]

        #hypopt_trials = Trials()
        spark_trials = SparkTrials(parallelism=8)
        early_stop_fn=stop
        best = fmin(obj_fnc, space, algo=tpe.suggest, max_evals=100, trials= spark_trials,early_stop_fn=early_stop_fn)
        best_parameter = space_eval(space,best)
 

    if ml_options['sampling'] in (0, 1, 2, 3):
        clf = RandomForestClassifier(n_estimators= best_parameter["n_estimators"], criterion = best_parameter["criterion"],
            max_features= best_parameter["max_features"], max_depth= best_parameter["max_depth"],
            min_samples_split= best_parameter["min_samples_split"], min_samples_leaf= best_parameter["min_samples_leaf"],
         oob_score=True, random_state=ml_options["seed"])

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

def predict(X_train, X_test, y_test, clf, ml_options):
    y_prediction = np.zeros((len(y_test), 3))
    y_prediction[:,0] = clf.predict(X_test)
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
                

    """ Calculate accuracy scores """

    accuracy = y_prediction.mean(axis=0)[2]
    accuracy_class1 = counter_class1_correct / (counter_class1_correct + counter_class1_incorrect) # Recall
    accuracy_class0 = counter_class0_correct / (counter_class0_correct + counter_class0_incorrect)
    balanced_accuracy = (accuracy_class1 + accuracy_class0) / 2
    precision = counter_class1_correct / (counter_class1_correct + counter_class0_incorrect)
    f1_score = 2 * ((accuracy_class1 * precision)/(accuracy_class1+precision))
    oob_accuracy = clf.oob_score_
    log_loss_value = log_loss(y_test, clf.predict_proba(X_test), normalize=True)

    """ Calculate feature importances """

    if ml_options['feature_selection_option'] == 0:
        feature_importances = clf.feature_importances_
        counter_features_selected = X_train.shape[1]
        feature_importances_count = np.ones((len(clf.feature_importances_)))
        

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
        feature_importances_count = np.zeros((len(rfe.support_)))
        counter_features_selected = 0
        for number_features in range(len(rfe.support_)):
            if rfe.support_[number_features] == True:
                feature_importances[number_features] = clf.feature_importances_[counter_features_selected]
                feature_importances_count[number_features] = 1
                counter_features_selected += 1
            else:
                feature_importances[number_features] = 0
                feature_importances_count[number_features] = 0
        print("Optimal number of features : %d" % rfe.n_features_)

    elif ml_options['feature_selection_option'] == 5:
        sfmpath = os.path.join(MODEL_PATH, 'sfm.pkl')
        sfm = pickle.load(open(sfmpath, 'rb'))
        feature_importances = np.zeros((len(sfm.get_support())))
        feature_importances_count = np.zeros((len(sfm.get_support())))
        counter_features_selected = 0
        for number_features in range(len(sfm.get_support())):
            if sfm.get_support()[number_features] == True:
                feature_importances[number_features] = clf.feature_importances_[counter_features_selected]
                feature_importances_count[number_features] = 1
                counter_features_selected += 1
            else:
                feature_importances[number_features] = 0
                feature_importances_count[number_features] = 0

    #print("Nr of selected features:", counter_features_selected)
    
    fpr, tpr, __ = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
    mean_fpr = np.linspace(0, 1, 100)
    tprs = np.interp(mean_fpr, fpr, tpr)
    roc_auc = auc(fpr, tpr)
    fraction_positives, mean_predicted_value = calibration_curve(y_test, clf.predict_proba(X_test)[:,1], n_bins=10)
    
  #  print('Round Number: ', str(ml_options["seed"]), '\nSelected Features: ', str(counter_features_selected),'\nAccuracy: ', str(accuracy), '\nAccuracy_class0: ', str(accuracy_class1), '\nAccuracy_class1/Recall: ', 
     #   str(accuracy_class0), '\nPrecision: ', str(precision), '\nF1_Score: ', str(f1_score), '\nOOB Accuracy ', str(oob_accuracy), '\nLog Loss value: ', str(log_loss_value))

    savepath = os.path.join(ROUND_PATH, ml_options['model_name'])
    if not os.path.exists(savepath):
        header = ['model', 'seed/run', 'n_features_selected', 'accuracy', 'accuracy_class1/recall', 'accuracy_class0', 'precision', 'f1_score','balanced_accuracy', 'oob_accuracy', 'log_loss_value', 'roc_auc']
        with open(savepath, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(header)
            # write outcome rows
            writer.writerow([ml_options["model_name"],ml_options["seed"], counter_features_selected, accuracy, accuracy_class1, accuracy_class0, precision, f1_score,balanced_accuracy, oob_accuracy, log_loss_value, roc_auc])
    else:
        with open(savepath, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ml_options["model_name"],ml_options["seed"], counter_features_selected,accuracy, accuracy_class1, accuracy_class0, precision, f1_score, balanced_accuracy, oob_accuracy, log_loss_value, roc_auc])

    outcome_list = [accuracy, accuracy_class1, accuracy_class0, precision, f1_score, balanced_accuracy,oob_accuracy, log_loss_value, feature_importances, fpr, tpr, tprs, roc_auc, fraction_positives, mean_predicted_value, counter_features_selected, feature_importances_count]
    return outcome_list
    




