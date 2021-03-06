import time
import os
import json
import random

TIMESTR = time.strftime("%Y%m%d-%H%M%S")

START_TIME = time.time()
STANDARDPATH = os.environ.get("PSY_PATH")
DATAPATH_IN = os.environ.get("DATA_PATH") + "prepared_data.csv"
DATAPATH_OUT = os.path.join(STANDARDPATH, "data", "processed")
MODEL_PATH = os.path.join(STANDARDPATH,"model")


""" Create dictionary with parameters and options """

ml_options = {}

"General Options"
ml_options["model_name"] = "RF_" + TIMESTR
ml_options["n_iterations"] = int(input("Choose number of iterations:"))
ml_options["seed"] = random.sample(range(1,100),1)[0]
ml_options["feature_columns"] = ['registration','studyVariant','coach','PRE_bdi1','PRE_bdi2',
               'PRE_bdi3','PRE_bdi4','PRE_bdi5','PRE_bdi6','PRE_bdi7','PRE_bdi8','PRE_bdi9','PRE_bdi10',
               'PRE_bdi11','PRE_bdi12','PRE_bdi13','PRE_bdi14','PRE_bdi15','PRE_bdi16','PRE_bdi17','PRE_bdi18',
               'PRE_bdi19','PRE_bdi20','PRE_bdi21','POST_phqD1','POST_phqD2','POST_phqD3','POST_phqD4','POST_phqD5',
               'POST_phqD6','POST_phqD7','POST_phqD8','POST_phqD9',
               'PRE_phqS1','PRE_phqS2','PRE_phqS3','PRE_phqS4','PRE_phqS5','PRE_phqS6','PRE_phqS7',
               'PRE_phqS8','PRE_phqS9','PRE_phqS10','PRE_phqD1','PRE_phqD2','PRE_phqD3','PRE_phqD4','PRE_phqD5',
               'PRE_phqD6','PRE_phqD7','PRE_phqD8','PRE_phqD9', 'PRE_birth','PRE_sex','PRE_education',
               'PRE_work','PRE_household','PRE_relation','PRE_residence','PRE_internet','PRE_height','PRE_weight',
               'PRE_treatment','PRE_support','PRE_kPT','PRE_ill','PRE_sickleave','PRE_doc',
               'PRE_neurol','PRE_selfhelp','PRE_counsel','PRE_therapy','PRE_med','PRE_hospital',
               'PRE_hos_other', 'PRE_eurohis1','PRE_eurohis2','PRE_eurohis3','PRE_eurohis4','PRE_eurohis5',
               'PRE_eurohis6','PRE_eurohis7','PRE_eurohis8', 'TI_score','TI_MDE','TI_dyst','TI_MDE_x',
               'TI_MDE_vr','TI_MDE_tr','TI_medik','PRE_gad1',
               'PRE_gad2','PRE_gad3','PRE_gad4','PRE_gad5','PRE_gad6','PRE_gad7', 'PRE_costa1', 'PRE_costa2', 'PRE_costa3',
              'PRE_costa4', 'PRE_costa5', 'PRE_costa6', 'PRE_costa7', 'PRE_costa8', 'PRE_costa9', 'PRE_costa10', 'PRE_costa11',
              'PRE_costa12', 'PRE_costa13', 'PRE_costa14', 'PRE_costa15', 'PRE_costa16', 'PRE_costa17', 'PRE_costa18',
              'PRE_costa19', 'PRE_costa20', 'PRE_costa21', 'PRE_pathev1', 'PRE_pathev2', 'PRE_pathev3', 'PRE_pathev4',
              'PRE_pathev5', 'PRE_pathev6', 'PRE_pathev7', 'PRE_pathev8', 'PRE_pathev9', 'PRE_pathev10', 
              'PRE_ipqr1','PRE_ipqr2','PRE_ipqr3','PRE_ipqr4','PRE_ipqr5',
               'PRE_ipqr6','PRE_ipqr7','PRE_ipqr8','PRE_ipqr9','PRE_ipqr10','PRE_ipqr11','PRE_ipqr12','PRE_ipqr13',
               'PRE_ipqr14','PRE_ipqr15','PRE_ipqr16','PRE_ipqr17','PRE_ipqr18', 'PRE_bsss1','PRE_bsss2','PRE_bsss3',
               'PRE_bsss4','PRE_bsss5','PRE_bsss6','PRE_bsss7','PRE_bsss8','PRE_bsss9','PRE_bsss10','PRE_bsss11',
               'PRE_bsss12','PRE_bsss13','PRE_gpse1','PRE_gpse2','PRE_gpse3','PRE_gpse4','PRE_gpse5','PRE_gpse6',
               'PRE_gpse7','PRE_gpse8','PRE_gpse9','PRE_gpse10','PRE_pvq1','PRE_pvq2','PRE_pvq3','PRE_pvq4','PRE_pvq5',
               'PRE_pvq6','PRE_pvq7','PRE_pvq8','PRE_pvq9','PRE_pvq10','PRE_pvq11','PRE_pvq12','PRE_pvq13','PRE_pvq14',
               'PRE_pvq15','PRE_pvq16','PRE_pvq17','PRE_pvq18','PRE_pvq19','PRE_pvq20','PRE_pvq21', 'PRE_imet1','PRE_imet2','PRE_imet3','PRE_imet4','PRE_imet5','PRE_imet6','PRE_imet7',
                'PRE_imet8','PRE_imet9','PRE_imet10', 'M1_phqD1','M1_phqD2','M1_phqD3','M1_phqD4','M1_phqD5','M1_phqD6',
               'M1_phqD7','M1_phqD8','M1_phqD9']


ml_options["target_columns_post"] =  ['POST_phqD1','POST_phqD2','POST_phqD3','POST_phqD4','POST_phqD5',
               'POST_phqD6','POST_phqD7','POST_phqD8','POST_phqD9']

ml_options["target_columns_pre"] = ['PRE_phqD1','PRE_phqD2','PRE_phqD3','PRE_phqD4','PRE_phqD5','PRE_phqD6','PRE_phqD7',
               'PRE_phqD8','PRE_phqD9']

ml_options["target_id"] = "phq_change"

ml_options["include_early_change"] = 0 # 0 = no, 1 = yes

ml_options['test_size_option'] = float(input("Set proportion of data points in test set (e.g. 0.33):"))

ml_options['sampling'] = int(input("""Choose an option for train/test splits:
                                    \n- 0: no balancing
                                    \n- 1: sample balancing for train only
                                    \n- 2: sample balancing for test only
                                    \n- 3: sample balancing for train and test
                                    \n- 4: balanced class_weights (RF only)
                                    \n- 5: balanced random forest estimator
                                    \n- 6: class weighting with custom dict
                                    """))

if ml_options["sampling"] in (0, 4, 5, 6):
    ml_options['stratify_option'] = int(input("Would you like to stratify? (0 = no, 1=yes):"))

ml_options['dummy_enconding'] = int(input("Would you like to one-hot encode categorical variables? (0 = no, 1=yes):"))

if ml_options['sampling'] == 6:
    ml_options['rf_classes_class_weight'] = dict({1:3, 0:1})

"Model Options"

ml_options["model"] = "RF"
ml_options["n_classes"] = 2
ml_options["metrics"] = "classification_metrics"

"Data preprocessing"

ml_options["preprocessor"] = "default"
ml_options['data_scaling_option'] = int(input("""Would you like to scale your data?:
                                    \n- 0: No Scaling
                                    \n- 1: Centering and Standardisation"""))

if ml_options['data_scaling_option'] == 1:
    ml_options['scaling_columns'] = ["bmi_score", "age", "PRE_internet"]

ml_options['missing_values_option'] = int(input("""Chose how to deal with NA:
                                    \n- 0: no Imputation/ just leave them
                                    \n- 1: delete rows with missing values
                                    \n- 2: replace NA with mode/median/most most frequent
                                    \n- 3: MICE Imputation"""))

ml_options['missing_outcome_option'] = int(input("""Would you like to drop missing rows in outcome variable?:
                                    \n- 0: No
                                    \n- 1: Yes """))                            

"Model Tuning"

ml_options['feature_selection_option'] = int(input("""Choose feature selection strategy:
                                    \n- 0: no feature selection
                                    \n- 1: remove features with low variance (< 80%)
                                    \n- 3: recursive
                                    \n- 4: recursive cross validated
                                    \n- 5: Elastic-Net (only with standardized data)
                                """))
if ml_options['feature_selection_option'] in (3, 4):
    ml_options['number_features_recursive'] = int(input("Choose max. number of features (default is 10):"))
    ml_options['step_reduction_recursive'] = int(input("Choose n of features to remove per step (default is 3):"))

if ml_options['feature_selection_option'] == 5:
    ml_options['treshold_option'] = 'mean'


ml_options['hyperparameter_tuning_option'] = int(input("""Choose hyperparameter tuning option:
                                    \n- 0: no hyperparameter tuning
                                    \n- 1: hyperparameter tuning per scikit-learn RandomizeSearch
                                    \n- 2: hyperparameter tuning per oob error
                                    \n- 3: hyperparameter tuning per Bayesian optimization (hyperopt)"""))


ml_options['hyperparameter_dict'] = {}

ml_options['n_iter_hyper_randsearch'] = 100 # Anzahl Durchgänge mit zufälligen Hyperparameter - Kombinationen; so hoch wie möglich
ml_options['cvs_hyper_randsearch'] = 5 # default-cvs bei Hyperparameter - Kombinationen; Höhere Anzahl weniger Overfitting

"Postprocessing"

ml_options['permutation_option'] = int(input("Would you like a permutation test? (0 = no, 1=yes):"))

if ml_options['permutation_option'] == 1:
    ml_options['n_permutations_option'] = int(input("Choose number of permutations (default is 5000):"))

ml_options['save_model_option'] = int(input("Would you like to safe the model? (0 = no, 1=yes):"))

ml_options['save_config_option'] = int(input("Would you like to safe the model configurations? (0 = no, 1=yes):"))




def rf_config(options_dict):
    options_dict['data_scaling_option'] = 1
    options_dict['hyperparameter_dict'] = {'n_estimators': [500, 2000, 10000],
                                    'criterion_hyper': ['gini', 'entropy'],
                                    'max_features_hyper': ['sqrt', 'log2'],
                                    'max_depth': [2, 3, 4, 5],
                                    'min_samples_split_hyper': [2,4,6,8,10],
                                    'min_samples_leaf_hyper': [1, 2, 3, 4, 5],
                                    'bootstrap_hyper': [True]} # [True, False] verfügbar, aber OOB-Estimate nur für True verfügbar
    return options_dict


