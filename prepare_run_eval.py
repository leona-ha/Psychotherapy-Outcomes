def process_and_run(numrun):

    global standardpath, ml_options

    random_state_option = numrun

    """
    "Import Data und Labels
    """

    X_train = pd.read_csv(full_path_X_train, sep="\s", header=None, engine='python')
    X_test = pd.read_csv(full_path_X_test, sep="\s", header=None, engine='python')
    y_train = pd.read_csv(full_path_y_train, sep="\s", header=None, engine='python')
    y_test = pd.read_csv(full_path_y_test, sep="\s", header=None, engine='python')

"""""""""""""""""
"
"
" Training-Set
"
"
"""""""""""""""""


"""
"Imputation missing values
"""

    if ml_options['missing_values_option'] == 0: # Just leave missing values
        X_train_imputed = copy.deepcopy(X_train)
    elif ml_options['missing_values_option']  == 1: # Drop missing values
        X_train = X_train.replace([999, 888,777], np.NaN)
        X_train_imputed = copy.deepcopy(X_train)
        X_train_imputed.dropna(inplace=True)
        y_train = y_train.drop(y_train.index[~y_train.index.isin(X_train_imputed.index)])
    elif ml_options['missing_values_option']  == 2: # Fill them with mean/median/mode
        imp_arith = SimpleImputer(missing_values=999, strategy='mean', verbose = 10)
        imp_median = SimpleImputer(missing_values=888, strategy='median')
        imp_mode = SimpleImputer(missing_values=777, strategy='most_frequent')
        imp_arith.fit(X_train)
        imp_median.fit(X_train)
        imp_mode.fit(X_train)
        X_train_imputed = imp_arith.transform(X_train)
        X_train_imputed = imp_median.transform(X_train_imputed)
        X_train_imputed = imp_mode.transform(X_train_imputed)
    elif ml_options['missing_values_option']  == 3: # MICE imputation
        scaffolding_arith = np.zeros((X_train.shape[0],X_train.shape[1]))
        scaffolding_median = np.zeros((X_train.shape[0],X_train.shape[1]))
        scaffolding_mode = np.zeros((X_train.shape[0],X_train.shape[1]))

        scaffolding_arith[X_train==999] = 1
        scaffolding_median[X_train==888] = 1
        scaffolding_mode[X_train==777] = 1

        X_train_arith = X_train.replace([777,888], 999)
        X_train_median = X_train.replace([777,999], 888)
        X_train_mode = X_train.replace([888,999], 777)

        imp_arith_mice = IterativeImputer(estimator=BayesianRidge(), missing_values=999, sample_posterior=True,
                        max_iter=10, initial_strategy="mean", random_state=random_state_option)
        imp_median_mice = IterativeImputer(estimator=BayesianRidge(), missing_values=888, sample_posterior=True,
                        max_iter=10, initial_strategy="median", random_state=random_state_option)
        imp_mode_mice = IterativeImputer(estimator=BayesianRidge(), missing_values=777, sample_posterior=True,
                        max_iter=10, initial_strategy="most_frequent", min_value=0, max_value=1, random_state=random_state_option)

        imp_arith_mice.fit(X_train_arith)
        imp_median_mice.fit(X_train_median)
        imp_mode_mice.fit(X_train_mode)
        X_train_arith_imputed = imp_arith_mice.transform(X_train_arith)
        X_train_median_imputed = imp_median_mice.transform(X_train_median)
        X_train_mode_imputed = imp_mode_mice.transform(X_train_mode)

        X_train_imputed = copy.deepcopy(X_train)

        for imputed_values_x in range(scaffolding_arith.shape[0]):
            for imputed_values_y in range(scaffolding_arith.shape[1]):
                if scaffolding_arith[imputed_values_x,imputed_values_y] == 1:
                    X_train_imputed.iloc[imputed_values_x,imputed_values_y] = X_train_arith_imputed[imputed_values_x,imputed_values_y]
                elif scaffolding_median[imputed_values_x,imputed_values_y] == 1:
                    X_train_imputed.iloc[imputed_values_x,imputed_values_y] = round(X_train_median_imputed[imputed_values_x,imputed_values_y])
                elif scaffolding_mode[imputed_values_x,imputed_values_y] == 1:
                    X_train_imputed.iloc[imputed_values_x,imputed_values_y] = round(X_train_mode_imputed[imputed_values_x,imputed_values_y])


"""
"Scaling
"""

    if ml_options['data_scaling_option'] == 0:
        X_train_imputed_scaled = copy.deepcopy(X_train_imputed)
    elif ml_options['data_scaling_option']  == 1:
        scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train_imputed)
        X_train_imputed_scaled = scaler.transform(X_train_imputed)


"""
"Feature Selection
"""

    y_train=np.ravel(y_train) # flattens y train


    if ml_options['feature_selection_option'] == 0:
        X_train_imputed_scaled_selected = copy.deepcopy(X_train_imputed_scaled)
    elif ml_options["feature_selection_option"] == 1:
        sel = VarianceThreshold(threshold=(.8 * (1 - .8))) # < 80% variance
        sel.fit(X_train_imputed_scaled)
        X_train_imputed_scaled_selected = sel.transform(X_train_imputed_scaled)
    elif ml_options["feature_selection_option"] == 3:
        clf_rfelim = RandomForestClassifier(n_estimators=1000, random_state=random_state_option)
        rfe = RFE(estimator=clf_rfelim, n_features_to_select=ml_options['number_features_recursive'],
                    step=ml_options['step_reduction_recursive'],random_state=random_state_option)
        rfe.fit(X_train_imputed_scaled, y_train)
        X_train_imputed_scaled_selected = copy.deepcopy(X_train_imputed_scaled)
        X_train_imputed_scaled_selected = X_train_imputed_scaled_selected[:,rfe.support_]
    elif ml_options['feature_selection_option'] == 4:
        X_train_scaled_imputed_rf = copy.deepcopy(X_train_imputed_scaled)
        clf_rfe = RandomForestClassifier(n_estimators=1000, random_state=random_state_option)
        rfe = RFECV(estimator=clf_rfe, step=ml_options['step_reduction_recursive'], cv=5,
                scoring=ml_options['scoring_recursive'], verbose = 1)
        rfe.fit(X_train_scaled_imputed_rf, y_train)
        X_train_imputed_scaled_selected = copy.deepcopy(X_train_scaled_imputed_rf)
        X_train_imputed_scaled_selected = X_train_imputed_scaled_selected[:,rfe.support_]

    elif ml_options['feature_selection_option'] == 5: #https://stats.stackexchange.com/questions/276865/interpreting-the-outcomes-of-elastic-net-regression-for-binary-classification
        if ml_options['data_scaling_option'] == 1:
            clf_elastic_logregression_features = SGDClassifier(loss='log', penalty='elasticnet', l1_ratio=0.5, fit_intercept=False, tol=0.0001, max_iter=1000, random_state=random_state_option)
            sfm = SelectFromModel(clf_elastic_logregression_features, threshold=ml_options['threshold_option'])
            sfm.fit(X_train_imputed_scaled, y_train)
            X_train_imputed_scaled_selected = sfm.transform(X_train_imputed_scaled)
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
                       'max_depth': None,
                       'min_samples_split': 2,
                       'min_samples_leaf': 1,
                       'bootstrap': True}
        best_parameter = standard_parameter
    elif ml_options['hyperparameter_tuning_option'] == 1:
        random_parameter = ml_options['hyperparameter_dict']

        clf_hyper_tuning = RandomForestClassifier(random_state=random_state_option)

        random_hyper_tuning = RandomizedSearchCV(estimator = clf_hyper_tuning, param_distributions = random_parameter,
                                n_iter = ml_options['n_iter_hyper_randsearch'], cv = ml_options['cvs_hyper_randsearch'],
                                verbose=0, random_state=random_state_option)
        random_hyper_tuning.fit(X_train_imputed_scaled_selected, y_train)
        best_parameter = random_hyper_tuning.best_params_

    elif ml_options['hyperparameter_tuning_option'] == 2:
        random_parameter = ml_options['hyperparameter_dict']

        param_list = list(ParameterSampler(random_parameter, n_iter=ml_options['n_iter_hyper_randsearch'],
                     random_state=random_state_option))
        oob_accuracy_hyper_tuning = np.zeros((ml_options['n_iter_hyper_randsearch']))
        counter_hyper_tuning = 0

        for current_parameter_setting in param_list:
            print("hyperparameter tuning iteration: {}".format(counter_hyper_tuning))
            clf_hyper_tuning = RandomForestClassifier(n_estimators= current_parameter_setting["n_estimators"],
                            criterion = current_parameter_setting["criterion"], max_features= current_parameter_setting["max_features"],
                            max_depth= current_parameter_setting["max_depth"], min_samples_split= current_parameter_setting["min_samples_split"],
                            min_samples_leaf= current_parameter_setting["min_samples_leaf"], bootstrap= current_parameter_setting["bootstrap"],
                            oob_score=True, random_state=random_state_option)
            clf_hyper_tuning = clf_hyper_tuning.fit(X_train_imputed_scaled_selected, y_train)
            oob_accuracy_hyper_tuning[counter_hyper_tuning] = clf_hyper_tuning.oob_score_
            counter_hyper_tuning = counter_hyper_tuning +1

        best_parameter = param_list[np.argmax(oob_accuracy_hyper_tuning)]

"""
"Random Forest Analyse
"""

    if ml_options['balanced_split_option'] == 0 or 1 or 2:
        clf = RandomForestClassifier(n_estimators= best_parameter["n_estimators"], criterion = best_parameter["criterion"],
            max_features= best_parameter["max_features"], max_depth= best_parameter["max_depth"],
            min_samples_split= best_parameter["min_samples_split"], min_samples_leaf= best_parameter["min_samples_leaf"],
            bootstrap= best_parameter["bootstrap"], oob_score=True, random_state=random_state_option)

    elif ml_options['balanced_split_option']  == 3:
        clf = RandomForestClassifier(n_estimators= best_parameter["n_estimators"], criterion = best_parameter["criterion"],
            max_features= best_parameter["max_features"], max_depth= best_parameter["max_depth"],
        min_samples_split= best_parameter["min_samples_split"], min_samples_leaf= best_parameter["min_samples_leaf"],
            bootstrap= best_parameter["bootstrap"], oob_score=True, class_weight='balanced', random_state=random_state_option)

    elif ml_options['balanced_split_option']  == 4:
        clf = BalancedRandomForestClassifier(n_estimators= best_parameter["n_estimators"], criterion = best_parameter["criterion"],
            max_features= best_parameter["max_features"], max_depth= best_parameter["max_depth"],
            min_samples_split= best_parameter["min_samples_split"], min_samples_leaf= best_parameter["min_samples_leaf"],
            bootstrap= best_parameter["bootstrap"], oob_score=True, random_state=random_state_option)

    elif ml_options['balanced_split_option']  == 5:
        clf = RandomForestClassifier(n_estimators= best_parameter["n_estimators"], criterion = best_parameter["criterion"],
            max_features= best_parameter["max_features"], max_depth= best_parameter["max_depth"],
            min_samples_split= best_parameter["min_samples_split"], min_samples_leaf= best_parameter["min_samples_leaf"],
            bootstrap= best_parameter["bootstrap"], oob_score=True, class_weight=ml_options['rf_classes_class_weight'],
            random_state=random_state_option)

    clf = clf.fit(X_train_imputed_scaled_selected, y_train)

    """""""""""""""""
    "
    "
    " Test-Set
    "
    "
    """""""""""""""""


    """
    "Imputation missing values - Test Set
    """

    if ml_options['missing_values_option'] == 0: # Just leave missing values
        X_test_imputed = copy.deepcopy(X_test)

    elif ml_options['missing_values_option']  == 1: # Drop missing values
        X_test = X_test.replace([999, 888,777], np.NaN)
        X_test_imputed = copy.deepcopy(X_train)
        X_test_imputed.dropna(inplace=True)
        y_test = y_test.drop(y_test.index[~y_test.index.isin(X_test_imputed.index)])

    elif ml_options['missing_values_option']  == 2: # Fill them with mean/median/mode
        X_test_imputed = imp_arith.transform(X_test)
        X_test_imputed = imp_median.transform(X_test_imputed)
        X_test_imputed = imp_mode.transform(X_test_imputed)

    elif ml_options['missing_values_option']  == 3: # MICE imputation
        scaffolding_arith_test = np.zeros((X_test.shape[0],X_test.shape[1]))
        scaffolding_median_test = np.zeros((X_test.shape[0],X_test.shape[1]))
        scaffolding_mode_test = np.zeros((X_test.shape[0],X_test.shape[1]))

        scaffolding_arith_test[X_test==999] = 1
        scaffolding_median_test[X_test==888] = 1
        scaffolding_mode_test[X_test==777] = 1

        X_test_arith = X_test.replace(777, 999)
        X_test_arith = X_test_arith.replace(888, 999)
        X_test_median = X_test.replace(777, 888)
        X_test_median = X_test_median.replace(999, 888)
        X_test_mode = X_test.replace(888, 777)
        X_test_mode = X_test_mode.replace(999, 777)

    X_test_arith_imputed = imp_arith_mice.transform(X_test_arith)
    X_test_median_imputed = imp_median_mice.transform(X_test_median)
    X_test_mode_imputed = imp_mode_mice.transform(X_test_mode)

    X_test_imputed = copy.deepcopy(X_test)

    for imputed_values_x in range(scaffolding_arith_test.shape[0]):
        for imputed_values_y in range(scaffolding_arith_test.shape[1]):
            if scaffolding_arith_test[imputed_values_x,imputed_values_y] == 1:
                X_test_imputed.iloc[imputed_values_x,imputed_values_y] = round(X_test_arith_imputed[imputed_values_x,imputed_values_y])
            elif scaffolding_median_test[imputed_values_x,imputed_values_y] == 1:
                X_test_imputed.iloc[imputed_values_x,imputed_values_y] = round(X_test_median_imputed[imputed_values_x,imputed_values_y])
            elif scaffolding_mode_test[imputed_values_x,imputed_values_y] == 1:
                X_test_imputed.iloc[imputed_values_x,imputed_values_y] = round(X_test_mode_imputed[imputed_values_x,imputed_values_y])

"""
"Scaling - Test Set
"""


    if ml_options['data_scaling_option']  == 0:
        X_test_imputed_scaled = copy.deepcopy(X_test_imputed)
    elif ml_optionsn['data_scaling_option']  == 1:
        X_test_imputed_scaled = scaler.transform(X_test_imputed)

    """
    "Feature Selection - Test Set
    """

    y_test=np.ravel(y_test)

    if ml_options['feature_selection_option'] == 0:
        X_test_scaled_imputed_selected = copy.deepcopy(X_test_imputed_scaled)
    elif ml_options['feature_selection_option'] == 1:
        X_test_scaled_imputed_selected = sel.transform(X_test_imputed_scaled)
    elif ml_options['feature_selection_option'] == 3 or 4:
        X_test_scaled_imputed_selected = copy.deepcopy(X_test_imputed_scaled)
        X_test_scaled_imputed_selected = X_test_scaled_imputed_selected[:,rfe.support_]
    elif ml_options['feature_selection_option'] == 5:
        X_test_scaled_imputed_selected = sfm.transform(X_test_imputed_scaled)

    else:
        print("Not working yet")
        sys.exit("Stop Stop Stop")

    """
    "Prediction im Test Set
    """

    y_prediction = np.zeros((len(y_test), 3))

    y_prediction[:,0] = clf.predict(X_test_scaled_imputed_selected)

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
    log_loss_value = log_loss(y_test, clf.predict_proba(X_test_scaled_imputed_selected), normalize=True)

""" Calculate feature importances """

    if ml_options['feature_selection_option'] == 0:
        feature_importances = clf.feature_importances_
    elif ml_options['feature_selection_option'] == 1:
        feature_importances = np.zeros((len(sel.get_support())))
        counter_features_selected = 0
        for number_features in range(len(sel.get_support())):
            if sel.get_support()[number_features] == True:
                feature_importances[number_features] = clf.feature_importances_[counter_features_selected]
                counter_features_selected += 1
            else:
                feature_importances[number_features] = 0

    elif ml_options['feature_selection_option'] == 3 or 4:
        feature_importances = np.zeros((len(rfe.support_)))
        counter_features_selected = 0
        for number_features in range(len(rfe.support_)):
            if rfe.support_[number_features] == True:
                feature_importances[number_features] = clf.feature_importances_[counter_features_selected]
                counter_features_selected += 1
            else:
                feature_importances[number_features] = 0

    elif ml_options['feature_selection_option'] == 5:
        feature_importances = np.zeros((len(sfm.get_support())))
        counter_features_selected = 0
        for number_features in range(len(sfm.get_support())):
            if sfm.get_support()[number_features] == True:
                feature_importances[number_features] = clf.feature_importances_[counter_features_selected]
                counter_features_selected += 1
            else:
                feature_importances[number_features] = 0


    fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test_scaled_imputed_selected)[:,1])
    mean_fpr = np.linspace(0, 1, 100)
    tprs = np.interp(mean_fpr, fpr, tpr)
    roc_auc = auc(fpr, tpr)
    fraction_positives, mean_predicted_value = calibration_curve(y_test, clf.predict_proba(X_test_scaled_imputed_selected)[:,1], n_bins=10)

"""""""""""""""""
"Permutationstest
"""""""""""""""""

save_option = Path(standardpath, 'individual_rounds', ml_options['name_model'], '_clf_round_', str(random_state_option))
if ml_options['save_clf_option'] == 1:
    with open(save_option, 'wb') as AutoPickleFile:
            pickle.dump((clf, y_test, y_train, X_train_imputed_scaled_selected, X_test_scaled_imputed_selected), AutoPickleFile)
    else:
        print("Clf wird nicht gespeichert")

    pvalue = 1
if ml_options['permutation_option']  == 1:
    counter_random = 0
    for j in range(ml_options['n_permutations_option']):
        print("\n Permutationstest: der aktuelle Durchgang ist %s" % (j))

        """
        "Permutierung Labels - Permutationstest
        """

        y_test_random = copy.deepcopy(y_test)
        y_test_random = np.ravel(y_test_random)
        y_test_random = shuffle(y_test_random, random_state=j)

        y_train_random = copy.deepcopy(y_train)
        y_train_random = np.ravel(y_train_random)
        y_train_random = shuffle(y_train_random, random_state=j)

        """
        "Random Forest Analyse - Permutationstest
        """

        if ml_options['balanced_split_option'] == 0 or 1 or 2:
            clf_perm = RandomForestClassifier(n_estimators= best_parameter["n_estimators"], criterion = best_parameter["criterion"], max_features= best_parameter["max_features"], max_depth= best_parameter["max_depth"], min_samples_split= best_parameter["min_samples_split"], min_samples_leaf= best_parameter["min_samples_leaf"], bootstrap= best_parameter["bootstrap"], oob_score=True, random_state=random_state_option)
        elif ml_options['balanced_split_option']  == 3:
            clf_perm = RandomForestClassifier(n_estimators= best_parameter["n_estimators"], criterion = best_parameter["criterion"], max_features= best_parameter["max_features"], max_depth= best_parameter["max_depth"], min_samples_split= best_parameter["min_samples_split"], min_samples_leaf= best_parameter["min_samples_leaf"], bootstrap= best_parameter["bootstrap"], oob_score=True, class_weight='balanced', random_state=random_state_option)
        elif ml_options['balanced_split_option']  == 4:
            clf_perm = BalancedRandomForestClassifier(n_estimators= best_parameter["n_estimators"], criterion = best_parameter["criterion"], max_features= best_parameter["max_features"], max_depth= best_parameter["max_depth"], min_samples_split= best_parameter["min_samples_split"], min_samples_leaf= best_parameter["min_samples_leaf"], bootstrap= best_parameter["bootstrap"], oob_score=True, random_state=random_state_option)
        elif ml_options['balanced_split_option']  == 5:
            clf_perm = RandomForestClassifier(n_estimators= best_parameter["n_estimators"], criterion = best_parameter["criterion"], max_features= best_parameter["max_features"], max_depth= best_parameter["max_depth"], min_samples_split= best_parameter["min_samples_split"], min_samples_leaf= best_parameter["min_samples_leaf"], bootstrap= best_parameter["bootstrap"], oob_score=True, class_weight=ml_options['rf_classes_class_weight'], random_state=random_state_option)
  
        y_prediction_perm = np.zeros((len(y_test_random), 3))
        y_prediction_perm[:,0] = clf_perm.predict(X_test_scaled_imputed_selected)
        y_prediction_perm[:,1] = y_test_random[:]

        counter_class1_correct_perm = 0
        counter_class2_correct_perm = 0
        counter_class1_incorrect_perm = 0
        counter_class2_incorrect_perm = 0

        for i in range(len(y_test_random)):
            if y_prediction_perm[i,0] == y_prediction_perm[i,1]:
                y_prediction_perm[i,2] = 1
                if y_prediction_perm[i,1] == 1:
                    counter_class1_correct_perm += 1
                else:
                    counter_class2_correct_perm += 1
            else:
                y_prediction_perm[i,2] = 0
                if y_prediction_perm[i,1] == 1:
                    counter_class1_incorrect_perm += 1
                else:
                    counter_class2_incorrect_perm += 1

        """
        "Check Significance - Permutationstest
        """
        accuracy_class1_perm = counter_class1_correct_perm / (counter_class1_correct_perm + counter_class1_incorrect_perm)
        accuracy_class2_perm = counter_class2_correct_perm / (counter_class2_correct_perm + counter_class2_incorrect_perm)
        balanced_accuracy_perm = (accuracy_class1_perm + accuracy_class2_perm) / 2

        if balanced_accuracy < balanced_accuracy_perm:
            counter_random += 1

    pvalue = (counter_random + 1)/(j + 1 + 1)


return accuracy, accuracy_class1, accuracy_class2, balanced_accuracy, oob_accuracy, log_loss_value, feature_importances, fpr, tpr, tprs, roc_auc, fraction_positives, mean_predicted_value, pvalue