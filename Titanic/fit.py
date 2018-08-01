# -*- coding:utf-8 -*-

import numpy as np

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold,cross_val_score,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier,VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier


def start_to_fit(X, y):
    classifiers = [
        KNeighborsClassifier(3),
        SVC(probability=True),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
        LogisticRegression()]

    res_cols = ['Classifier','Accuracy']
    res = pd.DataFrame(columns = res_cols)

    data_set = StratifiedShuffleSplit(n_splits=10, test_size=0.3, train_size=0.7, random_state=0)

    accuracy_dic ={}


    for train_index, test_index in data_set.split(X, y):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for clf in classifiers:
            name = clf.__class__.__name__
            clf.fit(X_train, y_train)
            #train_predictions = clf.predict(X_test)
            accuracy = accuracy_score(y_test, clf.predict(X_test))
            if name in accuracy_dic:
                accuracy_dic[name] += accuracy
            else:
                accuracy_dic[name] = accuracy

    for clf in accuracy_dic:
        accuracy_dic[clf] = accuracy_dic[clf] / 10.0
        res_entry = pd.DataFrame([[clf, accuracy_dic[clf]]], columns=res_cols)
        res = res.append(res_entry)

    print res

def fit_another_approach(X, y):
    kfold = StratifiedKFold(n_splits=10)
    random_state = 2
    classifiers = []
    classifiers.append(SVC(random_state = random_state))
    classifiers.append(DecisionTreeClassifier(random_state = random_state))
    classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state), random_state= random_state, learning_rate=0.1))
    classifiers.append(RandomForestClassifier(random_state=random_state))
    classifiers.append(ExtraTreesClassifier(random_state=random_state))
    classifiers.append(GradientBoostingClassifier(random_state=random_state))
    classifiers.append(MLPClassifier(random_state = random_state))
    classifiers.append(KNeighborsClassifier())
    classifiers.append(LogisticRegression(random_state = random_state))
    classifiers.append(LinearDiscriminantAnalysis())

    cv_results = []
    for classifier in classifiers:
        cv_results.append(cross_val_score(classifier, X,y, scoring = "accuracy", cv = kfold, n_jobs = 4))

    cv_means = []
    cv_std = []

    for cv_result in cv_results:
        cv_means.append(cv_result.mean())
        cv_std.append(cv_result.std())

    cv_res = pd.DataFrame(
        {"CrossValMeans": cv_means, "CrossValerrors": cv_std, "Algorithm": ["SVC", "DecisionTree", "AdaBoost",
                                                                            "RandomForest", "ExtraTrees",
                                                                            "GradientBoosting",
                                                                            "MultipleLayerPerceptron", "KNeighboors",
                                                                            "LogisticRegression",
                                                                            "LinearDiscriminantAnalysis"]})
    print cv_res


def fit_adaboost(X,y):
    # Adaboost
    kfold = StratifiedKFold(n_splits=10)
    DTC = DecisionTreeClassifier()

    adaDTC = AdaBoostClassifier(DTC, random_state=7)

    ada_param_grid = {"base_estimator__criterion": ["gini", "entropy"],
                      "base_estimator__splitter": ["best", "random"],
                      "algorithm": ["SAMME", "SAMME.R"],
                      "n_estimators": [1, 2],
                      "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1.5]}

    gsadaDTC = GridSearchCV(adaDTC, param_grid=ada_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

    gsadaDTC.fit(X, y)

    ada_best = gsadaDTC.best_estimator_
    print ada_best, gsadaDTC.best_score_
    return ada_best

def fit_extratree(X, y):
    kfold = StratifiedKFold(n_splits=10)
    # ExtraTrees
    ExtC = ExtraTreesClassifier()

    ## Search grid for optimal parameters
    ex_param_grid = {"max_depth": [None],
                     "max_features": [1, 3, 10],
                     "min_samples_split": [2, 3, 10],
                     "min_samples_leaf": [1, 3, 10],
                     "bootstrap": [False],
                     "n_estimators": [100, 300],
                     "criterion": ["gini"]}

    gsExtC = GridSearchCV(ExtC, param_grid=ex_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

    gsExtC.fit(X, y)

    ExtC_best = gsExtC.best_estimator_

    # Best score
    print ExtC_best, gsExtC.best_score_
    return ExtC_best

def fit_rf(X, y):
    kfold = StratifiedKFold(n_splits=10)
    # RFC Parameters tunning
    RFC = RandomForestClassifier()

    ## Search grid for optimal parameters
    rf_param_grid = {"max_depth": [None],
                     "max_features": [1, 3, 10],
                     "min_samples_split": [2, 3, 10],
                     "min_samples_leaf": [1, 3, 10],
                     "bootstrap": [False],
                     "n_estimators": [100, 300],
                     "criterion": ["gini"]}

    gsRFC = GridSearchCV(RFC, param_grid=rf_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

    gsRFC.fit(X, y)

    RFC_best = gsRFC.best_estimator_

    # Best score
    print RFC_best, gsRFC.best_score_
    return RFC_best

def fit_xgboost(X, y):
    # Gradient boosting tunning
    kfold = StratifiedKFold(n_splits=10)
    GBC = GradientBoostingClassifier()
    gb_param_grid = {'loss': ["deviance"],
                     'n_estimators': [100, 200, 300],
                     'learning_rate': [0.1, 0.05, 0.01],
                     'max_depth': [4, 8],
                     'min_samples_leaf': [100, 150],
                     'max_features': [0.3, 0.1]
                     }

    gsGBC = GridSearchCV(GBC, param_grid=gb_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

    gsGBC.fit(X, y)

    GBC_best = gsGBC.best_estimator_

    # Best score
    print GBC_best, gsGBC.best_score_
    return GBC_best

def fit_svc(X, y):
    kfold = StratifiedKFold(n_splits=10)
    SVMC = SVC(probability=True)
    svc_param_grid = {'kernel': ['rbf'],
                      'gamma': [0.001, 0.01, 0.1, 1],
                      'C': [1, 10, 50, 100, 200, 300, 1000]}

    gsSVMC = GridSearchCV(SVMC, param_grid=svc_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

    gsSVMC.fit(X, y)

    SVMC_best = gsSVMC.best_estimator_

    # Best score
    print SVMC_best, gsSVMC.best_score_
    return SVMC_best


def fit_lr(X, y):
    kfold = StratifiedKFold(n_splits=10)
    lrc = LogisticRegression()
    lr_param_grid = {
        'C': [1, 10, 50, 100, 200, 300, 1000],
        'max_iter':[100,300,500]
    }

    lr_m = GridSearchCV(lrc, param_grid= lr_param_grid, cv = kfold, scoring="accuracy", n_jobs = 4, verbose = 1)

    lr_m.fit(X, y)
    lr_best = lr_m.best_estimator_

    print lr_best, lr_m.best_score_
    return lr_best


def voting_fit(X, y, RESULT_TEST_PATH,RESULT_PATH):
    ada_best = fit_adaboost(X, y)
    extratree_best = fit_extratree(X, y)
    rf_best = fit_rf(X, y)
    gbdt_best = fit_xgboost(X, y)
    svc_best = fit_svc(X, y)
    lr_best = fit_lr(X, y)

    votingC = VotingClassifier(estimators=[('rfc', rf_best), ('extc', extratree_best),('lr',lr_best),
                                            ('adac', ada_best), ('gbc', gbdt_best)], voting='soft',
                               n_jobs=4)
    votingC.fit(X, y)

    test_df = pd.read_csv(RESULT_TEST_PATH)
    test = np.array(test_df)

    #test_Survived = pd.Series(votingC.predict(test), name="Survived")

    result = votingC.predict(test)
    test_df.insert(test_df.columns.size, 'Survived', result)

    test_df = test_df[['PassengerId', 'Survived']]
    test_df['PassengerId'] = test_df['PassengerId'].apply(np.int64)
    test_df.to_csv(RESULT_PATH, index=False)
    print("finish!")

def get_best_rf(X, y):
    parameter_grid = {
        'max_depth': [4, 6, 8],
        'n_estimators': [50, 100,200],
        'max_features': ['sqrt', 'auto', 'log2'],
         "min_samples_split": [2, 3, 10],
         "min_samples_leaf": [1, 3, 10],
        'bootstrap': [True, False],
    }
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(n_splits=10)

    grid_search = GridSearchCV(forest,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validation)

    grid_search.fit(X, y)
    model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))

def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)

def fit_with_rf(X,y, RESULT_TEST_PATH, RESULT_PATH):
    parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50,
                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}

    model = RandomForestClassifier(**parameters)
    model.fit(X, y)
    print (compute_score(model, X, y, scoring='accuracy'))

    rf = RandomForestClassifier(criterion='gini',
                                n_estimators=700,
                                min_samples_split=16,
                                min_samples_leaf=1,
                                max_features='auto',
                                oob_score=True,
                                random_state=1,
                                n_jobs=-1)
    rf.fit(X, y)
    print (compute_score(rf, X, y, scoring='accuracy'))

    test_df = pd.read_csv(RESULT_TEST_PATH)
    test = np.array(test_df)

    result = rf.predict(test)
    test_df.insert(test_df.columns.size, 'Survived', result)

    test_df = test_df[['PassengerId', 'Survived']]
    test_df['PassengerId'] = test_df['PassengerId'].apply(np.int64)
    test_df.to_csv(RESULT_PATH, index=False)
    print("finish!")



