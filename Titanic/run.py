# -*- coding:utf-8 -*-

from get_data_report import *
from preprocess_for_data import *
from fit_for_data import *
from sklearn.ensemble import GradientBoostingClassifier,VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score


RAW_TRAIN_FILE = r'./data/train.csv'
RAW_TEST_FILE = r'./data/test.csv'
RESULT_TRAIN_PATH = r'./data/res_train.csv'
RESULT_TEST_PATH = r'./data/res_test.csv'
RESULT_PATH = r'./data/gender_submission.csv'


def cleaning_data():
    get_data_report(RAW_TRAIN_FILE)
    preprocess_for_train_data(RAW_TRAIN_FILE, RESULT_TRAIN_PATH)
    preprocess_for_test_data(RAW_TEST_FILE, RESULT_TEST_PATH)


def get_data_from_file():
    train_df = pd.read_csv(RESULT_TRAIN_PATH)
    X = np.array(train_df.loc[:, train_df.columns != 'target'])
    y = np.array(train_df.iloc[:, -1])

    #start_to_fit(X, y)
    #fit_another_approach(X, y)
    #voting_fit(X,y,RESULT_TEST_PATH, RESULT_PATH)
    fit_with_rf(X, y, RESULT_TEST_PATH, RESULT_PATH)

def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)


def predict_test():
    train_df = pd.read_csv(RESULT_TRAIN_PATH)
    X = np.array(train_df.loc[:, train_df.columns != 'target'])
    y = np.array(train_df.iloc[:, -1])


    parameters ={
        'learning_rate': [0.01,0.15],
        'n_estimators': [100, 200,300,500],
        'max_depth': [5, 7]
    }

    clf = GridSearchCV(GradientBoostingClassifier(), parameters)
    clf.fit(X,y)
    print(clf.best_score_)
    print(clf.best_params_)


    test_df = pd.read_csv(RESULT_TEST_PATH)
    test = np.array(test_df)
    new_clf = GradientBoostingClassifier(learning_rate=0.01, max_depth=3, n_estimators=100)
    #new_clf = LinearDiscriminantAnalysis()
    new_clf.fit(X,y)
    print compute_score(new_clf, X, y)
    result = new_clf.predict(test)
    test_df.insert(test_df.columns.size, 'Survived', result)

    test_df = test_df[['PassengerId', 'Survived']]
    test_df['PassengerId'] = test_df['PassengerId'].apply(np.int64)
    test_df.to_csv(RESULT_PATH, index=False)
    print('finish!')




if __name__ == '__main__':
    cleaning_data()
    get_data_from_file()
    #predict_test()
