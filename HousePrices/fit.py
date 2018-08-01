import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import GradientBoostingRegressor

RAW_TRAIN_FILE = r'./data/train.csv'
RAW_TEST_FILE = r'./data/test.csv'



train = pd.read_csv(RAW_TRAIN_FILE)
test = pd.read_csv(RAW_TEST_FILE)

print train
selected_features = ['Foundation', 'Heating', 'Electrical','SaleType',
                     'SaleCondition', 'GarageArea','YearRemodAdd','YearBuilt',
                     '1stFlrSF', '2ndFlrSF', 'TotalBsmtSF', 'BsmtUnfSF', 'CentralAir']

X_train = train[selected_features]
X_test = test[selected_features]
y_train = train['SalePrice']

#fill nan
X_train['Electrical'].fillna('SBrkr', inplace=True)
X_train['SaleType'].fillna('WD', inplace=True)
X_train['GarageArea'].fillna(X_train['GarageArea'].mean(), inplace=True)
X_train['TotalBsmtSF'].fillna(X_train['TotalBsmtSF'].mean(), inplace=True)
X_train['BsmtUnfSF'].fillna(X_train['BsmtUnfSF'].mean(), inplace=True)
X_test['Electrical'].fillna('SBrkr', inplace=True)
X_test['SaleType'].fillna('WD', inplace=True)
X_test['GarageArea'].fillna(X_test['GarageArea'].mean(), inplace=True)
X_test['TotalBsmtSF'].fillna(X_test['TotalBsmtSF'].mean(), inplace=True)
X_test['BsmtUnfSF'].fillna(X_test['BsmtUnfSF'].mean(), inplace=True)


print X_train.info()
print X_test.info()

dict_vec = DictVectorizer(sparse=False)
X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
X_test = dict_vec.transform(X_test.to_dict(orient='record'))

rfRegressor = GradientBoostingRegressor()
rfRegressor.fit(X_train, y_train)
rfRegressor_predict = rfRegressor.predict(X_test)

rfRegressor_submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': rfRegressor_predict})
rfRegressor_submission.to_csv('submission_v1.csv', index=False)