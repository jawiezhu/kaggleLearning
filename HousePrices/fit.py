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

#X_train = train[selected_features]
#X_test = test[selected_features]

X_train = train[-((train.SalePrice<200000) & (train.GrLivArea>40000))]
X_test = test
y_train = train['SalePrice']

#check the null
for col in train.columns:
    if train[col].isnull().sum() > 0:
        print col, train[col].isnull().sum()



X_train = train.drop(["MiscFeature", "Id", "PoolQC", "Alley", "Fence","GarageFinish",
                      "KitchenAbvGr", "EnclosedPorch", "MSSubClass", "OverallCond", "YrSold", "LowQualFinSF",
                      "MiscVal","BsmtHalfBath", "BsmtFinSF2", "3SsnPorch", "MoSold", "PoolArea"], axis=1)

X_test = test.drop(["MiscFeature", "Id", "PoolQC", "Alley", "Fence","GarageFinish",
                      "KitchenAbvGr", "EnclosedPorch", "MSSubClass", "OverallCond", "YrSold", "LowQualFinSF",
                      "MiscVal","BsmtHalfBath", "BsmtFinSF2", "3SsnPorch", "MoSold", "PoolArea"], axis=1)

all_data = pd.concat((X_train,X_test))



#fill nan
for col in train.columns:
    if train[col].isnull().sum()>0:
        if train[col].dtypes == 'object':
            val = all_data[col].dropna().value_counts().idxmax()
            train[col] = train[col].fillna(val)
        else:
            val = all_data[col].dropna().mean()
            train[col] = train[col].fillna(val)

for col in test.columns:
    if test[col].isnull().sum() >0:
        if test[col].dtypes == 'object':
            val = all_data[col].dropna().value_counts().idxmax()
            test[col] = test[col].fillna(val)
        else:
            val = all_data[col].dropna().mean()
            test[col] = test[col].fillna(val)


'''
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

'''

for col in all_data.select_dtypes(include=[object]).columns:
    train[col] = train[col].astype('category', categories = all_data[col].dropna().unique())





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