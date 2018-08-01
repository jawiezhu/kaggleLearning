# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 17:10:03 2017

drop some columns
use 0 or 1 to replace the male and female


@author: jawiezhu
"""

import pandas as pd
import numpy as np
from MultiColumnLabelEncoder import *


def add_target_column(all_df):
    survived=all_df['Survived']
    all_df.drop(labels=['Survived'],axis=1,inplace=True)
    all_df.insert(all_df.columns.size, 'target', survived)


def drop_columns(all_df):
    #    drop name columns
    #all_df.drop(labels=['Name'], axis=1, inplace=True)
    # drop the column CABIN
    all_df.drop(['Cabin', 'Ticket'], axis=1, inplace=True)
    # drop the two NAN row of the Embarked
    #all_df.dropna(how='any', inplace=True)
    all_df['Embarked'].fillna(0.0, inplace=True)
    all_df['Fare'].fillna(0.0, inplace=True)

def fill_age(dataset):
    # Filling missing value of Age

    ## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp
    # Index of NaN age rows
    index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)

    for i in index_NaN_age:
        age_med = dataset["Age"].median()
        age_pred = dataset["Age"][(
                (dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (
                dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
        if not np.isnan(age_pred):
            dataset['Age'].iloc[i] = age_pred
        else:
            dataset['Age'].iloc[i] = age_med

    return dataset

def fill_non_age(all_df):
    age_df = all_df['Age']
    age_df.sort_values(ascending=False)
    age_0_12 = 0
    age_13_18 = 0
    age_19_30 = 0
    age_31_50 = 0
    age_51 = 0
    for item in age_df:
        if 0 < item <= 12:
            age_0_12 = age_0_12 + 1
        if 12 < item <= 18:
            age_13_18 = age_13_18 + 1
        if 18< item <= 30:
            age_19_30 = age_19_30 + 1
        if 30< item <= 50:
            age_31_50 = age_31_50 + 1
        if item > 51:
            age_51 = age_51 + 1

    #   there are a lot of people in the range 18-30,and the average of the people is 23,
    #   so use the average to fill the nan
    average_age = round(age_df.sum() / age_df.index.size)
    print('Average age is :', average_age)
    all_df['Age'].fillna(average_age, inplace=True)


def label_encoder(all_df):
    all_df = MultiColumnLabelEncoder(columns=['Sex', 'Embarked','Ticket','Cabin']).fit_transform(all_df)

    return all_df

def add_family_size(all_df):
    all_df['FamilySize'] = all_df['Parch'] + all_df['SibSp'] + 1
    #all_df['Single'] = all_df['FamilySize'].map(lambda s : 1 if s ==1 else 0)
    #all_df['SmallFamily'] = all_df['FamilySize'].map(lambda s: 1 if s==2 else 0)
    #all_df['MedFamily'] = all_df['FamilySize'].map(lambda s : 1 if 3<= s <= 4 else 0)
    #all_df['LargeFamily'] = all_df['FamilySize'].map(lambda s: 1 if s >=5 else 0)

def add_cabin_feature(all_df):
    all_df['Cabin'] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in all_df['Cabin']])
    #all_df = pd.get_dummies(all_df, columns=["Cabin"], prefix="Cabin")
    return all_df

def add_ticket_feature(all_df):
    Ticket = []
    for i in list(all_df.Ticket):
        if not i.isdigit():
            Ticket.append(i.replace(".", "").replace("/", "").strip().split(' ')[0])
        else:
            Ticket.append('X')
    all_df['Ticket'] = Ticket
    #all_df = pd.get_dummies(all_df, columns=["Ticket"], prefix="T")
    return all_df

def add_name_feature(all_df):
    df_title = [i.split(',')[1].split('.')[0].strip() for i in all_df["Name"]]
    all_df["Title"] = pd.Series(df_title)
    all_df["Title"] = all_df["Title"].replace(
        ['Lady', 'the Countess', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
        'Rare')
    all_df["Title"] = all_df["Title"].map(
        {"Master": 0, "Miss": 1, "Ms": 1, "Mme": 1, "Mlle": 1, "Mrs": 1, "Mr": 2, "Rare": 3})
    all_df["Title"] = all_df["Title"].astype(int)
    all_df.drop(labels=['Name'], axis=1, inplace=True)


def preprocess_for_train_data(TRAIN_PATH, RESULT_TRAIN_PATH):
    all_df = pd.read_csv(TRAIN_PATH)
    #fill_non_age(all_df)
    all_df = fill_age(all_df)
    add_family_size(all_df)
    all_df = add_cabin_feature(all_df)
    #all_df = add_ticket_feature(all_df)
    add_name_feature(all_df)
    add_target_column(all_df)
    all_df = label_encoder(all_df)
    drop_columns(all_df)
    all_df.to_csv(RESULT_TRAIN_PATH, index=False)

def preprocess_for_test_data(TEST_PATH, RESULT_TEST_PATH):
    all_df = pd.read_csv(TEST_PATH)
    #fill_non_age(all_df)
    all_df = fill_age(all_df)
    add_family_size(all_df)
    all_df = add_cabin_feature(all_df)
    #all_df = add_ticket_feature(all_df)
    add_name_feature(all_df)
    all_df = label_encoder(all_df)
    drop_columns(all_df)
    #all_df.drop(all_df.columns[len(all_df.columns) - 1], axis=1, inplace=True)
    all_df.to_csv(RESULT_TEST_PATH, index=False)