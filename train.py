#!/usr/bin/env python
# coding: utf-8

import pickle

import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
from sklearn.metrics import roc_auc_score

# parameters

C = 1.0
n_splits = 5
output_file = f'model_C={C}.bin'

# data preparation

df = pd.read_csv('anime_dataset.csv')

df = df.drop(columns=['related_anime', 'opening_themes', 'ending_themes', 'year', 'season', 'demographics', 'background', 'licensors', 'broadcast'])

columns_to_fill = ['producers', 'studios', 'genres', 'themes']

for col in columns_to_fill:
    df[col] = df[col].fillna('Unknown')
    
def create_success_target(row):
    conditions = [
        row['score'] > 6 if pd.notnull(row['score']) else True,  # Ignore if null
        row['scored_by'] > 8000 if pd.notnull(row['scored_by']) else True,  # Ignore if null
        row['popularity'] < 10000 if pd.notnull(row['popularity']) else True,  # Ignore if null
        row['members'] > 8000 if pd.notnull(row['members']) else True,  # Ignore if null
        row['favorites'] > 100 if pd.notnull(row['favorites']) else True,  # Ignore if null
        row['rank'] < 10000 if pd.notnull(row['rank']) else True # Ignore if null
    ]
    return 1 if all(conditions) else 0

df['success'] = df.apply(create_success_target, axis=1)

df = df.drop(columns=['score', 'scored_by', 'popularity', 'members', 'favorites', 'rank'])

df['type'] = df['type'].fillna(df['type'].mode()[0])

df['episodes'] = df['episodes'].fillna(df['episodes'].median())

df['rating'] = df['rating'].fillna(df['rating'].mode()[0])

df['synopsis'] = df['synopsis'].fillna("No synopsis available")

def parse_duration(duration):
    match = re.search(r'(?:(\d+)\s*hr)?\s*(\d+)\s*min', duration)
    if match:
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        return hours * 60 + minutes
    return None 


df['duration_minutes'] = df['duration'].apply(parse_duration)

median_duration_by_type = df.groupby('type')['duration_minutes'].median()

df['duration_minutes'] = df.apply(
    lambda row: median_duration_by_type[row['type']] if pd.isnull(row['duration_minutes']) else row['duration_minutes'],
    axis=1
)

df.drop(columns=['duration'], inplace=True)

df['synopsis_length'] = df['synopsis'].apply(len)

df.drop(columns=['synopsis', 'airing', 'title', 'id'], inplace=True)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

categorical = ['type', 'source', 'status', 'rating', 'studios', 'genres', 'themes', 'producers']
numerical = ['episodes', 'duration_minutes', 'synopsis_length']


# Training and Prediction Functions

def train(df_train, y_train):
    dicts = df_train[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    dtrain = xgb.DMatrix(X_train, label=y_train)

    xgb_params = {
        'eta': 0.1,
        'max_depth': 4,
        'min_child_weight': 1,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'gpu_id': 0,
        'nthread': 8,
        'seed': 1,
        'verbosity': 1,
    }

    model = xgb.train(xgb_params, dtrain, num_boost_round=500)
    return dv, model

def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    X = dv.transform(dicts)
    dmatrix = xgb.DMatrix(X)
    y_pred = model.predict(dmatrix)
    return y_pred


# Setting up KFold Cross-validation

print(f'Doing validation with C={C}')
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(df_full_train)):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.success.values
    y_val = df_val.success.values

    dv, model = train(df_train, y_train)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
    print(f'AUC on fold {fold} is {auc}')

print('Validation results:')
print('%.3f +- %.3f' % (np.mean(scores), np.std(scores)))

# Training the final model

print('Training the final model...')
dv, model = train(df_full_train, df_full_train.success.values)
y_pred = predict(df_test, dv, model)

y_test = df_test.success.values
auc = roc_auc_score(y_test, y_pred)
print(f'AUC on test set: {auc}')

# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'The model is saved to {output_file}')