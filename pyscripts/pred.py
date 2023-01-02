from itertools import permutations, product

import tqdm

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import os
from os import path
import numpy as np
import matplotlib.pyplot as plt

from warnings import simplefilter
from collections import OrderedDict

import pickle
import sys
sys.path.append('..')
from utils import data_utils


simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
simplefilter(action='ignore', category=FutureWarning)

def assess_model(model, threshold, dataset):
    ds = dataset['merged']['regular-4']
    # mergedX = merged['X_test']
    # mergedY = merged['y_test']
    X_train, y_train, X_test, y_test = ds['X_train'], ds['y_train'], ds['X_test'], ds['y_test']

    X = pd.concat([X_train, X_test])
    y = np.concatenate([ds['y_train'], ds['y_test']], axis=0)

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y, test_size=0.2, random_state=42)
    bp = {
        'ans': {
            '0': {'1000': 0, '3000': 0, '5000': 0, '10000': 0, '>10000': 0},
            '1': {'1000': 0, '3000': 0, '5000': 0, '10000': 0, '>10000': 0},
        },
        'preds': {
            '0': {'1000': 0, '3000': 0, '5000': 0, '10000': 0, '>10000': 0},
            '1': {'1000': 0, '3000': 0, '5000': 0, '10000': 0, '>10000': 0},
        }
    }
    ds = dataset['merged']['normalized-4']
    # mergedX = merged['X_test']
    # mergedY = merged['y_test']
    X_train, y_train, X_test, y_test = ds['X_train'], ds['y_train'], ds['X_test'], ds['y_test']

    X = pd.concat([X_train, X_test])
    y = np.concatenate([ds['y_train'], ds['y_test']], axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for r in tqdm.tqdm(range(len(X_test))):
        data = pd.DataFrame([X_test.iloc[r].to_list()], columns=X_test.iloc[r].index)
        # print(data)
        pred_proba = model.predict_proba(data)[:,1]
        pred = '1' if pred_proba >= threshold else '0'
        s = X_test_reg.iloc[r].sum()
        res = str(y_test_reg[r])
        if s < 1000:
            if pred == res:
                bp['preds'][res]['1000']+=1
            bp['ans'][res]['1000']+=1
        elif s < 3000:
            if pred == res:
                bp['preds'][res]['3000']+=1
            bp['ans'][res]['3000']+=1
        elif s < 5000:
            if pred == res:
                bp['preds'][res]['5000']+=1
            bp['ans'][res]['5000']+=1
        elif s <= 10000:
            if pred == res:
                bp['preds'][res]['10000']+=1
            bp['ans'][res]['10000']+=1
        elif s > 10000:
            if pred == res:
                bp['preds'][res]['>10000']+=1
            bp['ans'][res]['>10000']+=1

    # for r in tqdm.tqdm(range(len(X_test))):
    #     data = pd.DataFrame([X_test.iloc[r].to_list()], columns=mergedXNormed.iloc[r].index)
    #     # print(data)
    #     pred_proba = model.predict_proba(data)[:,1]
    #     pred = 1 if pred_proba >= 0.5 else 0
    print(bp['ans'])
    print(bp['preds'])
    # print accuracies for each bp length
    for k in bp['ans']:
        for i in bp['ans'][k]:
            print(k, i, bp['preds'][k][i]/bp['ans'][k][i])

    # print total accuracy over all bp lengths
    total = 0
    total_ans = 0
    for k in bp['ans']:
        for i in bp['ans'][k]:
            total += bp['preds'][k][i]
            total_ans += bp['ans'][k][i]
    print(total/total_ans)
    # print total number of samples
    print(total_ans)
    print(total)

dataset = data_utils.retrieveAllDatasets(dir='../data')


# model = pickle.load(open('models/curr_models/final_merged_stackingcv.pkl', 'rb'))
model = pickle.load(open('../models/curr_models/final_merged_stackingcv.pkl', 'rb'))
knn = pickle.load(open('../models/curr_models/knn.pkl', 'rb'))
rf = pickle.load(open('../models/curr_models/randforest.pkl', 'rb'))
mlp = pickle.load(open('../models/curr_models/mlpClassifier.pkl', 'rb'))
xgboost = pickle.load(open('../models/curr_models/xgb-gridsearch.pkl', 'rb'))
svm = pickle.load(open('../models/curr_models/svm.pkl', 'rb'))
xgbgrid = pickle.load(open('../models/curr_models/xgb-grid-real.pkl', 'rb'))

# assess_model(model=model, threshold=0.5, dataset=dataset)
assess_model(model=svm, threshold=0.1, dataset=dataset)