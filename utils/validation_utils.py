# ALWAYS reset X columns to the right order
from Bio import SeqIO, Entrez
import os
from urllib.error import HTTPError
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from itertools import permutations, product
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score
import tqdm
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split, cross_val_score

from numpy import mean
from numpy import std
import pickle
from os import path
from sklearn.model_selection import cross_val_score
from warnings import simplefilter
from collections import OrderedDict
from sklearn.metrics import accuracy_score, auc, confusion_matrix, balanced_accuracy_score, precision_recall_curve, auc, roc_curve, roc_auc_score

from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
from data_utils import transform_data, retrieveAllDatasets
from sklearn.model_selection import KFold
from sklearn.metrics import plot_roc_curve
import time
"""
ROC curves
"""

"""
Tested cross val with auc curve, seems to work well
"""
def draw_avg_roc_curve(model, name, X, y, multiple=False):
    # done w/ the help of https://stats.stackexchange.com/questions/186337/average-roc-for-repeated-10-fold-cross-validation-with-probability-estimates
    
    splits = 5
    kf = KFold(n_splits=splits)
    kf.get_n_splits(X)

    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    
    avgauc = 0
    for train, test in kf.split(X):
        # y_pred_proba = model.predict_proba(X.iloc[test])[::,1]
        # fpr, tpr, _ = roc_curve(y[test], y_pred_proba)
        # auc_thing = roc_auc_score(y[test], y_pred_proba)
        # print("roc: " + str(auc_thing))
        # print(train)
        # print(test)
        model = model.fit(X.iloc[train], y[train])
        print("fit done")
        y_score = model.predict_proba(X.iloc[test])
        fpr, tpr, _ = roc_curve(y[test], y_score[:, 1])
        auc = roc_auc_score(y[test], y_score[:,1])
        if not multiple:
            # plot variance
            plt.plot(fpr, tpr, alpha=0.15)
        # print("before, ", tpr)
        tpr = np.interp(base_fpr, fpr, tpr) # interpolate between fpr and tpr
        tpr[0] = 0.0

        # print("after, ", tpr)
        tprs.append(tpr)
        # print(auc, accuracy_score(y[test], model.predict(X.iloc[test])))
        avgauc += auc
        print("auc split: ", auc)
    
    avgauc /= splits

    tprs = np.array(tprs)
    
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)
    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    print(avgauc)

    plt.plot(base_fpr, mean_tprs, label=f"{name} avg. auc="+str(round(avgauc, 3)))
    # fill in areas between
    if not multiple:
        plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)
    
    if not multiple:
        plt.show()

def draw_avg_roc_multiple(models, X_test, y_test):
    plt.plot([0, 1], [0, 1],'r--') # plot line for comparison
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    # plt.axes().set_aspect('equal', 'datalim')

    for key in models:
        t = time.time()
        print(models[key])
        draw_avg_roc_curve(models[key], key, X_test, y_test, multiple=True)
        print("time for CV: ", time.time() - t)
        # plt.legend(loc='best')
    plt.legend(loc='best', fontsize=8)
    
    plt.show()

def draw_roc_curve(model, X_test, y_test, multiple=False):
    # implement Kfold cross validation before drawing ROC curve
    X_test = transform_data(model, X_test)
    y_thing = y_test
    precision, recall, thresholds = precision_recall_curve(y_thing, model.predict_proba(X_test)[::,1])
    aaa = auc(recall, precision)
    print("precision recall: " + str(aaa))

    y_pred_proba = model.predict_proba(X_test)[::,1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_thing = roc_auc_score(y_test, y_pred_proba)
    print("roc: " + str(auc_thing))
    if not multiple:
        plt.plot(fpr,tpr,label="auc="+str(round(auc_thing, 3)))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        plt.show()

    print(y_pred_proba)
    return fpr, tpr, auc_thing




def draw_roc_multiple(models, X_test, y_test):
    for key in models:
        fpr, tpr, auc = draw_roc_curve(models[key], X_test, y_test, multiple=True)
        plt.plot(fpr,tpr,label=f"{key}, auc="+str(round(auc, 3)))
    plt.legend(loc='best')

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()

def draw_accuracies(models, X_test, y_test, obj=None):
    if obj is None:
        plt.tick_params(axis='x', which='major', labelsize=10)
        plt.xticks(rotation=90)

        plt.bar([key for key in models], [cross_validate(models[key], X_test, y_test) for key in models])
        plt.show()
    else:
        fig, ax = plt.subplots()
        width = 0.35
        plt.xticks(rotation=45)
        assert type(obj).__name__=='OrderedDict'

        # sort dictionary by value
        p1 = ax.bar([key for key in obj], [round(obj[key]*100, 3) for key in obj], width, align='center')
        # p2 = ax.bar(ind + width/2, [obj[key] for key in obj], width, label='Accuracy')
        vals = list(obj.values())
        for rect in range(len(p1)):
            print(vals[rect])
            height = p1[rect].get_height()
            ax.text(p1[rect].get_x() + p1[rect].get_width()/2., height,
                    f'{round(vals[rect]*100, 3)}%',
                    ha='center', va='bottom')

        ax.set_ylabel('Cross-Validated Accuracy')
        ax.set_title('Scores by model')
        ax.set_xlabel('Model Type')

        # ax.set_xticks(ind, labels=["1", "2", "3", "4", "5"])
        ax.legend(loc=4)


"""
FEATURE IMPORTANCE GRAPHS
"""

def draw_feature_importances(model, X_test):
    # ALWAYS reset X columns to the right order
    l = type(model)
    results = None
    if l.__name__ == "GradientBoostingClassifier" or l.__name__ == "XGBClassifier":
        # print("gradboost")
        results = model.feature_importances_
    elif l.__name__ == "LogisticRegression":
        # print("xgboost")
        results = model.coef_[0]

    elif l.__name__ == "BalancedBaggedClassifier":
        # do it for RF and KNN?
        results = np.mean([est.steps[1][1].feature_importances_ for est in model.estimators_], axis=0)
    
    
    l = zip([x for x in X_test.columns.values],results)
    l = list(l)
    res = sorted(l, key= lambda x: x[1])

    plt.xticks(rotation=90)
    plt.tick_params(axis='x', which='major', labelsize=10)

    plt.bar([x[0] for x in res[230:]], [x[1] for x in res[230:]])
    plt.show()


"""
CROSS VALIDATION FUNCTIONS
"""

def cross_validate(model, X, y, verb = 1):
    X = transform_data(model, X)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise', verbose=verb)
    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
    
    return round(mean(n_scores), 3)

def get_conf_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)

def cross_validate_multiple(models, X_test, y_test):
    obj = {}
    for key in models:
        obj[key] = cross_validate(models[key], X_test, y_test)
    
    return obj
    

def test():
    obj = OrderedDict({
    "nardus": 0.98,
    "xgboost": 0.99,
    "gradboost": 0.98,
    "randomforest": 0.96,
    "logisticregression": 0.9,
    "mlp": 0.97,
    "svm": 0.94,
    "regnard": 0.94
    })
    fig, ax = plt.subplots()
    width = 0.35
    plt.xticks(rotation=90)
    assert type(obj).__name__=='OrderedDict'

    p1 = ax.bar([key for key in obj], [obj[key] for key in obj], width, align='center')
    # p2 = ax.bar(ind + width/2, [obj[key] for key in obj], width, label='Accuracy')
    vals = list(obj.values())
    for rect in range(len(p1)):
        print(vals[rect])
        height = p1[rect].get_height()
        ax.text(p1[rect].get_x() + p1[rect].get_width()/2.,height,
                f'{vals[rect]}',
                ha='center', va='bottom')

    ax.set_ylabel('Cross-Validated Accuracy')
    ax.set_title('Scores by model')
    ax.set_xlabel('Model Type')

    # ax.set_xticks(ind, labels=["1", "2", "3", "4", "5"])
    ax.legend(loc=4)

    # Label with label_type 'center' instead of the default 'edge'
    # ax.bar_label(p1, label_type='center')

    plt.show()

# model = pickle.load(open("../models/curr_models/stacking.pkl", "rb"))
# datasets = retrieveAllDatasets(dir="../data")
# ds = datasets['merged']['normalized-4']
# draw_avg_roc_curve(model, "stacked", pd.concat([ds['X_train'], ds['X_test']], axis=0), np.concatenate([ds['y_train'], ds['y_test']], axis=0))