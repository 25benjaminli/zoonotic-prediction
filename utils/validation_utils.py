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
from data_utils import transform_data
"""
ROC curves
"""

def draw_roc_curve(model, X_test, y_test):
    X_test = transform_data(model, X_test)

    y_thing = y_test
    precision, recall, thresholds = precision_recall_curve(y_thing, model.predict_proba(X_test)[::,1])
    aaa = auc(recall, precision)
    print("precision recall: " + str(aaa))

    y_pred_proba = model.predict_proba(X_test)[::,1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_thing = roc_auc_score(y_test, y_pred_proba)

    # print(y_pred_proba)
    return fpr, tpr, auc_thing
    

def draw_roc_multiple(models, X_test, y_test):
    for key in models:
        fpr, tpr, auc = draw_roc_curve(models[key], X_test, y_test)
        plt.plot(fpr,tpr,label=f"{key}, auc="+str(auc))
        

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

        p1 = ax.bar([key for key in obj], [obj[key] for key in obj], width, align='center')
        # p2 = ax.bar(ind + width/2, [obj[key] for key in obj], width, label='Accuracy')
        vals = list(obj.values())
        for rect in range(len(p1)):
            print(vals[rect])
            height = p1[rect].get_height()
            ax.text(p1[rect].get_x() + p1[rect].get_width()/2., height,
                    f'{vals[rect]}',
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

def cross_validate(model, X_test, y_test):
    X_test = transform_data(model, X_test)

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise', verbose=1)
    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
    tn, fp, fn, tp = confusion_matrix(y_test, model.predict(X_test)).ravel()
    print(f"tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}")
    return mean(n_scores)

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

# test()
