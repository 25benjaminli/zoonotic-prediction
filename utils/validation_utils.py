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
from model_utils import transform_data
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

    print(y_pred_proba)
    plt.plot(fpr,tpr,label="AUC="+str(auc_thing))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()


"""
FEATURE IMPORTANCE GRAPHS
"""
def draw_feature_importances_lr(lrmodel, X_test, y_test):
    results = lrmodel.coef_[0]

    l = zip([x for x in X_test.columns.values],results)
    l = list(l)
    res = sorted(l, key= lambda x: x[1])

    plt.xticks(rotation=90)
    plt.tick_params(axis='x', which='major', labelsize=5)

    plt.bar([x[0] for x in res[230:]], [x[1] for x in res[230:]])
    plt.show()

def draw_feature_importances_gradBoost(gradBoost, X_test, y_test):
    # ALWAYS reset X columns to the right order
    pass


"""
CROSS VALIDATION FUNCTIONS
"""

def cross_validate_XGBoost(xgboost, X_test, y_test):
    cols_when_model_builds = xgboost.get_booster().feature_names
    X_test=X_test[cols_when_model_builds]

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(xgboost, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
    tn, fp, fn, tp = confusion_matrix(y_test, xgboost.predict(X_test)).ravel()
    print(f"tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}")


def cross_validate_gradBoost(gradBoost, X_test, y_test):
    cols_when_model_builds = gradBoost.feature_names_in_
    X_test=X_test[cols_when_model_builds]

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(gradBoost, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
    tn, fp, fn, tp = confusion_matrix(y_test, gradBoost.predict(X_test)).ravel()
    print(f"tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}")


def cross_validate_normal(model, X_test, y_test):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
    tn, fp, fn, tp = confusion_matrix(y_test, model.predict(X_test)).ravel()
    print(f"tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}")

def cross_validate(model, X_test, y_test):
    X_test = transform_data(model, X_test)

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
    tn, fp, fn, tp = confusion_matrix(y_test, model.predict(X_test)).ravel()
    print(f"tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}")
