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
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split, cross_val_score, cross_validate

from numpy import mean
from numpy import std
import pickle
from os import path
from sklearn.model_selection import cross_val_score
from warnings import simplefilter
from collections import OrderedDict
from sklearn.metrics import accuracy_score, auc, confusion_matrix, balanced_accuracy_score, precision_recall_curve, auc, roc_curve, roc_auc_score, recall_score, precision_score, f1_score, brier_score_loss

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
    plt.ylim(0.50, 1.01)
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

    if name.lower() == "ensemble":
        plt.plot(base_fpr, mean_tprs, label=f"{name}", color="red")
    else:
        plt.plot(base_fpr, mean_tprs, label=f"{name}")
    # fill in areas between
    if not multiple:
        plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)
    
    if not multiple:
        plt.show()
    
    return round(avgauc, 3)

def draw_avg_roc_multiple(models, X_test, y_test):
    plt.plot([0, 1], [0, 1],'r--') # plot line for comparison
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    # plt.axes().set_aspect('equal', 'datalim')
    l = []
    for key in models:
        t = time.time()
        print(models[key])
        l.append(draw_avg_roc_curve(models[key], key, X_test, y_test, multiple=True))
        print("time for CV: ", time.time() - t)
        # plt.legend(loc='best')
    plt.legend(loc='best', fontsize=8)
    
    plt.show()
    return l

def draw_roc_curve(model, name, X_test, y_test, multiple=False):
    # implement Kfold cross validation before drawing ROC curve
    plt.ylim(0.50, 1.01)
    
    X_test = transform_data(model, X_test)
    y_thing = y_test
    precision, recall, thresholds = precision_recall_curve(y_thing, model.predict_proba(X_test)[:,1])
    aaa = auc(recall, precision)
    print("precision recall: " + str(aaa))

    y_pred_proba = model.predict_proba(X_test)[::,1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_thing = roc_auc_score(y_test, y_pred_proba)
    print("roc: " + str(auc_thing))
    
    if not multiple:
        
        # label="auc="+str(round(auc_thing, 3)
        plt.plot(fpr,tpr)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        plt.show()

    print(y_pred_proba)
    return fpr, tpr, auc_thing


def draw_prec_roc_curve(model, name, X_test, y_test, multiple=False):
    # implement Kfold cross validation before drawing ROC curve
    plt.ylim(0.50, 1.01)
    
    X_test = transform_data(model, X_test)
    y_thing = y_test
    precision, recall, thresholds = precision_recall_curve(y_thing, model.predict_proba(X_test)[:,1])
    aaa = auc(recall, precision)
    print("precision recall: " + str(aaa))

    if not multiple:
        
        # label="auc="+str(round(auc_thing, 3)
        plt.plot(recall,precision)
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.legend(loc=4)
        plt.show()

    return precision, recall

def draw_prec_roc_curve_multiple(models, X_test, y_test):
    for key in models:
        precision, recall = draw_prec_roc_curve(models[key], key, X_test, y_test, multiple=True)
        if key.lower() == "ensemble":
            plt.plot(recall, precision, color="red")
        else:
            plt.plot(recall, precision)
        # plt.plot(fpr,tpr,label=f"{key}, auc="+str(round(auc, 3)))
    plt.legend(loc='best')

    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.legend(loc=4)
    plt.show()

def draw_roc_multiple(models, X_test, y_test):
    for key in models:
        fpr, tpr, auc = draw_roc_curve(models[key], key, X_test, y_test, multiple=True)
        if key.lower() == "ensemble":
            plt.plot(fpr, tpr, color="red")
        else:
            plt.plot(fpr, tpr)
        # plt.plot(fpr,tpr,label=f"{key}, auc="+str(round(auc, 3)))
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
        
        fig, ax = plt.subplots(figsize=(10, 3.5))
        # width = 0.35
        # plt.yticks(rotation=30)
        plt.xlim(50, 101)
        plt.yticks(rotation=30)
        # plt.yticks([50, 60, 70, 80, 90, 100])
        assert type(obj).__name__=='OrderedDict'
        
        # sort dictionary by value
        # color=['black', 'red', 'green', 'blue', 'cyan']
        p1 = ax.barh([key for key in obj], [round(obj[key]*100, 3) for key in obj], edgecolor='black', color=['red' if key.lower() == "ensemble" else 'blue' for key in obj])
        # p2 = ax.bar(ind + width/2, [obj[key] for key in obj], width, label='Accuracy')
        vals = list(obj.values())
        for rect in range(len(p1)):
            # print(vals[rect])
            bar = p1[rect]
            width = bar.get_width()+2 #Previously we got the height
            label_y_pos = bar.get_y() + bar.get_height() / 2
            print(width, label_y_pos)
            ax.text(width, label_y_pos,
                    f'{round(vals[rect]*100, 3)}%',
                    ha='center', va='bottom', fontsize=10)

        ax.set_xlabel('Cross-Validated Accuracy')
        ax.set_title('Scores by model')
        ax.set_ylabel('Model Type')
        # plt.xlim(, 101)
        # plt.ylim(50, 101)
        # ax.set_xticks(ind, labels=["1", "2", "3", "4", "5"])
        plt.legend(loc=4)
        plt.show()


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

def cross_val(model, X, y, verb = 1, scoring = ['recall', 'f1', 'accuracy', 'precision', 'roc_auc', 'neg_brier_score'], cv=5):
    X = transform_data(model, X)
    # check if it is xgboost
    scoredi = {
        'recall': recall_score,
        'f1': f1_score,
        'accuracy': accuracy_score,
        'precision': precision_score,
        'roc_auc': roc_auc_score,
        'neg_brier_score': brier_score_loss
    }
    di = {}
    if type(model).__name__ == "XGBClassifier":
        # perform manual k fold validation with extra stopping
        kf = KFold(n_splits=cv)
        kf.get_n_splits(X)

        tprs = []
        base_fpr = np.linspace(0, 1, 101)
        
        avgauc = 0
        for train, test in kf.split(X):
            # divide train into train and validation
            X_train, X_val, y_train, y_val = train_test_split(X.iloc[train], y[train], test_size=0.25)
            model = model.fit(X_train, y_train, eval_metric='aucpr', eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=10)
            y_pred = model.predict_proba(X.iloc[test])[:, 1]
            for score in scoring:
                di[score] = di.get(score, []).append(scoredi[score](y.iloc[test], y_pred))
        
        # return mean of all scores
        for k, v in di.items():
            di[k] = round(np.mean(v), 3)


    print("cross validating", cv, "times")
    x = cross_validate(model, X, y, cv=cv, scoring=scoring)
    for k, v in x.items():
        print(k, round(v.mean(), 3))
        di[k]=round(v.mean(), 3)
    return di

def get_conf_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)
    

# def cross_validate_multiple_metrics(model, X, y):
#     X = transform_data(model, X)
#     di = {
#         'accuracy': (accuracy_score, {}),
#         'precision': (precision_score, {'average': 'macro'}),
#         'recall': (recall_score, {'average': 'macro'}),
#         'f1': (f1_score, {'average': 'macro'}),
#         'roc_auc': (roc_auc_score, {'average': 'macro'})
#     }

#     for s in scoring:
#         if s not in di:
#             print(f"Scoring metric {s} not supported")
#             return
#     # 
#     realdi = {
#         s: di[s] for s in scoring
#     }
#     print(realdi)
#     scorer = MultiScorer(realdi)


#     cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=3, random_state=1)
#     n_scores = cross_val_score(model, X, y, scoring=scorer, cv=cv, n_jobs=-1, error_score='raise', verbose=verb)
#     # print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
#     print(scorer)
#     # pickle.dump(scorer, open('scorer.pkl', 'wb'))
    
#     return n_scores, scorer
# datasets = retrieveAllDatasets(dir="../data")
# ds = datasets['merged']['normalized-4']
# merged_GBM = pickle.load(open('../models/curr_models/nardus_gridsearch.pkl', 'rb'))
# # # asdf = pickle.load(open('../models/curr_models/knn.pkl', 'rb'))

# em = pickle.load(open('../models/curr_models/final_merged_stackingcv.pkl', 'rb'))

# bestandworst = OrderedDict({
#     # 'merged_ensemble': em,
#     'ensemble': em,
#     'knn': merged_GBM.best_estimator_
    
# })

# # # # # scores_merged = [0.938, 0.943]
# # # draw_roc_multiple(bestandworst, ds['X_test'], ds['y_test'])

# # # # merged_dc = OrderedDict(sorted(OrderedDict(map(lambda i,j : (i,j) , bestandworst.keys(),scores_merged)).items(), key=lambda x: x[1], reverse=True))

# # # draw_avg_roc_multiple(bestandworst, ds['X_test'], ds['y_test'])
# # # # test()

# # draw_prec_roc_curve_multiple(bestandworst, ds['X_test'], ds['y_test'])
# scores_merged = [0.99, 0.98]
# merged_dc = OrderedDict(sorted(OrderedDict(map(lambda i,j : (i,j) , bestandworst.keys(),scores_merged)).items(), key=lambda x: x[1], reverse=True))
# # nardus_dc = OrderedDict(sorted(OrderedDict(map(lambda i,j : (i,j) , bestandworst.keys(),scores_nardus)).items(), key=lambda x: x[1], reverse=True))
# # zhang_dc = OrderedDict(sorted(OrderedDict(map(lambda i,j : (i,j) , bestandworst.keys(),scores_zhang)).items(), key=lambda x: x[1], reverse=True))
# print(merged_dc)
# draw_accuracies(bestandworst, None, None, merged_dc)
# draw_accuracies(bestandworst, ds['X_test'], ds['y_test'])