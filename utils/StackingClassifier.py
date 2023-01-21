from itertools import permutations, product

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split, cross_val_score, cross_validate, KFold
from sklearn.metrics import accuracy_score, auc, confusion_matrix, balanced_accuracy_score, precision_recall_curve, auc, roc_curve, roc_auc_score, f1_score, recall_score, precision_score, brier_score_loss, average_precision_score, classification_report
from sklearn.inspection import permutation_importance
from sklearn import preprocessing

from sklearn.neural_network import MLPClassifier
from collections import Counter
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier

import numpy as np
from numpy import mean,std
from sklearn.model_selection import GridSearchCV

import pickle

from ctgan import CTGANSynthesizer
from mlxtend.classifier import StackingCVClassifier

from os import path
import tqdm
import matplotlib.pyplot as plt

from warnings import simplefilter
from collections import OrderedDict
from sklearn.svm import SVC

class StackingClassifier():
    def __init__(self, classifiers, meta_classifier, n_folds=5, use_probas=True):
        self.classifiers = classifiers # assume pretrained
        self.meta_classifier = meta_classifier # logistic regression
        self.n_folds = n_folds
        self.X_train_new=None
        self.X_test_new=None
        self.y_train_new=None
        self.use_probas = use_probas

    def fit_pretrained(self, X_train, y_train): # 
        self.X_train_new = np.zeros((X_train.shape[0], len(self.classifiers)))
        self.y_train_new = y_train
        print(X_train.shape[0], len(y_train))

        for i, clf in enumerate(self.classifiers):
            if self.use_probas:
                self.X_train_new[:, i] = model.predict_proba(X_train)[:,1]
            else:
                self.X_train_new[:, i] = model.predict(X_train)

        print(len(self.X_train_new))
        
        self.meta_classifier = self.meta_classifier.fit(self.X_train_new, self.y_train_new)

    def fit_not_pretrained(self, X_train, y_train): # assume NOT pretrained
        print(X_train.shape[0], len(y_train))
        for index, clf in enumerate(self.classifiers):
            print(type(clf).__name__)
            if type(clf).__name__ != 'XGBClassifier':
                self.classifiers[index] = clf.fit(X_train, y_train)
            else:
                print("xgboost detected")
                X_train_temp, X_validation, y_train_temp, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
                self.classifiers[index] = clf.fit(X_train_temp, y_train_temp, eval_metric='aucpr', eval_set=[(X_validation, y_validation)], early_stopping_rounds=15, verbose=10)
        
        self.fit_pretrained(X_train, y_train)

    def predict(self, X):
        meta_features = np.column_stack([
            clf.predict(X) for clf in self.classifiers
        ])
        return self.meta_classifier.predict(meta_features)

    def predict_proba(self, X):
        meta_features = np.column_stack([
            clf.predict_proba(X)[:,1] for clf in self.classifiers
        ])
        return self.meta_classifier.predict_proba(meta_features)

    def cross_validate(self, X, y, scoring=['precision', 'recall', 'f1', 'average_precision'], cv=5):
        kfold = KFold(n_splits=cv)
        scores = {s: [] for s in scoring}
        metrics = {
            'recall': recall_score,
            'f1': f1_score,
            'accuracy': accuracy_score,
            'precision': precision_score,
            'roc_auc': roc_auc_score,
            'neg_brier_score': brier_score_loss,
            'average_precision': average_precision_score
        }
        X = X.to_numpy()
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
            self.fit_not_pretrained(X_train, y_train)
            if self.use_probas:
                y_pred = self.predict_proba(X_test)[:,1]

                print(y_pred.sum())

                for s in scoring:
                    # if s == 'accuracy' or s == 'precision' or s == 'recall':
                    y_pred = np.where(y_pred > 0.5, 1, 0)
                    scores[s].append(metrics[s](y_test, y_pred))
            else:
                print("not use probas")
                y_pred = self.predict(X_test)

                print(y_pred.sum())

                for s in scoring:
                    met = metrics[s](y_test, y_pred)
                    print(s, met)
                    scores[s].append(met)

        return scores
    