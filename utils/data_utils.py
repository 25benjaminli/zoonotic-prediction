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

def resetkmerdict(permset)->OrderedDict:
        kmerdict = OrderedDict()
        for i in permset:
            kmerdict[i]=0
        return kmerdict

def assign_kmers_to_dict(row, permset, kmer) -> OrderedDict:
    kmerdict=resetkmerdict(permset)
    st = row # tune for which column the sequence is in
    for j in range(len(st)-kmer+1):
        if not st[j:j+kmer] in permset: continue
        kmerdict[st[j:j+kmer]]+=1
    return kmerdict

def getTrainParams(mergedDf, kmer):
    print(mergedDf)
    s = product('acgt',repeat = kmer)
    permset = set(["".join(x) for x in list(s)])

    l = []
    
    for row in tqdm.tqdm(mergedDf.itertuples()):
        l.append(assign_kmers_to_dict(row, permset, kmer))

    finalkmerdict=pd.DataFrame(l)
    
    # shouldn't need to fill NAs
    # mergedDf.fillna(0, inplace=True)

    X = finalkmerdict
    Y = mergedDf['isZoonotic']
    X = X.apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=1)
    # print(X.head())

    global asdfX
    asdfX = X.copy()
    global asdfY
    asdfY = Y.copy()


    place = pd.concat([X, Y], axis=1)
    
    # print(place)

    place.to_csv('data/info.csv', index=False)

    return train_test_split(X, Y, test_size=0.2, random_state=1)

def transform_data(model, X_test):
    l = type(model)
    print(l.__name__)

    if l.__name__ == "GradientBoostingClassifier":
        print("gradboost")
        cols_when_model_builds = model.feature_names_in_
        X_test=X_test[cols_when_model_builds]
    elif l.__name__ == "XGBClassifier":
        print("xgboost")
        cols_when_model_builds = model.get_booster().feature_names
        X_test=X_test[cols_when_model_builds]

    return X_test
