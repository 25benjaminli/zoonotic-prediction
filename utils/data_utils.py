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

def getTrainParams(mergedDf, kmer, f):
    print(mergedDf)
    s = product('acgt',repeat = kmer)
    permset = set(["".join(x) for x in list(s)])

    l = []
    
    for row in tqdm.tqdm(mergedDf.itertuples()):
        l.append(assign_kmers_to_dict(row, permset, kmer))

    finalkmerdict=pd.DataFrame(l)
    

    X = finalkmerdict
    Y = mergedDf['isZoonotic']

    vec = pd.concat([X, Y], axis=1)
    vec.to_csv(f'data/{f}/kmers-{str(kmer)}.csv', index=False)

    # only apply normalization but could try other feature defs as well    
    X = X.apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=1)

    place = pd.concat([X, Y], axis=1)
    
    place.to_csv(f'data/{f}/normalized-{str(kmer)}.csv', index=False)

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

def retrieveAllDatasets():
    dataset1 = OrderedDict({})

    dataset2 = OrderedDict({})

    mergedDataset = OrderedDict({})

    # load datasets with different kmer values
    print("working directory: " + os.getcwd())
    for kmer in range(3, 7):
        df_1_reg = pd.read_csv(f'data/dataset1/kmers-{str(kmer)}.csv')
        df_1_norm = pd.read_csv(f'data/dataset1/normalized-{str(kmer)}.csv')
        df_2_reg = pd.read_csv(f'data/dataset2/kmers-{str(kmer)}.csv')
        df_2_norm = pd.read_csv(f'data/dataset2/normalized-{str(kmer)}.csv')

        df_reg_merge = pd.concat([df_1_reg, df_2_reg])
        df_reg_merge.reset_index(drop=True, inplace=True)

        df_norm_merge = pd.concat([df_1_norm, df_2_norm])
        df_norm_merge.reset_index(drop=True, inplace=True)

        print("kmer: " + str(kmer))

        X_train, X_test, y_train, y_test = train_test_split(df_1_reg.loc[:, df_1_reg.columns != 'isZoonotic'], df_1_reg['isZoonotic'], test_size=0.2, random_state=1)
        # for col in df.columns:
        #     col != 'isZoonotic' and X_train[col].isnull().sum() != 0 and print(X_train[col].isnull().sum())
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()

        dataset1[f'regular-{kmer}'] = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

        X_train, X_test, y_train, y_test = train_test_split(df_1_norm.loc[:, df_1_norm.columns != 'isZoonotic'], df_1_norm['isZoonotic'], test_size=0.2, random_state=1)
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()

        dataset1[f'normalized-{kmer}'] = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

        X_train, X_test, y_train, y_test = train_test_split(df_2_reg.loc[:, df_2_reg.columns != 'isZoonotic'], df_2_reg['isZoonotic'], test_size=0.2, random_state=1)
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()

        dataset2[f'regular-{kmer}'] = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

        X_train, X_test, y_train, y_test = train_test_split(df_2_norm.loc[:, df_2_norm.columns != 'isZoonotic'], df_2_norm['isZoonotic'], test_size=0.2, random_state=1)
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()

        dataset2[f'normalized-{kmer}'] = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}


        X_train, X_test, y_train, y_test = train_test_split(df_reg_merge.loc[:, df_reg_merge.columns != 'isZoonotic'], df_reg_merge['isZoonotic'], test_size=0.2, random_state=1)
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()

        mergedDataset[f'regular-{kmer}'] = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

        X_train, X_test, y_train, y_test = train_test_split(df_norm_merge.loc[:, df_norm_merge.columns != 'isZoonotic'], df_norm_merge['isZoonotic'], test_size=0.2, random_state=1)
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()
        
        mergedDataset[f'normalized-{kmer}'] = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

    datasets = {"zhang": dataset1, "nardus": dataset2, "merged": mergedDataset}

    return datasets
