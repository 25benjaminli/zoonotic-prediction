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

def retrieveAllDatasets(dir = "data", verb=False):
    datasets = OrderedDict({"zhang":{}, "nardus":{}, "merged":{}})

    # load datasets with different kmer values
    print("working directory: " + os.getcwd())

    for kmer in range(3, 7):
        df_1_reg = pd.read_csv(f'{dir}/dataset1/kmers-{str(kmer)}.csv')
        df_1_norm = pd.read_csv(f'{dir}/dataset1/normalized-{str(kmer)}.csv')
        df_2_reg = pd.read_csv(f'{dir}/dataset2/kmers-{str(kmer)}.csv')
        df_2_norm = pd.read_csv(f'{dir}/dataset2/normalized-{str(kmer)}.csv')

        df_reg_merge = pd.concat([df_1_reg, df_2_reg])
        df_reg_merge.reset_index(drop=True, inplace=True)
        df_reg_merge.drop_duplicates(inplace=True)

        df_norm_merge = pd.concat([df_1_norm, df_2_norm])
        df_norm_merge.reset_index(drop=True, inplace=True)
        df_norm_merge.drop_duplicates(inplace=True)

        l = [[df_1_reg, df_1_norm], [df_2_reg, df_2_norm], [df_reg_merge, df_norm_merge]]
        if verb:
            print("kmer: " + str(kmer))
            print('zhang reg', len(df_1_reg))
            print('zhang norm', len(df_1_norm))
            print('nardus reg', len(df_2_reg))
            print('nardus norm', len(df_2_norm))
            print('merge reg', len(df_reg_merge))
            print('merge norm', len(df_norm_merge))

        dstypes = ["zhang", "nardus", "merged"]
        
        for i, dataset in enumerate(l):
            reg_dataset, norm_dataset = dataset[0], dataset[1]
            
            X_train, X_test, y_train, y_test = train_test_split(reg_dataset.loc[:, reg_dataset.columns != 'isZoonotic'], reg_dataset['isZoonotic'], test_size=0.2, random_state=42)
            # split into validation
            # X_test, X_val, y_test, y_val = train_test_split(X_rem, y_rem, test_size=0.5, random_state=42)
            
            y_train = y_train.values.ravel()
            y_test = y_test.values.ravel()
            # y_val = y_val.values.ravel()

            datasets[dstypes[i]][f'regular-{kmer}'] = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

            X_train, X_test, y_train, y_test = train_test_split(norm_dataset.loc[:, norm_dataset.columns != 'isZoonotic'], norm_dataset['isZoonotic'], test_size=0.2, random_state=42)
            # split into validation
            # X_test, X_val, y_test, y_val = train_test_split(X_rem, y_rem, test_size=0.5, random_state=42)

            y_train = y_train.values.ravel()
            y_test = y_test.values.ravel()
            # y_val = y_val.values.ravel()

            datasets[dstypes[i]][f'normalized-{kmer}'] = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}


    return datasets