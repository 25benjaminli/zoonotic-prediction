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
import data_utils

def queryKmer(ID, isZoonotic_list, index, everything):
    FileName = "{}.gb".format(ID)
    try:
        QueryHandle = Entrez.efetch(db="nucleotide", id=ID, 
                                    rettype="gb", retmode="text")
    except HTTPError as Error:
        if Error.code == 400:  # Bad request
            raise ValueError(f"Accession number not found: {ID}")
        else:
            raise

    SeqRec = SeqIO.read(QueryHandle, "genbank")
    info = {'accession': ID, 'sequence': str(SeqRec.seq).lower(), 'isZoonotic': isZoonotic_list[index]}
    everything.append(info)

    pickle.dump(info, open(f"sequences/{ID}.pkl", "wb"))



def getSingleSequence(accessionID) -> pd.DataFrame:
    try:
        QueryHandle = Entrez.efetch(db="nucleotide", id=accessionID, 
                                    rettype="gb", retmode="text")
    except HTTPError as Error:
        if Error.code == 400:  # Bad request
            raise ValueError(f"Accession number not found: {accessionID}")
        else:
            raise

    SeqRec = SeqIO.read(QueryHandle, "genbank")
    X_info = SeqRec.seq.lower()
    kmer = 4
    s = product('acgt',repeat = kmer)
    permset = set(["".join(x) for x in list(s)])


    oDict = data_utils.assign_kmers_to_dict(X_info, permset, kmer) # convert ordereddict to pandas dataframe

    kmer_df = pd.DataFrame()

    for i in oDict:
        kmer_df.at[0, i]=oDict[i]
    # print(best_gradBoost.predict_proba(kmer_df))
    kmer_df = kmer_df.apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=1)

    return kmer_df
    # print("gridsearch result")
    # nardus_gradBoost = pickle.load(open(f'models/nardus_gridsearch.pkl', 'rb'))
    # print(model_utils.pred_res_proba(nardus_gradBoost.best_estimator_, kmer_df))

    # print("regular gradboost")

    # othergrad = pickle.load(open('models/curr_models/gradBoost.pkl', 'rb'))
    # print(model_utils.pred_res_proba(othergrad, kmer_df))

    # print("xgboost")
    # othergrad = pickle.load(open('models/curr_models/xgBoost.pkl', 'rb'))
    # print(model_utils.pred_res_proba(othergrad, kmer_df))

def getFromSeq(X_info) -> pd.DataFrame:
    kmer = 4
    s = product('acgt',repeat = kmer)
    permset = set(["".join(x) for x in list(s)])
    X_info = X_info.lower()

    oDict = data_utils.assign_kmers_to_dict(X_info, permset, kmer) # convert ordereddict to pandas dataframe
    
    kmer_df = pd.DataFrame()

    for i in oDict:
        kmer_df.at[0, i]=oDict[i]
    # print(best_gradBoost.predict_proba(kmer_df))
    kmer_df = kmer_df.apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=1)

    return kmer_df