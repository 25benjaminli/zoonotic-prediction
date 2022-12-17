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