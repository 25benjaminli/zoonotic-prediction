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
import validation_utils

def saveModel(model, name, X_test, y_test, params=None, dir='models/curr_models', gradBoost=False, xgBoost=False):
    if not path.exists(f"{dir}/{name}.pkl"):
        print("does not exist")

        pickle.dump(model, open(f'{dir}/{name}.pkl', 'wb'))
    else:
        predictions = model.predict(X_test)
        currAcc = accuracy_score(y_test, predictions)

        pickled_model = pickle.load(open(f'{dir}/{name}.pkl', 'rb'))
        
        if gradBoost:
            # get features here 
            cols_when_model_builds = pickled_model.feature_names_in_
            X_test=X_test[cols_when_model_builds]
        elif xgBoost:
            # put features into the same order that the model was trained in
            cols_when_model_builds = pickled_model.get_booster().feature_names
            X_test=X_test[cols_when_model_builds]
        
        # .values?
        
        picklePredictions=pickled_model.predict(X_test)
        pickleAcc=accuracy_score(y_test, picklePredictions)
        
        if currAcc > pickleAcc:
            print("update!")

            # TP, FP, FN, TN
            print(confusion_matrix(y_test, picklePredictions).ravel())

            print("curr", currAcc, "pickle", pickleAcc)
            pickle.dump(model, open(f'{dir}/{name}.pkl', 'wb'))

            if params != None:
                pickle.dump(params, open(f'{dir}/{name}-params.pkl', 'wb'))
        else:
            print("no update")
            print("curr", currAcc, "pickle", pickleAcc)
            
            # TP, FP, FN, TN
            print(confusion_matrix(y_test, picklePredictions).ravel())

            model=pickled_model
    return model


def pred_res_proba(model, X_val):
    l = type(model)
    if l.__name__ == "GradientBoostingClassifier":
        # print("gradboost")
        cols_when_model_builds = model.feature_names_in_
        X_val=X_val[cols_when_model_builds]
    elif l.__name__ == "XGBClassifier":
        # print("xgboost")
        cols_when_model_builds = model.get_booster().feature_names
        X_val=X_val[cols_when_model_builds]
    return model.predict_proba(X_val)

def getOptimalModels(clf, datasets, kmer_range=[3, 6], onlyNormalized = False, onlyRegular = False):
    normalized_scores = []
    regular_scores = []
    # datasets is a dictionary containing names (e.g. zhang, nardus, merged) mapping to dataset values

    # include last index
    for kmer in range(kmer_range[0], kmer_range[1]+1):
        # FOUR different fits.
        bestInfo = {} # stores optimal model to save

        print("kmer: " + str(kmer))

        # assess on each dataset
        tempnorm = []
        tempreg = []

        # clf = svm.SVC(kernel='linear', probability=True)

        mergedRegular = datasets['merged'][f'regular-{kmer}']
        mergedNormalized = datasets['merged'][f'normalized-{kmer}']

        # for each dataset, construct a normalized & non-normalized version of the model, and pick the optimal one
        for index, name in enumerate(datasets):
            dataset = datasets[name]
            # normalized
            print("dataset: " + name)

            if not onlyRegular:
                curr_dataset = dataset[f'normalized-{kmer}']

                clf.fit(curr_dataset['X_train'], curr_dataset['y_train'])
                
                print("finished training, beginning cross validation")
                
                temp = validation_utils.cross_validate(clf, mergedNormalized['X_test'], mergedNormalized['y_test'])

                if temp > bestInfo.get('acc', 0):
                    # print("norm zhang best")
                    bestInfo['dataset_name'] = name
                    bestInfo['dataset_type'] = 'normalized'
                    bestInfo['kmer'] = kmer
                    bestInfo['model'] = clf
                    bestInfo['acc'] = temp

                print(f"normalized {name} kmer: {kmer}", temp)

                tempnorm.append(temp)

            # regular
            # clf = MLPClassifier(hidden_layer_sizes=(256,128,64,32, 16),activation="relu",random_state=1)
            if not onlyNormalized:
                curr_dataset = dataset[f'regular-{kmer}']

                clf.fit(curr_dataset['X_train'], curr_dataset['y_train'])

                print("finished training, beginning cross validation")

                temp = validation_utils.cross_validate(clf, mergedRegular['X_test'], mergedRegular['y_test'])

                if temp > bestInfo.get('acc', 0):
                    # print("norm zhang best")
                    bestInfo['dataset_name'] = name
                    bestInfo['dataset_type'] = 'regular'
                    bestInfo['kmer'] = kmer
                    bestInfo['model'] = clf
                    bestInfo['acc'] = temp
                
                print(f"regular {name} kmer: {kmer}", temp)

                tempreg.append(temp)

                pickle.dump(bestInfo, open(f"models/searched/-{name}-{kmer}-{bestInfo['dataset_type']}.pkl", "wb"))

        # for each dataset, then for each type of training - normalized, regular, store the cross validated scores
        
        normalized_scores.append(tempnorm)
        regular_scores.append(tempreg)
    return normalized_scores, regular_scores