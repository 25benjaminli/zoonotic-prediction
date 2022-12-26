import argparse
import os
import pickle
import sys
import time
import pandas as pd
from itertools import permutations, product

sys.path.append('..')
from utils import data_utils
from warnings import simplefilter

# ignore performance warning
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
simplefilter(action='ignore', category=FutureWarning)

def getFromSeq(model, X_info) -> pd.DataFrame:
    kmer = 4
    s = product('acgt',repeat = kmer)
    permset = set(["".join(x) for x in list(s)])
    X_info = X_info.lower()

    oDict = data_utils.assign_kmers_to_dict(X_info, permset, kmer) # convert ordereddict to pandas dataframe
    
    kmer_df = pd.DataFrame()

    for i in oDict:
        kmer_df.at[0, i]=oDict[i]

    # print(best_gradBoost.predict_proba(kmer_df))
    kmer_df = kmer_df.apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=1) # normalizing
    
    kmer_df = data_utils.transform_data(model, kmer_df)

    return kmer_df

parser = argparse.ArgumentParser(
                    prog = 'run models',
                    description = 'select a model')

parser.add_argument('--model', type=str, help='path to model')
parser.add_argument('--data', type=str, help='txt file with sequence to predict - formatted in such a way that ID & Sequence alternate')
parser.add_argument('--output', type=str, help='output file name')

args = parser.parse_args()

# print(args)

model = '../'+args.model
data = args.data
output = args.output

model = pickle.load(open(model, 'rb'))
ct = 0
seenId = ""
for line in open(data, 'r').readlines():
    print("hi")
    if ct % 2 != 0:
        print(type(line))
        realmodel = model
        # if it is a gridsearch model
        try:
            realmodel = model.best_estimator_
        except:
            pass
        
        df = getFromSeq(realmodel, line)
        print(df)
        res = realmodel.predict_proba(df)[:,1]
        # print(res)
        with open(output, 'w') as f:
            s = f"zoonotic potential of {seenId}: " + str(round(res[0]*100, 3)) + "%"
            f.write(s)
            f.close()
    else:
        seenId = line
        ct+=1