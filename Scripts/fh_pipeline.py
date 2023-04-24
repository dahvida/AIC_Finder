"""Frequent Hitters overlap script.

Pipeline to evaluate how many false positives found by MVS-A are also predicted to be
frequent hitters by a LightGBM classifier with ECFP featurization.

Before running the script, make sure that you have already ran eval_pipeline.py with
logging enabled, since those .csv files will be used for this analysis. Also, make sure
to unzip "frequent_hitters.zip" in ../Misc. 
If no flags are passed, the analysis will use the default parameters used in our study. 
The output will be saved as ../Results/fh_overlap.csv

Steps:
    1. Load preprocessed frequent hitter dataset from ../Misc
    2. Calculate ECFPs for training set
    3. Train LightGBM model
    4. For each logged dataset:
        4.1. Load and featurize FPs predicted by MVS-A
        4.2. Calculate predictions with LightGBM
        4.3. Calculate overlap by dividing number of predicted frequent hitters versus
             total MVS-A FPs
    5. Save
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from utils import *
import pickle as pkl
from sklearn.metrics import *
from lightgbm import LGBMClassifier
from hyperopt import tpe, hp, fmin, Trials
import argparse
import os

###############################################################################

parser = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--n_estimators', default=500, type=int,
                    help="LightGBM optimization iterations")

parser.add_argument('--lr', default=0.03, type=float,
                    help="LightGBM learning rate")

args = parser.parse_args()

##############################################################################

def main(
        n_estimators,
        lr,
        ):

    print("[fh]: Starting analysis...")
    
    #load and featurize FH dataset
    db = pd.read_csv("../Misc/frequent_hitters.csv")
    y = np.array(db["y"])
    ecfp = list(db["SMILES"])
    ecfp = [Chem.MolFromSmiles(x) for x in ecfp]
    ecfp = get_ECFP(ecfp)
    
    print("[fh]: Data loaded and featurized")
        
    #create LightGBM model according to flags
    clf = LGBMClassifier(n_estimators=n_estimators,
                         learning_rate=lr,
                         class_weight="balanced",
                         max_bin = 15,
                         verbose = -10)
    clf.fit(ecfp, y)
    
    print("[agg]: Model trained")

    #get all logged dataset names
    names = os.listdir("../Logs/eval/")
    overlaps = np.zeros((len(names), 1))
    
    #loop over all logged datasets
    for i in range(len(names)):
        
        print(f"Processing dataset: {names[i]}")

        #load csv and select all compounds flagged by MVS-A as FP 
        temp = pd.read_csv("../Logs/eval/" + names[i])
        temp = temp.loc[temp["FP - mvsa"] == 1]
        smiles = list(temp["SMILES"])
        mols = [Chem.MolFromSmiles(x) for x in smiles]

        #compute and preprocess descriptors for new csv
        val = get_ECFP(mols)

        #get predictions and select highest fluorescence likelihood
        #for each compound
        preds = clf.predict(val)
        
        #compute overlap
        overlaps[i, 0] = np.sum(preds) * 100 / preds.shape[0]
        print(f"Overlap: {overlaps[i]}")
        print("--------------------------")
    
    #store in pandas dataframe and save
    output = pd.DataFrame(
            data = overlaps,
            index = names,
            columns = ["Overlap %"]
            )
    output.to_csv("../Results/fh_overlap.csv")
    print("[agg]: Analysis finished, file saved at ../Results/fh_overlap.csv")

    

if __name__ == "__main__":
    main(
        args.n_estimators,
        args.lr
        )



















