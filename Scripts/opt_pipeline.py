"""Fluorescence predictor model development pipeline.

Pipeline to create a set of fluorescence predictors to identify HTS false positives
according to InterPred (https://pubmed.ncbi.nlm.nih.gov/32421835/).
Some details could not be reproduced due to the usage in the WebApp of Python 2
packages, missing data (many compounds had CAS identifies that could not be parsed
by the Chemical Identifier Resolver), missing information on the model development process
(e.g. target metric for optimization, parameter ranges, splits). 

Steps:
    1. Load assays
    2. Compute ~1600 molecular descriptors via Mordred
    3. Split train/test 90/10
    4. Random undersample inactives to enforce 70/30 ratio inactives/actives
    5. Optimize hyperparameters via grid search on 10-fold CV on the train set
    6. Create a bagged ensemble (10 models) on the non-sampled train set and
        test it on the test set
    5. Repeat steps 3-6 for all 13 endpoints
    6. Save ensembles
"""

import pandas as pd
from utils import *
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from joblib import dump
import csv
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import pickle as pkl
from rdkit import Chem

###############################################################################

def main():
 
    print("[opt]: Beginning autofluorescence optimization...")
    db = pd.read_csv("../Misc/autofluorescence.csv", index_col=0)
    smiles = list(db.index)
    mols = [Chem.MolFromSmiles(x) for x in smiles]
    x = get_mordred(mols)
    y = np.array(db)
    metrics = np.zeros((13, 6))
        
    for i in range(13):
        y_temp = y[:,i]
        x_train, x_test, y_train, y_test = train_test_split(x, y_temp,
                                                                test_size=0.1,
                                                                stratify=y_temp,
                                                                random_state=42)
            
        param_grid={
                'n_estimators': [50, 100, 250, 500],
                'max_features': ["sqrt", "log2", None],
                }
        kfolds = StratifiedKFold(10)
            
        rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
        x_train_r, y_train_r = rus.fit_resample(x_train, y_train)
            
        grid = GridSearchCV(RandomForestClassifier(random_state=42),
                           param_grid = param_grid,
                           cv = kfolds,
                           scoring = "average_precision",
                           n_jobs = -1
                           )
        grid.fit(x_train_r, y_train_r)
        optimum = grid.best_params_
        
        model = BalancedBaggingClassifier(RandomForestClassifier(n_jobs=-1,
                                                                     random_state=42,
                                                                     **optimum),
                                              sampling_strategy=0.5,
                                              replacement=False,
                                              n_estimators=10)
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        probs = model.predict_proba(x_test)[:,1]
        
        metrics[i, 0] = matthews_corrcoef(y_test, preds)
        metrics[i, 1] = roc_auc_score(y_test, probs)
        metrics[i, 2] = average_precision_score(y_test, probs)
        metrics[i, 3] = balanced_accuracy_score(y_test, preds)
        metrics[i, 4] = recall_score(y_test, preds)
        metrics[i, 5] = recall_score(y_test, preds, pos_label=0)
        
        model = BalancedBaggingClassifier(RandomForestClassifier(n_jobs=-1,
             								random_state=42,
             								**optimum),
                                              sampling_strategy=0.5,
                                              n_estimators=10)
        model.fit(x, y_temp)
        dump(model, f"../Misc/model_{i}.joblib")
        
    metrics = np.mean(metrics, axis=0)
    performance = {
            "MCC": metrics[0],
            "ROC-AUC": metrics[1],
            "PR-AUC": metrics[2],
            "B-Acc": metrics[3],
            "Sensitivity": metrics[4],
            "Specificity": metrics[5],
            }    
        
    with open('../Misc/performance.csv','w') as f:
            w = csv.writer(f)
            w.writerows(performance.items())
    print("[opt]: Autofluorescence model trained and saved")


if __name__ == "__main__":
    main()

