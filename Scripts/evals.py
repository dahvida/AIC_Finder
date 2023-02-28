from utils import *
import numpy as np
from sklearn.metrics import precision_score
from MVS_A import *
from fragment_filter import *
import time
from catboost import Pool
from catboost import CatBoostClassifier as clf
from rdkit import Chem
import pandas as pd

###############################################################################

def run_logger(
        mols,
        idx,
        y_f,
        y_c,
        flags,
        flags_alt,
        algorithm
        ):
    
    #get primary actives with confirmatory measurement
    mols_subset = [mols[x] for x in idx]

    #store results in db
    smiles = [Chem.MolToSmiles(x) for x in mols_subset]
    db = pd.DataFrame({
        "SMILES": smiles,
        "False positives": y_f,
        "True positives": y_c,
        "FP - " + algorithm: flags,
        "TP - " + algorithm: flags_alt
        })

    return db

#-----------------------------------------------------------------------------#

def run_mvsa(
        mols,
        x,
        y_p,
        y_f,
        y_c,
        idx,
        replicates,
        log_predictions = False
        ):
    
    #create results containers
    temp = np.zeros((replicates,4))
    logs = pd.DataFrame([])
    
    #loop analysis over replicates
    for j in range(replicates):
        
        #create MVS-A object (ONLY FEEDS PRIMARY DATA, CONFIRMATORY NEVER
        #OBSERVED BY THE MODEL DURING THIS ANALYSIS)
        obj = sample_analysis(x = x,
                              y = y_p, 
                              params = "default",
                              verbose = False,
                              seed = j)
        
        #get sample importances and measure time
        start = time.time()
        obj.get_model()
        vals = obj.get_importance()
        temp[j,0] = time.time() - start
        
        #convert importances into labels (top 90% -> FP, bottom 10% -> TP)
        flags, flags_alt = process_FP(y_p, vals)

        #get precision@90 for FP and TP retrieval
        temp[j,1] = precision_score(y_f, flags[idx])
        temp[j,2] = precision_score(y_c, flags_alt[idx])
        
        #get scaffold diversity for compounds that got flagged as FPs
        idx_fp = np.where(flags == 1)[0]
        mols_fp = [mols[x] for x in idx_fp]
        temp[j,3] = get_scaffold_rate(mols_fp)
    
    #optionally fill up logger
    if log_predictions is True:
        logs = run_logger(mols, idx, y_f, y_c, flags[idx],
                                flags_alt[idx], "mvsa")

    return temp, logs
    
#-----------------------------------------------------------------------------#

def run_filter(
        mols,
        idx,
        filter_type,
        y_f,
        y_c,
        log_predictions = False
        ):

    #create results containers
    temp = np.zeros((1,4))
    logs = pd.DataFrame([])   

    #get primary actives with confirmatory measurement
    mol_subset = [mols[x] for x in idx]
            
    #use substructure filters to flag FPs, time it and measure precision
    start = time.time()
    flags = filter_predict(mol_subset, filter_type)
    temp[0,0] = time.time() - start
    temp[0,1] = precision_score(y_f, flags)
    
    #invert filter flagging to find TPs and measure precision
    flags_alt = (flags - 1) * -1
    temp[0,2] = precision_score(y_c, flags_alt)
            
    #get scaffold diversity for compounds that got flagged as FPs
    idx_fp = np.where(flags == 1)[0]
    mols_fp = [mol_subset[x] for x in idx_fp]
    temp[0,3] = get_scaffold_rate(mols_fp)
    
    #optionally fill up logger
    if log_predictions is True:
        logs = run_logger(mols, idx, y_f, y_c, flags,
                                flags_alt, "filter")
            
    return temp, logs  
    
#-----------------------------------------------------------------------------#

def run_catboost(
        mols,
        x,
        y_p,
        y_f,
        y_c,
        idx,
        replicates,
        log_predictions = False
        ):
    
    #create results containers
    temp = np.zeros((replicates,4))
    logs = pd.DataFrame([])
    
    #loop analysis over replicates
    for j in range(replicates):
        
        #train catboost model, get sample importance and time it
        start = time.time()
        model = clf(iterations=100, random_seed=j)
        model.fit(x, y_p, verbose=False)
        idx_act = np.where(y_p==1)[0]
        pool1 = Pool(x, y_p)
        pool2 = Pool(x[idx_act], y_p[idx_act])
        idx_cat, scores = model.get_object_importance(pool2, pool1)
        temp[j,0] = time.time() - start
        
        #rearrange output into a single vector, then binarize importance
        #scores (bottom 10% -> FP, top 90% TP)
        vals = np.empty((y_p.shape[0]))
        vals[idx_cat] = scores
        flags, flags_alt = process_FP(y_p, vals)

        #get precision@90 for FP and TP retrieval
        temp[j,1] = precision_score(y_f, flags_alt[idx])
        temp[j,2] = precision_score(y_c, flags[idx])   
	
        #get scaffold diversity for compounds that got flagged as FPs
        idx_fp = np.where(flags == 1)[0]
        mols_fp = [mols[x] for x in idx_fp]
        temp[j,3] = get_scaffold_rate(mols_fp)
    
    #optionally store logs
    if log_predictions is True:
        logs = run_logger(mols, idx, y_f, y_c, flags_alt[idx],
                                flags[idx], "catboost")
    
    return temp, logs


























