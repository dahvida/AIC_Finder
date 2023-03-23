from typing import *
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
import rdkit

###############################################################################

def run_logger(
        mols: List[rdkit.Chem.rdchem.Mol],
        idx: List[int],
        y_f: np.ndarray,
        y_c: np.ndarray,
        flags: np.ndarray,
        flags_alt: np.ndarray,
        score: np.ndarray,
        algorithm: str
        ) -> pd.DataFrame:
    """Logs raw predictions for a given algorithm

    Args:
        mols:       (M,) mol objects from primary data
        idx:        (V,) positions of primary actives with confirmatory
                    readout
        y_f:        (V,) false positive labels (1=FP)
        y_c:        (V,) true positive labels (1=FP)
        flags:      (V,) FP predictions
        flags_alt:  (V,) TP predictions
        algorithm:  name of the algorithm for FP/TP detection
    
    Returns:
        Dataframe (V,5) containing SMILES, true labels and raw predictions
    """
    
    #get primary actives with confirmatory measurement
    mols_subset = [mols[x] for x in idx]

    #store results in db
    smiles = [Chem.MolToSmiles(x) for x in mols_subset]
    db = pd.DataFrame({
        "SMILES": smiles,
        "False positives": y_f,
        "True positives": y_c,
        "FP - " + algorithm: flags,
        "TP - " + algorithm: flags_alt,
        "Score - " + algorithm: score
        })

    return db

#-----------------------------------------------------------------------------#

def run_mvsa(
        mols: List[rdkit.Chem.rdchem.Mol],
        x: np.ndarray,
        y_p: np.ndarray,
        y_f: np.ndarray,
        y_c: np.ndarray,
        idx: List[int],
        replicates: int,
        log_predictions: bool = True
        ) -> Tuple[np.ndarray, pd.DataFrame]:
    """Executes MVS-A analysis on given dataset
    
    Uses MVS-A importance scores to rank primary screen actives in terms of 
    likelihood of being false positives or true positives. Top ranked compounds
    are FPs, as indicated in TracIn, while bottom ranked compounds are TPs.
    Finally, the function computes precision@90, EF10, BEDROC20
    and scaffold diversity metrics.

    Args:
        mols:               (M,) mol objects from primary data
        x:                  (M, 1024) ECFPs of primary screen molecules
        y_p:                (M,) primary screen labels
        y_f:                (V,) false positive labels (1=FP)        
        y_c:                (V,) true positive labels (1=FP)
        idx:                (V,) positions of primary actives with confirmatory    
                            readout  
        replicates:         number of replicates to use for the run
        log_predictions:    enables raw predictions logging
     
    Returns:
        Tuple containing one array (1,9) with precision@90 for FP and TP retrieval, 
        scaffold diversity and training time, and one dataframe (V,5) with
        SMILES, true labels and raw predictions
    """
    
    #create results containers
    temp = np.zeros((replicates,9))
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
        flags, flags_alt = process_ranking(y_p, vals)

        #get precision@90 for FP and TP retrieval
        temp[j,1] = precision_score(y_f, flags[idx])
        temp[j,2] = precision_score(y_c, flags_alt[idx])
        
        #get EF10 for FPs and TPs
        temp[j,3] = enrichment_factor_score(y_f, flags[idx])
        temp[j,4] = enrichment_factor_score(y_c, flags_alt[idx])

        #get BEDROC20 for FPs and TPs
        temp[j,5] = bedroc_score(y_f, vals[idx], reverse=True)
        temp[j,6] = bedroc_score(y_c, vals[idx], reverse=False)	 
        
        #get scaffold diversity for compounds that got flagged as FPs
        idx_fp = np.where(flags == 1)[0]
        mols_fp = [mols[x] for x in idx_fp]
        temp[j,7] = get_scaffold_rate(mols_fp)

        #get scaffold diversity for compounds that got flagged as TPs
        idx_tp = np.where(flags_alt == 1)[0]
        mols_tp = [mols[x] for x in idx_tp]
        temp[j,8] = get_scaffold_rate(mols_tp)
    
    #optionally fill up logger
    if log_predictions is True:
        logs = run_logger(mols, idx, y_f, y_c, flags[idx],
                                flags_alt[idx], vals[idx], "mvsa")

    return temp, logs
    
#-----------------------------------------------------------------------------#

def run_filter(
        mols: List[rdkit.Chem.rdchem.Mol],
        idx: List[int],
        filter_type: str,
        y_f: np.ndarray,
        y_c: np.ndarray,
        log_predictions: bool = True
        ) -> Tuple[np.ndarray, pd.DataFrame]:
    """Executes structural alert analysis on given dataset
    
    Uses structural alerts to mark primary hits as TPs or FPs. Then, 
    it computes precision@90, EF10, BEDROC20 and scaffold diversity indices. 
    
    Args:
        mols:               (M,) mol objects from primary data
        idx:                (V,) positions of primary actives with confirmatory    
                            readout  
        filter_type:        name of the structural alerts class to use                   
        y_f:                (V,) false positive labels (1=FP)        
        y_c:                (V,) true positive labels (1=FP)
        log_predictions:    enables raw predictions logging
     
    Returns:
        Tuple containing one array (1,9) with precision@90 for FP and TP retrieval, 
        scaffold diversity and training time, and one dataframe (V,5) with
        SMILES, true labels and raw predictions
    """

    #create results containers
    temp = np.zeros((1,9))
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
    
    #get EF10 for FPs and TPs
    temp[0,3] = enrichment_factor_score(y_f, flags)
    temp[0,4] = enrichment_factor_score(y_c, flags_alt)

    #get BEDROC20 for FPs and TPs
    temp[0,5] = bedroc_score(y_f, flags, reverse=True)
    temp[0,6] = bedroc_score(y_c, flags_alt, reverse=True)	

    #get scaffold diversity for compounds that got flagged as FPs
    idx_fp = np.where(flags == 1)[0]
    mols_fp = [mol_subset[x] for x in idx_fp]
    temp[0,7] = get_scaffold_rate(mols_fp)
    
    #get scaffold diversity for compounds that got flagged as TPs
    idx_tp = np.where(flags_alt == 1)[0]
    mols_tp = [mol_subset[x] for x in idx_tp]
    temp[0,8] = get_scaffold_rate(mols_tp)

    #optionally fill up logger
    if log_predictions is True:
        logs = run_logger(mols, idx, y_f, y_c, flags,
                                flags_alt, flags, "filter")
            
    return temp, logs  
    
#-----------------------------------------------------------------------------#

def run_catboost(
        mols: List[rdkit.Chem.rdchem.Mol],
        x: np.ndarray,
        y_p: np.ndarray,
        y_f: np.ndarray,
        y_c: np.ndarray,
        idx: List[int],
        replicates: int,
        log_predictions: bool = True
        ) -> Tuple[np.ndarray, pd.DataFrame]:
    """Executes CatBoost analysis on given dataset

    Uses CatBoost object importance function to rank primary screen actives
    in terms of likelihood of being false positives or true positives. Unlike
    for MVS-A, top ranked compounds are TPs, bottom ranked compounds FPs. Finally,
    the function computes precision@90, EF10, BEDROC20 and scaffold diversity metrics.
    
    Args:
        mols:               (M,) mol objects from primary data
        x:                  (M, 1024) ECFPs of primary screen molecules
        y_p:                (M,) primary screen labels
        y_f:                (V,) false positive labels (1=FP)        
        y_c:                (V,) true positive labels (1=FP)
        idx:                (V,) positions of primary actives with confirmatory    
                            readout  
        replicates:         number of replicates to use for the run
        log_predictions:    enables raw predictions logging
     
    Returns:
        Tuple containing one array (1,9) with precision@90 for FP and TP retrieval, 
        scaffold diversity and training time, and one dataframe (V,5) with
        SMILES, true labels and raw predictions
    """

    #create results containers
    temp = np.zeros((replicates,9))
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
        flags, flags_alt = process_ranking(y_p, vals)

        #get precision@90 for FP and TP retrieval
        temp[j,1] = precision_score(y_f, flags_alt[idx])
        temp[j,2] = precision_score(y_c, flags[idx])   
    
        #get EF10 for FPs and TPs
        temp[j,3] = enrichment_factor_score(y_f, flags_alt[idx])
        temp[j,4] = enrichment_factor_score(y_c, flags[idx])

        #get BEDROC20 for FPs and TPs
        temp[j,5] = bedroc_score(y_f, vals[idx], reverse=False)
        temp[j,6] = bedroc_score(y_c, vals[idx], reverse=True)	
        
        #get scaffold diversity for compounds that got flagged as FPs
        idx_fp = np.where(flags_alt == 1)[0]
        mols_fp = [mols[x] for x in idx_fp]
        temp[j,7] = get_scaffold_rate(mols_fp)
        
        #get scaffold diversity for compounds that got flagged as TPs
        idx_tp = np.where(flags == 1)[0]
        mols_tp = [mols[x] for x in idx_tp]
        temp[j,8] = get_scaffold_rate(mols_tp)

    #optionally store logs
    if log_predictions is True:
        logs = run_logger(mols, idx, y_f, y_c, flags_alt[idx],
                                flags[idx], vals[idx], "catboost")
    
    return temp, logs

#-----------------------------------------------------------------------------#

def run_score(
    df: pd.DataFrame,
    mols: List[rdkit.Chem.rdchem.Mol],
    idx: np.ndarray,
    y_p: np.ndarray,
    y_f: np.ndarray,
    y_c: np.ndarray,
    log_predictions: bool = True
    ) -> Tuple[np.ndarray, pd.DataFrame]:
    """Executes Score analysis on given dataset

    Uses raw primary screen scores (as indicated in PUBCHEM_ACTIVITY_SCORE) to 
    rank primary screen actives in terms of likelihood of being false positives 
    or true positives. Top ranked compounds (most active primaries) are TPs, 
    bottom ranked compounds FPs. Finally, the function computes precision@90, 
    EF10, BEDROC20 and scaffold diversity metrics.
    
    Args:
        mols:               (M,) mol objects from primary data
        x:                  (M, 1024) ECFPs of primary screen molecules
        y_p:                (M,) primary screen labels
        y_f:                (V,) false positive labels (1=FP)        
        y_c:                (V,) true positive labels (1=FP)
        idx:                (V,) positions of primary actives with confirmatory    
                            readout  
        replicates:         number of replicates to use for the run
        log_predictions:    enables raw predictions logging
     
    Returns:
        Tuple containing one array (1,9) with precision@90 for FP and TP retrieval, 
        scaffold diversity and training time, and one dataframe (V,5) with
        SMILES, true labels and raw predictions
    """ 
    #create results containers
    temp = np.zeros((1,9))
    logs = pd.DataFrame([])
    
    #get scores
    scores = np.array(df["Score"])
            
    #convert into labels (top 90% -> TP, bottom 10% -> FP)
    flags, flags_alt = process_ranking(y_p, scores)
        
    #get precision@90 for FPs and TPs
    temp[0,1] = precision_score(y_f, flags_alt[idx])
    temp[0,2] = precision_score(y_c, flags[idx])
    
    #get EF10 for FPs and TPs
    temp[0,3] = enrichment_factor_score(y_f, flags_alt[idx])
    temp[0,4] = enrichment_factor_score(y_c, flags[idx])

    #get BEDROC20 for FPs and TPs
    temp[0,5] = bedroc_score(y_f, scores[idx], reverse=False)
    temp[0,6] = bedroc_score(y_c, scores[idx], reverse=True)
    
    #get diversity for FPs
    idx_fp = np.where(flags_alt == 1)[0]
    mols_fp = [mols[x] for x in idx_fp]
    temp[0,7] = get_scaffold_rate(mols_fp)
        
    #get diversity for TPs
    idx_tp = np.where(flags == 1)[0]
    mols_tp = [mols[x] for x in idx_tp]
    temp[0,8] = get_scaffold_rate(mols_tp)

    #optionally fill up logger
    if log_predictions is True:
        logs = run_logger(mols, idx, y_f, y_c, flags_alt[idx],
                                    flags[idx], scores[idx], "score")
                
    return temp, logs  


