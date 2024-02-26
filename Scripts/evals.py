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
from sklearn.ensemble import IsolationForest
from vae import *

###############################################################################

def run_logger(
        mols,
        y_f,
        y_c,
        fp_box,
        tp_box,
        replicates) -> pd.DataFrame:
    """Logs raw predictions for a given algorithm

    Args:
        mols:       (M,) mol objects from primary data
        y_f:        (V,) false positive labels (1=FP)
        y_c:        (V,) true positive labels (1=FP)
        fp_box:     (V,) FP predictions across all replicates
        tp_box:     (V,) TP predictions across all replicates
        replicates: number of replicates
    
    Returns:
        Dataframe (V, 3 + 2*replicates) containing SMILES, true labels and
        raw predictions
    """

    #get primary actives with confirmatory measurement
    smiles = [Chem.MolToSmiles(x) for x in mols]

    fp_labs = [str(x) + "FP" for x in range(replicates)]
    tp_labs = [str(x) + "TP" for x in range(replicates)]

    fp_arr = y_f.reshape(-1,1)
    tp_arr = y_c.reshape(-1,1)
    for i in range(replicates):
        fp_arr = np.concatenate((fp_arr, fp_box[i].reshape(-1,1)), axis=1)
        tp_arr = np.concatenate((tp_arr, tp_box[i].reshape(-1,1)), axis=1)
    
    fp_db = pd.DataFrame(
            data = fp_arr,
            index = smiles,
            columns = ["FP"] + fp_labs
            )
    tp_db = pd.DataFrame(
            data = tp_arr,
            index = smiles,
            columns = ["TP"] + tp_labs
            )
    
    final = pd.concat([fp_db, tp_db], axis=1)

    return final

#-----------------------------------------------------------------------------#

def run_mvsa(
        mols: List[rdkit.Chem.rdchem.Mol],
        x: np.ndarray,
        y_p: np.ndarray,
        y_f: np.ndarray,
        y_c: np.ndarray,
        idx: List[int],
        replicates: int,
        ) -> Tuple[np.ndarray, pd.DataFrame]:
    """Executes MVS-A analysis on given dataset
    
    Uses MVS-A importance scores to rank primary screen actives in terms of 
    likelihood of being false positives or true positives. Top ranked compounds
    are FPs, as indicated in TracIn, while bottom ranked compounds are TPs.
    Finally, the function computes precision@90, EF10, BEDROC20, time
    and scaffold diversity metrics.

    Args:
        mols:               (M,) mol objects from primary data
        x:                  (M, K) fingerprints/descriptors of primary screen
        y_p:                (M,) primary screen labels
        y_f:                (V,) false positive labels (1=FP)        
        y_c:                (V,) true positive labels (1=FP)
        idx:                (V,) positions of primary actives with confirmatory    
                            readout  
        replicates:         number of replicates to use for the run
     
    Returns:
        Tuple containing one array (1,9) with metrics for FP and TP retrieval, 
        scaffold diversity and training time, and one dataframe (V,5) with
        SMILES, true labels and raw predictions
    """
    
    #create results containers
    temp = np.zeros((replicates,9))
    fp_box = []
    tp_box = []
    
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
        
        #append raw predictions
        fp_box.append(flags[idx])
        tp_box.append(flags_alt[idx])

    #create logger dataframe
    mols_subset = [mols[x] for x in idx]
    logs = run_logger(mols_subset, y_f, y_c,
                          fp_box, tp_box, replicates)

    return temp, logs
    
#-----------------------------------------------------------------------------#

def run_filter(
        mols: List[rdkit.Chem.rdchem.Mol],
        idx: List[int],
        y_p: np.ndarray,
        y_f: np.ndarray,
        y_c: np.ndarray,
        ) -> Tuple[np.ndarray, pd.DataFrame]:
    """Executes structural alert analysis on given dataset
    
    Uses structural alerts to mark primary hits as TPs or FPs. Then, 
    it computes precision@90, EF10, BEDROC20, time and scaffold diversity indices. 
    
    Args:
        mols:               (M,) mol objects from primary data
        idx:                (V,) positions of primary actives with confirmatory    
                            readout  
        y_p:                (M,) primary screen labels    
        y_f:                (V,) false positive labels (1=FP)        
        y_c:                (V,) true positive labels (1=FP)
     
    Returns:
        Tuple containing one array (1,9) with metrics for FP and TP retrieval, 
        scaffold diversity and training time, and one dataframe (V,5) with
        SMILES, true labels and raw predictions
    """

    #create results containers
    temp = np.zeros((1,9))

    #get primary actives with confirmatory measurement
    mol_subset = [mols[x] for x in idx]
            
    #use substructure filters to flag FPs and time it 
    start = time.time()
    preds = filter_predict(mol_subset)
    temp[0,0] = time.time() - start
    
    #store preds and convert to percentiles
    preds_dummy = np.zeros((len(mols)))
    preds_dummy[idx] = preds
    flags, flags_alt = process_ranking(y_p, preds_dummy)
        
    #measure precision
    temp[0,1] = precision_score(y_f, flags[idx])
    temp[0,2] = precision_score(y_c, flags_alt[idx])
    
    #get EF10 for FPs and TPs
    temp[0,3] = enrichment_factor_score(y_f, flags[idx])
    temp[0,4] = enrichment_factor_score(y_c, flags_alt[idx])

    #get BEDROC20 for FPs and TPs
    temp[0,5] = bedroc_score(y_f, preds, reverse=True)
    temp[0,6] = bedroc_score(y_c, preds, reverse=False)	

    #get scaffold diversity for compounds that got flagged as FPs
    idx_fp = np.where(flags == 1)[0]
    mols_fp = [mols[x] for x in idx_fp]
    temp[0,7] = get_scaffold_rate(mols_fp)
    
    #get scaffold diversity for compounds that got flagged as TPs
    idx_tp = np.where(flags_alt == 1)[0]
    mols_tp = [mols[x] for x in idx_tp]
    temp[0,8] = get_scaffold_rate(mols_tp)

    #append raw predictions
    fp_box = [flags[idx]]
    tp_box = [flags_alt[idx]]

    #create logger dataframe
    logs = run_logger(mol_subset, y_f, y_c,
                          fp_box, tp_box, 1)
            
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
        ) -> Tuple[np.ndarray, pd.DataFrame]:
    """Executes CatBoost analysis on given dataset

    Uses CatBoost object importance function to rank primary screen actives
    in terms of likelihood of being false positives or true positives. Unlike
    for MVS-A, top ranked compounds are TPs, bottom ranked compounds FPs. Finally,
    the function computes precision@90, EF10, BEDROC20, time and scaffold 
    diversity metrics.
    
    Args:
        mols:               (M,) mol objects from primary data
        x:                  (M, K) fingerprints/descriptors of primary screen
        y_p:                (M,) primary screen labels
        y_f:                (V,) false positive labels (1=FP)        
        y_c:                (V,) true positive labels (1=FP)
        idx:                (V,) positions of primary actives with confirmatory    
                            readout  
        replicates:         number of replicates to use for the run
     
    Returns:
        Tuple containing one array (1,9) with metrics for FP and TP retrieval, 
        scaffold diversity and training time, and one dataframe (V,5) with
        SMILES, true labels and raw predictions
    """

    #create results containers
    temp = np.zeros((replicates,9))
    fp_box = []
    tp_box = []

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
        
        #getscaffold diversity for compounds that got flagged as TPs
        idx_tp = np.where(flags == 1)[0]
        mols_tp = [mols[x] for x in idx_tp]
        temp[j,8] = get_scaffold_rate(mols_tp)

        #append raw predictions
        fp_box.append(flags_alt[idx])
        tp_box.append(flags[idx])

    #create logger dataframe
    mols_subset = [mols[x] for x in idx]
    logs = run_logger(mols_subset, y_f, y_c,
                          fp_box, tp_box, replicates)
    
    return temp, logs

#-----------------------------------------------------------------------------#

def run_score(
    df: pd.DataFrame,
    mols: List[rdkit.Chem.rdchem.Mol],
    idx: np.ndarray,
    y_p: np.ndarray,
    y_f: np.ndarray,
    y_c: np.ndarray,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
    """Executes Score analysis on given dataset

    Uses raw primary screen scores (as indicated in PUBCHEM_ACTIVITY_SCORE) to 
    rank primary screen actives in terms of likelihood of being false positives 
    or true positives. Top ranked compounds (most active primaries) are TPs, 
    bottom ranked compounds FPs. Finally, the function computes precision@90, 
    EF10, BEDROC20 and scaffold diversity metrics.
    
    Args:
        df:                 (M,4) dataframe of the HTS to analyse
        mols:               (M,) mol objects from primary data
        idx:                (V,) positions of primary actives with confirmatory    
                            readout  
        y_p:                (M,) primary screen labels
        y_f:                (V,) false positive labels (1=FP)        
        y_c:                (V,) true positive labels (1=FP)
     
    Returns:
        Tuple containing one array (1,9) with metrics for FP and TP retrieval and 
        scaffold diversity, and one dataframe (V,5) with SMILES, true labels 
        and raw predictions
    """ 
    #create results containers
    temp = np.zeros((1,9))
    
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

    #append raw predictions
    fp_box = [flags_alt[idx]]
    tp_box = [flags[idx]]

    #optionally create logger dataframe
    mols_subset = [mols[x] for x in idx]
    logs = run_logger(mols_subset, y_f, y_c,
                          fp_box, tp_box, 1)
                
    return temp, logs  

#-----------------------------------------------------------------------------#

def run_isoforest(
        mols: List[rdkit.Chem.rdchem.Mol],
        x: np.ndarray,
        y_p: np.ndarray,
        y_f: np.ndarray,
        y_c: np.ndarray,
        idx: List[int],
        replicates: int,
        ) -> Tuple[np.ndarray, pd.DataFrame]:
    """Executes Isolation Forest analysis on given dataset

    Uses Isolation Forest anomaly detection to rank primary screen actives
    in terms of likelihood of being false positives or true positives. According
    to scikit-learn's documentation, low scores correspond to anomalies, so the
    ranking is inverted.
    Finally, the function computes precision@90, EF10, BEDROC20, time and scaffold
    diversity metrics.
    
    Args:
        mols:               (M,) mol objects from primary data
        x:                  (M, K) fingerprints/descriptors of primary screen
        y_p:                (M,) primary screen labels
        y_f:                (V,) false positive labels (1=FP)        
        y_c:                (V,) true positive labels (1=FP)
        idx:                (V,) positions of primary actives with confirmatory    
                            readout  
        replicates:         number of replicates to use for the run
     
    Returns:
        Tuple containing one array (1,9) with metrics for FP and TP retrieval, 
        scaffold diversity and training time, and one dataframe (V,5) with
        SMILES, true labels and raw predictions
    """

    #create results containers
    temp = np.zeros((replicates,9))
    fp_box = []
    tp_box = []
    
    primary_idx = np.where(y_p==1)[0]
    
    x_train = x[primary_idx]
    
    #loop analysis over replicates
    for j in range(replicates):
        
        #train Isolation Forest model, get sample importance and time it
        start = time.time()
        model = IsolationForest(n_jobs=-1, random_state=j)
        model.fit(x)
        temp[j,0] = time.time() - start
        
        #get predictions and reshape
        preds = model.score_samples(x_train)
        preds_dummy = np.zeros((len(mols)))
        preds_dummy[primary_idx] = -preds
        flags, flags_alt = process_ranking(y_p, preds_dummy)

        #get precision@90 for FP and TP retrieval
        temp[j,1] = precision_score(y_f, flags[idx])
        temp[j,2] = precision_score(y_c, flags_alt[idx])
        
        #get EF10 for FPs and TPs
        temp[j,3] = enrichment_factor_score(y_f, flags[idx])
        temp[j,4] = enrichment_factor_score(y_c, flags_alt[idx])

        #get BEDROC20 for FPs and TPs
        temp[j,5] = bedroc_score(y_f, preds_dummy[idx], reverse=True)
        temp[j,6] = bedroc_score(y_c, preds_dummy[idx], reverse=False)	 
        
        #get scaffold diversity for compounds that got flagged as FPs
        idx_fp = np.where(flags == 1)[0]
        mols_fp = [mols[x] for x in idx_fp]
        temp[j,7] = get_scaffold_rate(mols_fp)

        #get scaffold diversity for compounds that got flagged as TPs
        idx_tp = np.where(flags_alt == 1)[0]
        mols_tp = [mols[x] for x in idx_tp]
        temp[j,8] = get_scaffold_rate(mols_tp)
        
        #append raw predictions
        fp_box.append(flags[idx])
        tp_box.append(flags_alt[idx])

    #create logger dataframe
    mols_subset = [mols[x] for x in idx]
    logs = run_logger(mols_subset, y_f, y_c,
                          fp_box, tp_box, replicates)
    
    return temp, logs

#-----------------------------------------------------------------------------#

def run_vae(
        mols: List[rdkit.Chem.rdchem.Mol],
        y_p: np.ndarray,
        y_f: np.ndarray,
        y_c: np.ndarray,
        idx: List[int],
        replicates: int,
        ) -> Tuple[np.ndarray, pd.DataFrame]:
    """Executes VAE analysis on given dataset

    Uses VAE anomaly detection to rank primary screen actives in terms of 
    likelihood of being false positives or true positives. The ranking is calculated
    according to the reconstruction error, with high reconstruction indicating
    anomalies.
    Finally, the function computes precision@90, EF10, BEDROC20 and scaffold
    diversity metrics.
    
    Args:
        mols:               (M,) mol objects from primary data
        y_p:                (M,) primary screen labels
        y_f:                (V,) false positive labels (1=FP)        
        y_c:                (V,) true positive labels (1=FP)
        idx:                (V,) positions of primary actives with confirmatory    
                            readout  
        replicates:         number of replicates to use for the run
     
    Returns:
        Tuple containing one array (1,9) with metrics for FP and TP retrieval, 
        scaffold diversity and training time, and one dataframe (V,5) with
        SMILES, true labels and raw predictions
    """

    #create results containers
    temp = np.zeros((replicates,9))
    fp_box = []
    tp_box = []
    
    primary_idx = np.where(y_p==1)[0]
    
    primary_mols = [mols[x] for x in primary_idx]
    smiles = [Chem.MolToSmiles(x) for x in primary_mols]
    x = tokenize(smiles)
    dataset = VAEDataset(x)
    generator = gen2 = DataLoader(dataset, batch_size=32, shuffle=True)
    
    #loop analysis over replicates
    for j in range(replicates):
        
        #train Isolation Forest model, get sample importance and time it
        start = time.time()
        model = VAE(dict_size=x.shape[2],
                    max_length=x.shape[1]-1)
        model.train(generator)
        temp[j,0] = time.time() - start
        
        #get predictions and reshape
        preds = model.predict(x)
        preds_dummy = np.zeros((len(mols)))
        preds_dummy[primary_idx] = preds
        flags, flags_alt = process_ranking(y_p, preds_dummy)

        #get precision@90 for FP and TP retrieval
        temp[j,1] = precision_score(y_f, flags[idx])
        temp[j,2] = precision_score(y_c, flags_alt[idx])
        
        #get EF10 for FPs and TPs
        temp[j,3] = enrichment_factor_score(y_f, flags[idx])
        temp[j,4] = enrichment_factor_score(y_c, flags_alt[idx])

        #get BEDROC20 for FPs and TPs
        temp[j,5] = bedroc_score(y_f, preds_dummy[idx], reverse=True)
        temp[j,6] = bedroc_score(y_c, preds_dummy[idx], reverse=False)	 
        
        #get scaffold diversity for compounds that got flagged as FPs
        idx_fp = np.where(flags == 1)[0]
        mols_fp = [mols[x] for x in idx_fp]
        temp[j,7] = get_scaffold_rate(mols_fp)

        #get scaffold diversity for compounds that got flagged as TPs
        idx_tp = np.where(flags_alt == 1)[0]
        mols_tp = [mols[x] for x in idx_tp]
        temp[j,8] = get_scaffold_rate(mols_tp)
        
        #append raw predictions
        fp_box.append(flags[idx])
        tp_box.append(flags_alt[idx])

    #create logger dataframe
    mols_subset = [mols[x] for x in idx]
    logs = run_logger(mols_subset, y_f, y_c,
                          fp_box, tp_box, replicates)
    
    return temp, logs
