from typing import *
import numpy as np
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Scoring.Scoring import CalcBEDROC
import pandas as pd

###############################################################################

def get_ECFP(
        mols: List[rdkit.Chem.rdchem.Mol],
        radius:int = 2,
        nbits:int = 1024
        ) -> np.ndarray:
    """Calculates ECFPs for given set of molecules
    
    Args:
        mols:   (M,) mols to compute ECFPs  
        radius: radius for fragment calculation
        nbits:  bits available for folding ECFP

    Returns:
        array (M,1024) of ECFPs 
    """
    #create empty array as container for ecfp
    array = np.empty((len(mols), nbits), dtype=np.float32)

    #get ecfps in list
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, radius, nbits) for x in mols]
    
    #store each element in list into array via cython
    for i in range(len(array)):
        DataStructs.ConvertToNumpyArray(fps[i], array[i])

    return array

#-----------------------------------------------------------------------------#

def get_MACCS(
	mols: List[rdkit.Chem.rdchem.Mol]
	) -> np.ndarray:
    """Calculates MACCS keys for a given set of molecules

    Args:
        mols:   (M,) mols to compute MACCS for

    Returns:
        array (M, 167) of MACCS keys
    """
    #prealloc
    array = np.empty((len(mols), 167), dtype=np.float32) 

    #get MACCS keys
    fps = [MACCSkeys.GenMACCSKeys(x) for x in mols]

    #store each element in list into array via cython
    for i in range(len(array)):
        DataStructs.ConvertToNumpyArray(fps[i], array[i])
    
    return array

#-----------------------------------------------------------------------------#
	
def get_2D(
    mols: List[rdkit.Chem.rdchem.Mol]
    ) -> np.ndarray:
    """Calculates a set of 2D descriptors for a given set of molecules

    Args:
        mols:   (M,) mols to compute 2D descs for

    Returns:
        array (M, 100) of 2D descriptors
    """
    #fetch descriptor names
    names = [x[0] for x in Descriptors._descList]
    
    #select only top 100 for calculation speed
    names = names[:100]
    
    #create calculator object only for selected descs
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(names)

    #prealloc array
    descs = np.zeros((len(mols), 100))
    
    #loop over each molecule
    for i in range(len(descs)):

        #calc desc and store if not None
        temp = calc.CalcDescriptors(mols[i])
        if temp is not None:
            descs[i,:] = temp
    
    #prune outliers
    descs = np.nan_to_num(descs, posinf=10e10, neginf=-10e10)
    
    return descs

#-----------------------------------------------------------------------------#

def get_scaffold_rate(
        mols: List[rdkit.Chem.rdchem.Mol]
        ) -> float:
    """Computes scaffold diversity for given set of molecules
    
    Args:
        mols:   (M,) mols to check for scaffold diversity

    Returns:
        percentage of unique Murcko scaffolds in the set, as the number of
        unique Murcko scaffolds divided by the number of molecules
    """
    
    #safety check (can happen with datasets with very low % of
    #primary actives in confirmatory dataset)
    if len(mols) > 0:

        #count mols, create empty Murcko scaffold list
        tot_mols = len(mols)
        scaffs = [0]*len(mols)

        #check SMILES for Murcko scaffold
        smiles = [Chem.MolToSmiles(x) for x in mols]
        for i in range(len(mols)):
            scaffs[i] = MurckoScaffold.MurckoScaffoldSmiles(smiles[i])
        
        #remove scaffold duplicates and compare to mol count
        scaffs = list(set(scaffs))
        n_scaffs = len(scaffs)
        rate = n_scaffs * 100 / tot_mols
    else:
        rate = 0.0

    return rate
    
#-----------------------------------------------------------------------------#
   
def get_labels(
        dataframe: pd.DataFrame,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
    """Fetches all necessary labels for AIC analysis
    
    Args:
        dataframe:  (M,3) table with merged primary and confirmatory assay 
                    information

    Returns:
        A tuple containing:
            1. (M,) Primary screen labels
            2. (V,) FP labels for primary actives with confirmatory readout
            3. (V,) TP labels for primary actives with confirmatory readout
            4. (V,) Position of primary actives with confirmatory readout inside
            primary screen data

    """

    #primary labels for training ML models
    y_p = np.array(dataframe["Primary"])

    #get slice with compounds with confirmatory measurement
    selected_rows = dataframe[~dataframe['Confirmatory'].isnull()]
    
    #further cut slice to select only compounds that were primary actives
    selected_rows = selected_rows.loc[selected_rows['Primary'] == 1]
    
    #confirmatory readout becomes TP vector (primary matches confirmatory)
    y_c = np.array(selected_rows["Confirmatory"])

    #FP vector as opposite of TP vector
    y_f = (y_c - 1) * -1

    #position of final compounds in primary screen data
    idx = np.array(selected_rows.index)
    
    return y_p, y_c, y_f, idx

#-----------------------------------------------------------------------------#

def process_ranking(
        y: np.ndarray,
        vals_box: List[float],
        percentile: int = 90
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Converts raw importance scores in binary predictions using percentiles
    
    Args:
        y:          (M,) primary screen labels
        vals_box:   (M,) raw importance scores
        percentile: value to use for thresholding

    Returns:
        A tuple containing two arrays (M,) with labels indicating whether
        compounds are TPs or FPs according to the importances predicted by
        the ML model. First element of the tuple are the indexes of compounds
        who have a score >90%, the second element are the ones <10%.
    """

    #select importance scores from primary actives
    idx_pos = np.where(y == 1)[0]
    scores_pos = vals_box[idx_pos]
    
    #compute top 90% and bottom 10% percentile thresholds. since this is done
    #only by looking primary screening data, it is completely unbiased against
    #leakage from confirmatory screen data
    t1 = np.percentile(scores_pos, percentile)
    t2 = np.percentile(scores_pos, 100 - percentile)
    
    #create empty arrays (proponent = compound above 90% threshold)
    proponent = np.zeros((len(y),))
    opponent = np.zeros((len(y),))
    
    #find primary actives which fall outside of either threshold
    idx_pro = np.where(scores_pos >= t1)[0]
    idx_opp = np.where(scores_pos <= t2)[0]
    
    #fill respective arrays with respective labels. in this context,
    #proponents are samples >90%, while opponents are <10%
    proponent[idx_pos[idx_pro]] = 1
    opponent[idx_pos[idx_opp]] = 1
    
    return proponent, opponent

#-----------------------------------------------------------------------------#

def enrichment_factor_score(
        y_true: np.ndarray,
        y_pred: np.ndarray
        ) -> float:
    """
    Function to compute Enrichment Factor using precomputed binary labels
    according to the threshold set in process_ranking
    """
    compounds_at_k = np.sum(y_pred)
    total_compounds = len(y_true)
    total_actives = np.sum(y_true)
    tp_at_k = len(np.where(y_true + y_pred == 2)[0])

    return (tp_at_k / compounds_at_k) * (total_actives / total_compounds)

#-----------------------------------------------------------------------------#

def bedroc_score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        reverse: bool = True
        ) -> float:
    """
    Function to compute BEDROC score from raw rankings, using alpha=20 as
    default. Adapted from https://github.com/deepchem
    """
    
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    scores = list(zip(y_true, y_pred))
    scores = sorted(scores, key=lambda pair: pair[1], reverse=reverse)
    
    output = CalcBEDROC(scores = scores,
                        col = 0,
                        alpha = 20)

    return output

#-----------------------------------------------------------------------------#

def store_row(
        analysis_array: np.ndarray,
        dataset_array: np.ndarray,
        fp_rate: float,
        tp_rate: float,
        index: int
        ) -> np.ndarray:
    """Stores i-th dataset results in performance container for X datasets
    
    Args:
        analysis_array: (X,20) dataframe that stores results of a given
                        algorithm for all datasets
        dataset_array:  (1,9) array with the results of a given algorithm on
                        the i-th dataset
        fp_rate:        fraction of false positives in the confirmatory dataset
        tp_rate:        fraction of true positives in the confirmatory dataset
        index:          i-th row position to store results in

    Returns:
        Updated analysis array with results stored in the correct row (not the
        most elegant solution but at least it provides a straightforward way
        to handle both single and multi dataset performance collection)
    """

    analysis_array[index, 0] = np.mean(dataset_array[:,0])      #mean training time
    analysis_array[index, 1] = np.std(dataset_array[:,0])       #STD training time

    analysis_array[index, 2] = fp_rate                          #baseline FP rate
    analysis_array[index, 3] = np.mean(dataset_array[:,1])      #mean precision@90 FP
    analysis_array[index, 4] = np.std(dataset_array[:,1])       #STD precision@90 FP
    
    analysis_array[index, 5] = tp_rate                          #baseline TP rate
    analysis_array[index, 6] = np.mean(dataset_array[:,2])      #mean precision@90 TP
    analysis_array[index, 7] = np.std(dataset_array[:,2])       #STD precision@90 TP

    analysis_array[index, 8] = np.mean(dataset_array[:,3])      #EF10 for FP
    analysis_array[index, 9] = np.std(dataset_array[:,3])       #STD EF10 FP

    analysis_array[index, 10] = np.mean(dataset_array[:,4])     #EF10 for TP
    analysis_array[index, 11] = np.std(dataset_array[:,4])      #STD EF10 TP

    analysis_array[index, 12] = np.mean(dataset_array[:,5])     #BEDROC20 for FP
    analysis_array[index, 13] = np.std(dataset_array[:,5])      #STD

    analysis_array[index, 14] = np.mean(dataset_array[:,6])     #BEDROC20 for TP
    analysis_array[index, 15] = np.std(dataset_array[:,6])      #STD

    analysis_array[index, 16] = np.mean(dataset_array[:,7])     #means FP scaffold diversity
    analysis_array[index, 17] = np.std(dataset_array[:,7])      #STD FP scaffold diversity

    analysis_array[index, 18] = np.mean(dataset_array[:,8])     #means TP scaffold diversity
    analysis_array[index, 19] = np.std(dataset_array[:,8])      #STD TP scaffold diversity
    
    return analysis_array

#-----------------------------------------------------------------------------#

def save_results(
        results: List[np.ndarray],
        dataset_names: List,
        filename: str,
        filter_type: str
        ) -> None:
    """Saves results from all algorithms to their respective .csv files
    
    Args:
        results:        list (4,) containing results arrays for all algorithms.
                        In case one algorithm was not selected for the run, it
                        is stored as an empty array in this list and it will 
                        not be saved to .csv
        dataset_names:  list (X,) of all dataset names analysed in the run
        filename:       common name of the .csv files to use when saving (i.e.
                        if filename=output, the .csv with MVS-A results will be
                        saved as "mvsa_output.csv")
        filter_type:    structural alerts name to append when saving the performance
                        of fragment filters (i.e. "filter_PAINS_output.csv")

    Returns:
        None
    """
    
    column_names = [
                "Time - mean", "Time - STD",
                "FP rate",
                "FP Precision@90 - mean", "FP Precision@90 - STD",
                "TP rate",
                "TP Precision@90 - mean", "TP Precision@90 - STD",
                "FP EF10 - mean", "FP EF10 - STD",
                "TP EF10 - mean", "TP EF10 - STD",
                "FP BEDROC20 - mean", "FP BEDROC20 - STD",
                "TP BEDROC20 - mean", "TP BEDROC20 - STD",
                "FP Scaffold - mean", "FP Scaffold - STD",
                "TP Scaffold - mean", "TP Scaffold - STD"
                ]
    
    prefix = "../Results/"
    suffix = ".csv"
    algorithm = ["mvsa_", "catboost_", "score_", "filter_" + filter_type + "_"]
    
    for i in range(len(results)):
        if np.sum(results[i]) != 0:             #save only if array is not empty
            db = pd.DataFrame(
                    data = results[i],
                    index = dataset_names,
                    columns = column_names
                    )
            db.to_csv(prefix + algorithm[i] + filename + suffix)


    

