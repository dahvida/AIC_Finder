import numpy as np
from rdkit.ML.Scoring.Scoring import CalcBEDROC
from sklearn.metrics import precision_score
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors
from rdkit import DataStructs
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import Chem
import pandas as pd

###############################################################################

def get_ECFP(mols, radius = 2, nbits = 1024):

    #create empty array as container for ecfp
    array = np.empty((len(mols), nbits), dtype=np.float32)

    #get ecfps in list
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, radius, nbits) for x in mols]
    
    #store each element in list into array via cython
    for i in range(len(array)):
        DataStructs.ConvertToNumpyArray(fps[i], array[i])
    return array

#-----------------------------------------------------------------------------#

def get_scaffold_rate(mols):
    
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
        rate = 0

    return rate
    
#-----------------------------------------------------------------------------#
    
def get_labels(dataframe):

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

def process_FP(y, vals_box, percentile = 90):

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
    idx_pro = np.where(scores_pos > t1)[0]
    idx_opp = np.where(scores_pos < t2)[0]
    
    #fill respective arrays with respective labels
    proponent[idx_pos[idx_pro]] = 1
    opponent[idx_pos[idx_opp]] = 1
    
    return proponent, opponent

#-----------------------------------------------------------------------------#

def store_row(analysis_array,
              dataset_array,
              fp_rate,
              tp_rate,
              index):
    
    analysis_array[index, 0] = np.mean(dataset_array[:,0])
    analysis_array[index, 1] = np.std(dataset_array[:,0])
    analysis_array[index, 2] = fp_rate
    analysis_array[index, 3] = np.mean(dataset_array[:,1])
    analysis_array[index, 4] = np.std(dataset_array[:,1])
    analysis_array[index, 5] = tp_rate
    analysis_array[index, 6] = np.mean(dataset_array[:,2])
    analysis_array[index, 7] = np.std(dataset_array[:,2])
    analysis_array[index, 8] = np.mean(dataset_array[:,3])
    analysis_array[index, 9] = np.std(dataset_array[:,3])
    
    return analysis_array

#-----------------------------------------------------------------------------#

def save_results(results,
                 dataset_names,
                 filename,
                 filter_type
                 ):
    
    column_names = [
                "Time - mean", "Time - STD",
                "FP rate",
                "FP Precision@90 - mean", "FP Precision@90 - STD",
                "TP rate",
                "TP Precision@90 - mean", "TP Precision@90 - STD",
                "Scaffold - mean", "Scaffold - std"
                ]
    
    prefix = "../Results/"
    suffix = ".csv"
    algorithm = ["mvsa_", "catboost_", "filter_" + filter_type + "_"]
    
    for i in range(len(results)):
        if np.sum(results[i]) != 0:
            db = pd.DataFrame(
                data = results[i],
                index = dataset_names,
                columns = column_names
                )
            db.to_csv(prefix + algorithm[i] + filename + suffix)
    

    

