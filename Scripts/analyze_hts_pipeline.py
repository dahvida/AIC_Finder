"""Script to automate HTS analysis on new datasets.

Helper script to automate MVS-A analysis on a new dataset. All you need to do is
provide a .csv file with two columns: one with SMILES, one with activity labels,
so that active compounds are indicated with "1" and inactive compounds with "0".
As an example, you can check any of the datasets in the ../Datasets folder. 
The script will load the file, compute ECFP fingerprints for each compound, run
the MVS-A analysis and check for structural alerts. It will have two outputs:
    1. A .csv file with all HTS hits, sorted from most likely to be true positive
    to least likely to be true positive (aka a false positive)
    2. A .csv file with all HTS non-hits, sorted from most likely to be false negative
    to least likely to be false negative (aka a true negative)
Given an input file, e.g. "input.csv", the first file will be named "input_actives.csv",
While the second will be "input_inactives.csv". They will be saved in the same path
as the input file.
For each compound, the files will contain its SMILES, the MVS-A score and all
triggered structural alerts. To visualize the SMARTS fragments, you can use
this tool: https://smarts.plus/

Steps:
    1. Load dataset and compute chosen molecular feature set
    2. Run MVS-A analysis
    3. Run structural alerts analysis
    4. Sort in terms of true positive and false negative likelihood
    5. Save outputs
"""

from MVS_A import sample_analysis
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, rdmolops
from utils import get_ECFP
import argparse

###############################################################################

parser = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--input_path', default="all",
                    help="Path to target .csv file to use for the analysis")

parser.add_argument('--smiles_col', default="smiles",
                    help="Name of column containing the SMILES of the dataset")

parser.add_argument('--y_col', default="y",
                    help="Name of the column containing the activity labels. 1 = active, 0 = inactive")

args = parser.parse_args()

###############################################################################

def GSK_REOS_filter(
        mols
        ):
    
    filter_db = pd.read_csv("filters.csv")
    smarts = list(filter_db["SMARTS"])
    checks = [Chem.MolFromSmarts(x) for x in smarts]
    [Chem.GetSSSR(x) for x in checks]
    
    alerts = []
    for i in range(len(mols)):
        mol_alerts = []
        for j in range(len(checks)):
            if mols[i].HasSubstructMatch(checks[j]):
                mol_alerts.append(smarts[j])
                
        mw = Descriptors.ExactMolWt(mols[i])
        logp = Crippen.MolLogP(mols[i])
        hbd = rdMolDescriptors.CalcNumHBD(mols[i])
        hba = rdMolDescriptors.CalcNumHBA(mols[i])
        charge = rdmolops.GetFormalCharge(mols[i])
        rb = rdMolDescriptors.CalcNumRotatableBonds(mols[i])
        ha = mols[i].GetNumHeavyAtoms()
        
        if mw < 200 or mw > 500:
            mol_alerts.append("Molecular weight")
        if logp < -5.0 or logp > 5.0:
            mol_alerts.append("LogP")
        if hbd > 5:
            mol_alerts.append("Hydrogen bond donor")
        if hba > 10:
            mol_alerts.append("Hydrogen bond acceptor")
        if charge < -2 or charge > 2:
            mol_alerts.append("Charge")
        if rb > 8:
            mol_alerts.append("Rotatable bonds")
        if ha < 15 or ha > 50:
            mol_alerts.append("Heavy atom count")
        
        alerts.append(mol_alerts)
    
    return alerts

def main(input_path, smiles_col, y_col):
    
    print("### Analysis started ###")
    
    # load file
    db = pd.read_csv(input_path)
    
    # extract relevant cols
    smiles = list(db[smiles_col])
    y = np.array(db[y_col])
    
    # generate ECFPs
    mols = [Chem.MolFromSmiles(x) for x in smiles]
    feats = get_ECFP(mols)
    
    # run MVS-A
    obj = sample_analysis(x = feats,
                          y = y, 
                          params = "default",
                          verbose = True,
                          seed = 0)
    obj.get_model()
    vals = obj.get_importance()
    
    # run structural alerts analysis
    alerts = GSK_REOS_filter(mols)
    
    # store in dataframe
    db_results = pd.DataFrame({
        "SMILES": smiles,
        "Activity": y,
        "MVS-A": vals,
        "Alerts": alerts
        })
    
    # slice and sort
    db_act = db_results.loc[db_results["Activity"] == 1]
    db_act.sort_values("MVS-A", inplace=True, ascending=True)
    db_inact = db_results.loc[db_results["Activity"] == 0]
    db_inact.sort_values("MVS-A", inplace=True, ascending=False)
    
    # save outputs
    db_act.to_csv(input_path[:-4] + "_actives.csv")
    db_inact.to_csv(input_path[:-4] + "_inactives.csv")
    
    print("### Analysis finished ###")

if __name__ == "__main__":
    main(
        args.input_path,
        args.smiles_col,
        args.y_col
        )
    


