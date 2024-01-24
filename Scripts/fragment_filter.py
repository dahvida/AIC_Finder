from typing import *
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, rdmolops
import numpy as np
import pandas as pd

###############################################################################

def filter_predict(
        mols: List[rdkit.Chem.rdchem.Mol]
        ) -> List[int]:
    """Uses structural alerts (REOS and GSK) to predict whether compounds 
    are TP or FP. Sources:
        https://www.sciencedirect.com/science/article/pii/S2472555222068800#ec1
        https://www.nature.com/articles/nrd1063#Sec3    
    
    Args:
        mols:           (M,) molecules to predict

    Returns:
        list (M,) of predictions according to the structural alert. The integer
        denotes the number of alerts triggered by the compound.
    """
    
    filter_db = pd.read_csv("filters.csv")
    smarts = list(filter_db["SMARTS"])
    checks = [Chem.MolFromSmarts(x) for x in smarts]
    [Chem.GetSSSR(x) for x in checks]
    
    structural_alerts = np.zeros(len(mols))
    for i in range(len(mols)):
        for j in range(len(checks)):
            if mols[i].HasSubstructMatch(checks[j]):
                structural_alerts[i] += 1
    
    physchem_alerts = np.zeros(len(mols))
    for i in range(len(mols)):
        mw = Descriptors.ExactMolWt(mols[i])
        logp = Crippen.MolLogP(mols[i])
        hbd = rdMolDescriptors.CalcNumHBD(mols[i])
        hba = rdMolDescriptors.CalcNumHBA(mols[i])
        charge = rdmolops.GetFormalCharge(mols[i])
        rb = rdMolDescriptors.CalcNumRotatableBonds(mols[i])
        ha = mols[i].GetNumHeavyAtoms()
        if mw < 200 or mw > 500:
            physchem_alerts[i] += 1
        if logp < -5.0 or logp > 5.0:
            physchem_alerts[i] += 1
        if hbd > 5:
            physchem_alerts[i] += 1
        if hba > 10:
            physchem_alerts[i] += 1
        if charge < -2 or charge > 2:
            physchem_alerts[i] += 1
        if rb > 8:
            physchem_alerts[i] += 1
        if ha < 15 or ha > 50:
            physchem_alerts[i] += 1
    
    return structural_alerts + physchem_alerts    
    

    
