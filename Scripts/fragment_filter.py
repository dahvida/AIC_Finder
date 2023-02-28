from typing import *
import rdkit
from rdkit import Chem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
import numpy as np

###############################################################################

def filter_predict(
        mols: List[rdkit.Chem.rdchem.Mol],
        catalog_name: str
        ) -> List[int]:
    """Uses structural alerts to predict whether compounds are TP or FP
    
    Args:
        mols:           (M,) molecules to predict
        catalog_name:   name of the structural alerts set to use for
                        predictions

    Returns:
        list (M,) of predictions according to chosen structural alert
    """
    #create RDKIT filter catalog dictionary (could be expanded)
    catalogs = {
        "PAINS":    FilterCatalogParams.FilterCatalogs.PAINS,
        "PAINS_A":  FilterCatalogParams.FilterCatalogs.PAINS_A,
        "PAINS_B":  FilterCatalogParams.FilterCatalogs.PAINS_B,
        "PAINS_C":  FilterCatalogParams.FilterCatalogs.PAINS_C,
        "NIH":      FilterCatalogParams.FilterCatalogs.NIH
        }
    
    #create substructure checker according to fragment set
    params = FilterCatalogParams()
    params.AddCatalog(catalogs[catalog_name])
    catalog = FilterCatalog(params)

    #check all mols and flag ones that have a match
    verdict = np.zeros((len(mols),))
    for i in range(len(mols)):
        if mols[i] is not None:
            entry = catalog.GetFirstMatch(mols[i])
            if entry is not None:
                verdict[i] = 1
    
    return verdict
    
    

    
