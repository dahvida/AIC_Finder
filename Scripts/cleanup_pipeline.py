import pandas as pd
from rdkit import Chem
import numpy as np
from chembl_structure_pipeline import standardizer
import sys
import argparse

###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--AID_1')		
parser.add_argument('--AID_2')
parser.add_argument('--filename')
args = parser.parse_args()

###############################################################################

#Preprocessing function to handle duplicate measurements in one assay
def f_1(x):
    if x > 0.5:
        return 1
    else:
        return 0
    
###############################################################################

def main(
        AID_1,
        AID_2,
        filename
        ):
    
    #print info of run
    print("[cleanup]: Beginning dataset cleanup run...")
    print(f"[cleanup]: Currently merging AID {AID_1} (Primary) and AID {AID_2} (Confirmatory)")
    print(f"[cleanup]: Dataset will be saved as {filename}.csv")
    
    #create preprocessing log container
    logs = []

    #load and remove NaN SMILES
    t1 = pd.read_csv("../Raw_data/AID_"+AID_1+"_datatable_all.csv", low_memory=False)
    t2 = pd.read_csv("../Raw_data/AID_"+AID_2+"_datatable_all.csv", low_memory=False)
    t1.dropna(subset=["PUBCHEM_EXT_DATASOURCE_SMILES"], inplace=True)
    t2.dropna(subset=["PUBCHEM_EXT_DATASOURCE_SMILES"], inplace=True)

    #get smiles
    smiles_1 = list(t1["PUBCHEM_EXT_DATASOURCE_SMILES"])
    smiles_2 = list(t2["PUBCHEM_EXT_DATASOURCE_SMILES"])
    
    #cut off possible non-string last rows
    for i in range(len(smiles_1)):
        if isinstance(smiles_1[i], str) is True:
            lim_1 = i
            break
    smiles_1 = smiles_1[lim_1:]
    
    for i in range(len(smiles_2)):
        if isinstance(smiles_2[i], str) is True:
            lim_2 = i
            break
    smiles_2 = smiles_2[lim_2:]

    #get labels
    act_1 = list(t1["PUBCHEM_ACTIVITY_OUTCOME"])[lim_1:]
    act_2 = list(t2["PUBCHEM_ACTIVITY_OUTCOME"])[lim_2:]

    #convert to mol, sanitize and store original dataset sizes
    mols_1 = [Chem.MolFromSmiles(x, sanitize=True) for x in smiles_1]
    mols_2 = [Chem.MolFromSmiles(x, sanitize=True) for x in smiles_2]
    n_original_1 = len(mols_1)
    n_original_2 = len(mols_2)
    n_act_1 = act_1.count("Active")
    n_act_2 = act_2.count("Active")

    #filter NoneType
    idx = [x for x in list(range(len(mols_1))) if mols_1[x] is not None]
    mols_1 = [mols_1[x] for x in idx]
    act_1 = [act_1[x] for x in idx]
    idx = [x for x in list(range(len(mols_2))) if mols_2[x] is not None]
    mols_2 = [mols_2[x] for x in idx]
    act_2 = [act_2[x] for x in idx]

    #remove salts
    mols_1 = [standardizer.get_parent_mol(x)[0] for x in mols_1]
    mols_2 = [standardizer.get_parent_mol(x)[0] for x in mols_2]

    #return to SMILES
    smiles_1 = [Chem.MolToSmiles(x) for x in mols_1]
    smiles_2 = [Chem.MolToSmiles(x) for x in mols_2]

    #convert labels in binary
    idx = [x for x in list(range(len(act_1))) if act_1[x] == "Active"]
    act_1 = np.zeros((len(act_1),))
    act_1[idx] = 1
    idx = [x for x in list(range(len(act_2))) if act_2[x] == "Active"]
    act_2 = np.zeros((len(act_2),))
    act_2[idx] = 1

    #remove duplicates by aggregating according to consensus
    db_1 = pd.DataFrame({"SMILES": smiles_1, "Primary":act_1})   
    db_1 = db_1.groupby(["SMILES"], as_index=False).mean()
    db_1["Primary"] = db_1["Primary"].apply(f_1)
    db_2 = pd.DataFrame({"SMILES": smiles_2, "Confirmatory":act_2})   
    db_2 = db_2.groupby(["SMILES"], as_index=False).mean()
    db_2["Confirmatory"] = db_2["Confirmatory"].apply(f_1)
    
    #merge and remove molecules that somehow were constructed incorrectly
    db_final = pd.merge(db_1, db_2, how="left")
    db_final["Mols"] = [Chem.MolFromSmiles(x) for x in list(db_final["SMILES"])]
    db_final.dropna(inplace=True, subset=["Mols"])
    db_final.drop(["Mols"], inplace=True, axis=1)
    
    #log differences before/after preprocessing
    n_primary = len(db_final)
    n_confirmatory = len(db_final[~db_final['Confirmatory'].isnull()])
    logs.append("Primary - Compounds before preprocessing: " + str(n_original_1))
    logs.append("Primary - Actives before preprocessing: " + str(n_act_1))
    logs.append("Primary - Compounds after preprocessing: " + str(n_primary))
    logs.append("Primary - Actives after preprocessing: " + str(np.sum(db_final["Primary"])))
    logs.append("Confirmatory - Compounds before preprocessing: " + str(n_original_2))
    logs.append("Confirmatory - Actives before preprocessing: " + str(n_act_2))
    logs.append("Confirmatory - Compounds after preprocessing: " + str(n_confirmatory))
    logs.append("Confirmatory - Actives after preprocessing: " + str(np.sum(db_final["Confirmatory"])))

    #test final dataframe
    #current tests check that all confirmatory compounds were present before in the primary,
    #and that all duplicate records in either assay have been merged / removed
    db_primary = db_final.dropna(subset=["Primary"])
    assert len(db_final) == len(db_primary), "[cleanup]: All compounds in confirmatory should be present in primary assay"
    db_cut = db_final.drop_duplicates(subset=["SMILES"])
    assert len(db_final) == len(db_cut), "[cleanup]: There shouldn't be any more duplicate SMILES"
    print(f"[cleanup]: Tests passed successfully")

    #save dataframe
    path = "../Datasets/" + filename + ".csv"
    db_final.to_csv(path)
    print(f"[cleanup]: File saved at {path}")

    #save logs
    logpath = "../Logs/cleanup/" + filename + ".txt"
    with open(logpath, 'w') as f:
        for line in logs:
            f.write(f"{line}\n")
    print(f"[cleanup]: Log saved at {logpath}")


if __name__ == "__main__":
    main(
        AID_1 = args.AID_1,
        AID_2 = args.AID_2,
        filename = args.filename
        )



