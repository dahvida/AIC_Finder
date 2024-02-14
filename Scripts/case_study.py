import pandas as pd
from MVS_A import *
from utils import get_ECFP
from rdkit import Chem
from scipy.stats import percentileofscore
from fragment_filter import *

def main():
    db_confirmed = pd.read_csv("../Misc/transporter_confirmed.csv")
    db_original = pd.read_csv("../Datasets/transporter.csv")
    scores = np.array(db_original["Score"])
 
    smi_confirmed = list(db_confirmed["SMILES"])
    smi_original = list(db_original["SMILES"])
    
    idx_original = [None]*len(smi_confirmed)
    idx_confirmed = []
    for i in range(len(smi_confirmed)):
        for j in range(len(smi_original)):
            if smi_confirmed[i] == smi_original[j]:
                idx_original[i] = j
                idx_confirmed.append(i)
    
    idx_original = list(filter(None, idx_original))

    mols = [Chem.MolFromSmiles(x) for x in smi_original]
    ecfp = get_ECFP(mols)
    y = np.array(db_original["Primary"])
    AIC_finder = sample_analysis(
                    		x = ecfp,			#input data i.e. ECFP
                      		y = y,              #binary labels from primary screen
                      		params = "default",	#could pass a dict with custom params (not recommended)
                      		verbose = True,	    #option to print updates on calculation status
                      		seed = 0		    #random seed for reproducibility
                      		)

    #train base lightgbm model
    AIC_finder.get_model()

    #get raw sample importances
    vals = AIC_finder.get_importance()

    #run ranking for TPs and FNs and store in dataframes
    idx_actives = np.where(y==1)[0]
    idx_inactives = np.where(y==0)[0]

    db_tp = pd.DataFrame({
        "SMILES": [smi_original[x] for x in idx_actives],
        "MVS-A": vals[idx_actives],
        "Score": scores[idx_actives]
        })
    db_tp.sort_values(["MVS-A"], ascending=True, inplace=True)

    db_fn = pd.DataFrame({
        "SMILES": [smi_original[x] for x in idx_inactives],
        "MVS-A": vals[idx_inactives],
        "Score": scores[idx_inactives]
        })
    db_fn.sort_values(["MVS-A"], ascending=False, inplace=True)

    #get percentiles and add them
    mvsa_tp = np.array(db_tp["MVS-A"])
    score_tp = np.array(db_tp["MVS-A"])
    mvsa_tp_p = [100-percentileofscore(mvsa_tp, x) for x in mvsa_tp]
    score_tp_p = [percentileofscore(score_tp, x) for x in score_tp]
    db_tp["MVS-A percentile"] = mvsa_tp_p
    db_tp["Score percentile"] = score_tp_p

    mvsa_fn = np.array(db_fn["MVS-A"])
    mvsa_fn_p = [percentileofscore(mvsa_fn, x) for x in mvsa_fn]
    score_fn_p = [0]*len(mvsa_fn_p)                 #percentiles for score are 0s for primary inactives
    db_fn["MVS-A percentile"] = mvsa_fn_p
    db_fn["Score percentile"] = score_fn_p

    #add confirmatory labels
    smi_target = [smi_confirmed[x] for x in idx_confirmed]
    smi_tp = list(db_tp["SMILES"])
    smi_fn = list(db_fn["SMILES"])
    confirm_tp = [0]*len(smi_tp)
    confirm_fn = [0]*len(smi_fn)
    
    for i in range(len(smi_target)):
        for j in range(len(smi_tp)):
            if smi_target[i] == smi_tp[j]:
                confirm_tp[j] = 1
                
    for i in range(len(smi_target)):
        for j in range(len(smi_fn)):
            if smi_target[i] == smi_fn[j]:
                confirm_fn[j] = 1
    
    db_tp["Confirmed"] = confirm_tp
    db_fn["Confirmed"] = confirm_fn

    #add fragment filter predictions to TP dataframe
    mols_tp = [Chem.MolFromSmiles(x) for x in smi_tp]
    filter_preds = filter_predict(mols_tp)
    db_tp["Filter alerts"] = filter_preds

    #save db
    db_tp.to_csv("../Results/case_study_tp.csv")
    db_fn.to_csv("../Results/case_study_fn.csv")

if __name__ == "__main__":
    main()
    
