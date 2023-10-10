import pandas as pd
from MVS_A import *
from utils import get_ECFP
from rdkit import Chem
from scipy.stats import percentileofscore



def main():
    db_confirmed = pd.read_csv("../Misc/transporter_confirmed.csv")
    db_original = pd.read_csv("../Datasets/transporter.csv")
 
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
                      		y = y, 		#binary labels from primary screen
                      		params = "default",	#could pass a dict with custom params (not recommended)
                      		verbose = True,	#option to print updates on calculation status
                      		seed = 0		#random seed for reproducibility
                      		)

    #train base lightgbm model
    AIC_finder.get_model()

    #get raw sample importances
    vals = AIC_finder.get_importance()

    #get percentiles for mvsa and score for all confirmed smiles
    idx_actives = np.where(y==1)[0]
    vals_actives = vals[idx_actives]
    scores = np.array(db_original["Score"])
    score_actives = scores[idx_actives]
    mvsa_percentiles = [100-percentileofscore(vals_actives, vals[x]) for x in idx_original]
    score_percentiles = [percentileofscore(score_actives, scores[x]) for x in idx_original]
    
    
    output = pd.DataFrame({
        "SMILES": [smi_confirmed[x] for x in idx_confirmed],
        "Primary label": y[idx_original],
        "MVS-A percentile": mvsa_percentiles,
        "Score percentile": score_percentiles
        })
    
    output.to_csv("../Results/case_study.csv")
    
if __name__ == "__main__":
    main()
    
