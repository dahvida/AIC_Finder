import pandas as pd
import numpy as np
from sklearn.metrics import *
import os
from utils import *
from scipy.stats import wilcoxon

def top_k(preds, k = 10):
    t = np.percentile(preds, 100-k)
    output = preds.copy()
    output[output>t] = 1
    output[output<=t] = 0
    return output

def main(): 
    
    autofluo = []
    colloidal = []
    frequent = []
    filters = []
    
    performance_autofluo = np.zeros((17, 3))
    performance_colloidal = np.zeros((17, 3))
    performance_frequent = np.zeros((17, 3))

    names = os.listdir("../Logs/eval/mvsa")
    path_1 = ["../Logs/eval/fp_detectors/" + x for x in names]
    path_2 = ["../Logs/eval/mvsa/" + x for x in names]
    
    for i in range(len(path_1)):
        db1 = pd.read_csv(path_1[i], index_col=0)
        db2 = pd.read_csv(path_2[i], index_col=0)
        db2["SMILES"] = db2.index
        cols = db2.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        db2 = db2[cols]
        db3 = pd.merge(db2, db1, how="left")
        db3.drop("SMILES", axis=1, inplace=True)
        arr = np.array(db3)
        
        fp_labels = arr[:,0]
        fp_rate = np.sum(fp_labels) / len(fp_labels)
        fluo_preds = arr[:,-4]
        colloidal_preds = arr[:,-3]
        frequent_preds = arr[:,-2]
        
        performance_autofluo[i,0] = precision_score(fp_labels, top_k(fluo_preds)) - fp_rate
        performance_autofluo[i,1] = enrichment_factor_score(fp_labels, top_k(fluo_preds))
        performance_autofluo[i,2] = bedroc_score(fp_labels, fluo_preds, reverse=True)
        
        performance_colloidal[i,0] = precision_score(fp_labels, top_k(colloidal_preds)) - fp_rate
        performance_colloidal[i,1] = enrichment_factor_score(fp_labels, top_k(colloidal_preds))
        performance_colloidal[i,2] = bedroc_score(fp_labels, colloidal_preds, reverse=True)
        
        performance_frequent[i,0] = precision_score(fp_labels, top_k(frequent_preds)) - fp_rate
        performance_frequent[i,1] = enrichment_factor_score(fp_labels, top_k(frequent_preds))
        performance_frequent[i,2] = bedroc_score(fp_labels, frequent_preds, reverse=True)
        
        arr[arr>0.5] = 1
        arr[arr<=0.5] = 0
    
        for k in range(10):
            filters.append(recall_score(arr[:,k+1], arr[:,-1]))
            frequent.append(recall_score(arr[:,k+1], arr[:,-2]))
            colloidal.append(recall_score(arr[:,k+1], arr[:,-3]))
            autofluo.append(recall_score(arr[:,k+1], arr[:,-4]))
        
    output = pd.DataFrame({"Autofluorescence": autofluo,
                           "Colloidal": colloidal,
                           "Frequent": frequent,
                           "Filters": filters})
    output.to_csv("../Results/fp_detectors/overlaps.csv")
    
    dataset_names = [x[:-4] for x in names]
    colnames = ["fp precision", "fp ef10", "fp bedroc"]
    
    db_autofluo = pd.DataFrame(performance_autofluo, index=dataset_names,
                               columns = colnames)
    db_colloidal = pd.DataFrame(performance_colloidal, index=dataset_names,
                               columns = colnames)
    db_frequent = pd.DataFrame(performance_frequent, index=dataset_names,
                               columns = colnames)
    
    db_autofluo.fillna(0.0, inplace=True)
    db_colloidal.fillna(0.0, inplace=True)
    db_frequent.fillna(0.0, inplace=True)
    
    db_autofluo.sort_index(ascending=True, inplace=True)
    db_colloidal.sort_index(ascending=True, inplace=True)
    db_frequent.sort_index(ascending=True, inplace=True)
    
    db_autofluo.to_csv("../Results/fp_detectors/autofluo.csv")
    db_colloidal.to_csv("../Results/fp_detectors/colloidal.csv")
    db_frequent.to_csv("../Results/fp_detectors/frequent.csv")

    mvsa = pd.read_csv("../Results/summary/mvsa.csv", index_col=0)
    stats = np.zeros((3, 3))
    
    for i in range(len(colnames)):
        _, p1 = wilcoxon(mvsa[colnames[i]], db_autofluo[colnames[i]], alternative="greater")
        _, p2 = wilcoxon(mvsa[colnames[i]], db_colloidal[colnames[i]], alternative="greater")
        _, p3 = wilcoxon(mvsa[colnames[i]], db_frequent[colnames[i]], alternative="greater")
        stats[i,0] = p1
        stats[i,1] = p2
        stats[i,2] = p3
        
    db_stats = pd.DataFrame(stats, index=colnames, columns=["Autofluorescence",
                                                            "Colloidal", 
                                                            "Frequent hitter"])
    db_stats.to_csv("../Results/fp_detectors/stats.csv")    
    
    
if __name__ == "__main__":
    main()