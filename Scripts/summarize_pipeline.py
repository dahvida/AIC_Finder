import pandas as pd
import numpy as np
import os
from scipy.stats import wilcoxon
import os

###############################################################################

def get_means():
    algorithms = ["mvsa", "catboost", "score", "filter", "isoforest", "vae"]
    for j in range(len(algorithms)):
        path = "../Results/" + algorithms[j] + "/"
        files = os.listdir(path)
        names = [x[7:] for x in files]
        names = [x[:-4] for x in names]
        files = [path + x for x in files]
        array = np.zeros((len(files),11))
        for i in range(len(files)):
            db = pd.read_csv(files[i], index_col=0)
            col_names = list(db.columns)
            arr = np.array(db)
            arr = np.mean(arr, axis=0)
            array[i,:] = arr
        array[:,3] = array[:,3] - array[:,0]
        array[:,4] = array[:,4] - array[:,1]
        db = pd.DataFrame(data=array,
                          columns=col_names,
                          index=names
                         )
        db.sort_index(inplace=True, ascending=True)
        path = "../Results/summary/" + algorithms[j] + ".csv"
        db.to_csv(path)

#-----------------------------------------------------------------------------#

def get_stats():
    algorithms = ["mvsa", "catboost", "score", "filter", "isoforest", "vae"]
    db_list = []
    for j in range(len(algorithms)):
        path = "../Results/summary/" + algorithms[j] + ".csv"
        db_j = pd.read_csv(path, index_col=0)
        db_list.append(db_j)
    
    col_names = list(db_j.columns)[3:]
    baselines = db_list[1:]
    array = np.zeros((len(col_names), len(baselines)))
    
    for i in range(len(col_names)):
        arr1 = np.array(db_list[0][col_names[i]])
        for k in range(len(baselines)):
            arr2 = np.array(baselines[k][col_names[i]])
            _, p = wilcoxon(arr1, arr2, alternative="greater")
            array[i,k] = p * len(baselines)
    
    db = pd.DataFrame(data=array,
                      columns=algorithms[1:],
                      index=col_names
                      )
    db.to_csv("../Results/summary/stats.csv")

#-----------------------------------------------------------------------------#

def main():
    path = "../Results/summary/"
    if not os.path.exists(path):
        os.makedirs(path)
    
    get_means()
    get_stats()

    print("[summarize]: Data summarized at", path)

if __name__ == "__main__":
    main()
