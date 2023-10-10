import pandas as pd
import numpy as np
import os
from scipy.stats import wilcoxon

def get_means():
    algorithms = ["mvsa", "catboost", "score", "filter"]
    results_list = []
    for j in range(len(algorithms)):
        path = "../Results/" + algorithms[j] + "/"
        files = os.listdir(path)
        names = [x[7:] for x in files]
        names = [x[:-4] for x in names]
        files = [path + x for x in files]
        array = np.zeros((len(files),12))
        for i in range(len(files)):
            db = pd.read_csv(files[i])
            col_names = list(db.columns)
            arr = np.array(db)
            arr = np.mean(arr, axis=0)
            array[i,:] = arr
        array[:,4] = array[:,4] / array[:,1]
        array[:,5] = array[:,5] / array[:,2]
        db = pd.DataFrame(data=array,
                          columns=col_names,
                          index=names
                         )
        db = db.iloc[:, 3:]
        db.sort_index(inplace=True, ascending=False)
        results_list.append(db)
    return results_list

def get_stats():
    algorithms = ["mvsa", "catboost", "score", "filter"]
    path = "../Results/" + "mvsa" + "/"
    files = os.listdir(path)
    names = [x[7:] for x in files]
    names = [x[:-4] for x in names]
    
    stats1 = np.zeros((len(names), 6))
    stats2 = np.zeros((len(names), 6))
    stats3 = np.zeros((len(names), 6))
    col_names = ["fp precision",
                "tp precision",
                "fp ef10",
                "tp ef10",
                "fp bedroc",
                "tp bedroc"]
    
    for j in range(len(files)):
        arr1 = np.array(pd.read_csv("../Results/mvsa/"+files[j]))[:, 4:10]
        arr2 = np.array(pd.read_csv("../Results/catboost/"+files[j]))[:, 4:10]
        arr3 = np.array(pd.read_csv("../Results/score/"+files[j]))[:, 4:10]
        arr4 = np.array(pd.read_csv("../Results/filter/"+files[j]))[:, 4:10]
        
        for i in range(arr1.shape[1]):
            _, stats1[j,i] = wilcoxon(arr1[:,i], arr2[:,i],
                                     alternative="greater")
            _, stats2[j,i] = wilcoxon(arr1[:,i] - arr3[0,i],
                                     alternative="greater")
            _, stats3[j,i] = wilcoxon(arr1[:,i] - arr4[0,i],
                                     alternative="greater")
    db1 = pd.DataFrame(data=stats1, columns=col_names, index=names)
    db2 = pd.DataFrame(data=stats2, columns=col_names, index=names)
    db3 = pd.DataFrame(data=stats3, columns=col_names, index=names)
    db1.sort_index(inplace=True, ascending=False)
    db2.sort_index(inplace=True, ascending=False)
    db3.sort_index(inplace=True, ascending=False)

    return db1, db2, db3

def get_all():
    algorithms = ["mvsa", "catboost", "score", "filter"]
    path = "../Results/" + "mvsa" + "/"
    files = os.listdir(path)
    names = [x[7:] for x in files]
    names = [x[:-4] for x in names]
    
    all1 = np.zeros((10, 6))
    all2 = np.zeros((10, 6))
    all3 = np.zeros((10, 6))
    all4 = np.zeros((10, 6))
    col_names = ["fp precision",
                "tp precision",
                "fp ef10",
                "tp ef10",
                "fp bedroc",
                "tp bedroc"]
    
    for j in range(len(files)):
        arr1 = np.array(pd.read_csv("../Results/mvsa/"+files[j]))[:, :10]
        arr2 = np.array(pd.read_csv("../Results/catboost/"+files[j]))[:, :10]
        arr3 = np.array(pd.read_csv("../Results/score/"+files[j]))[:, :10]
        arr4 = np.array(pd.read_csv("../Results/filter/"+files[j]))[:, :10]

        arr1[:,4] = arr1[:,4] / arr1[:,1]
        arr1[:,5] = arr1[:,5] / arr1[:,2]
        arr1 = arr1[:, 4:]
        
        arr2[:,4] = arr2[:,4] / arr2[:,1]
        arr2[:,5] = arr2[:,5] / arr2[:,2]
        arr2 = arr2[:, 4:]
        
        arr3[:,4] = arr3[:,4] / arr3[:,1]
        arr3[:,5] = arr3[:,5] / arr3[:,2]
        arr3 = arr3[:, 4:]
        
        arr4[:,4] = arr4[:,4] / arr4[:,1]
        arr4[:,5] = arr4[:,5] / arr4[:,2]
        arr4 = arr4[:, 4:]
        
        all1 += arr1
        all2 += arr2
        all3 += arr3
        all4 += arr4
        
    db1 = pd.DataFrame(data = all1/17,
                      columns=col_names,
                      index=list(range(10)))
    db2 = pd.DataFrame(data = all2/17,
                      columns=col_names,
                      index=list(range(10)))
    db3 = pd.DataFrame(data = all3/17,
                      columns=col_names,
                      index=list(range(10)))
    db4 = pd.DataFrame(data = all4/17,
                      columns=col_names,
                      index=list(range(10)))

    return db1, db2, db3, db4
