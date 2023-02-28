from utils import *
from evals import *
import numpy as np
from rdkit import Chem
import pandas as pd
import argparse
import os
import time

###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="all")		
parser.add_argument('--mvs_a', default="yes")
parser.add_argument('--catboost', default="yes")
parser.add_argument('--fragment_filter', default="yes")
parser.add_argument('--filter_type', default="PAINS")
parser.add_argument('--replicates', default=5, type=int)
parser.add_argument('--filename', default="output")
parser.add_argument('--log_predictions', default="no")
args = parser.parse_args()

###############################################################################

def main(dataset,
         mvs_a,
         catboost,
         fragment_filter,
         filter_type,
         replicates,
         filename,
         log_predictions):
    
    #turn log_predictions into boolean, then check if it is compatible with other params
    if log_predictions == "no":
        log_predictions = False
    else:
        log_predictions = True
        assert mvs_a == catboost == fragment_filter == "yes", "[eval]: To log predictions, all algorithms must be enabled for the run"
    
    #adjust dataset_names depending on if the user chose a single dataset
    if dataset != "all":
        dataset_names = [dataset]
    else:
        dataset_names = os.listdir("../Datasets")    
        dataset_names = [x[:-4] for x in dataset_names]
        
    #create boxes to store results
    mvs_a_box = np.zeros((len(dataset_names), 10))
    catboost_box = np.zeros((len(dataset_names), 10))
    filter_box = np.zeros((len(dataset_names), 10))
    
    #print run info
    print("[eval]: Beginning eval run...")
    print("[eval]: Run parameters:")
    print(f"        dataset: {dataset_names}")
    print(f"        algorithms: mvs_a = {mvs_a}, catboost = {catboost}, fragment_filter = {fragment_filter} with {filter_type}")
    print(f"        replicates: {replicates}")
    print(f"        prediction logging: {log_predictions}")
    print(f"        file identifier: {filename}")
    
    #loop analysis over all datasets
    for i in range(len(dataset_names)):

        #load i-th dataset, get mols and then ECFPs
        print("----------------------")
        name = dataset_names[i]
        print(f"[eval]: Processing dataset: {name}")
        db = pd.read_csv("../Datasets/" + name + ".csv")
        mols = list(db["SMILES"])
        mols = [Chem.MolFromSmiles(x) for x in mols]
        ecfp = get_ECFP(mols)

        #get labels for the analysis and get random-guess probabilities for
        #FP and TP. y_p are the labels from the primary screen, y_c and
        #y_f are the TP/FP labels for the primary actives which also had a
        #readout in the confirmatory. idx is the position of these compounds
        #in the primary screen
        y_p, y_c, y_f, idx = get_labels(db)
        fp_rate = np.sum(y_f) / len(y_f)
        tp_rate = 1 - fp_rate
        
        #depending on user options, run analysis and store results
        if mvs_a == "yes":
            print("[eval]: Running MVS-A analysis...")
            temp, mvs_log = run_mvsa(mols, ecfp, y_p, y_f, y_c, idx, replicates,
                                    log_predictions)
            mvs_a_box = store_row(mvs_a_box, temp, fp_rate, tp_rate, i)
            print("[eval]: MVS-A analysis finished")

        if catboost == "yes":
            print("[eval]: Running CatBoost analysis...")
            temp, cb_log = run_catboost(mols, ecfp, y_p, y_f, y_c, idx, replicates,
                                        log_predictions)
            catboost_box = store_row(catboost_box, temp, fp_rate, tp_rate, i)
            print("[eval]: CatBoost analysis finished")            
        
        if fragment_filter == "yes":
            print("[eval]: Running filter analysis...")
            temp, filter_log = run_filter(mols, idx, filter_type, y_f, y_c,
                                        log_predictions)
            filter_box = store_row(filter_box, temp, fp_rate, tp_rate, i)
            print("[eval]: Filter analysis finished")
        
        #optionally store logs (raw predictions for all compounds)
        if log_predictions is True:
            logs = pd.merge(mvs_log, cb_log, how = "inner")
            logs = pd.merge(logs, filter_log, how = "inner")
            logpath = "../Logs/eval/" + filename + "_" + name + ".csv"
            logs.to_csv(logpath)
            print(f"[eval]: Log saved at {logpath}")
            
    #save results for all algorithms as .csv files
    save_results(
        [mvs_a_box, catboost_box, filter_box],
        dataset_names,
        filename,
        filter_type
        )
    print("----------------------")
    print("[eval]: Results saved in ../Results/*_" + filename + ".csv")
    

if __name__ == "__main__":
    main(dataset = args.dataset,
         mvs_a = args.mvs_a.lower(),
         catboost = args.catboost.lower(),
         fragment_filter = args.fragment_filter.lower(),
         filter_type = args.filter_type.upper(),
         replicates = args.replicates,
         filename = args.filename,
         log_predictions = args.log_predictions)

















