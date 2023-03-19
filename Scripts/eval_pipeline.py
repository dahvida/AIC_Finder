"""Performance analysis script.

Pipeline to run the AIC analysis using MVS-A, CatBoost and structural alerts. Provides
options to control the target dataset, which algorithms to employ for the run, 
number of iterations and whether to log raw predictions. The target dataset(s) 
must be stored in the ../Datasets folder. To log predictions, all algorithms
must be enabled for the run.
"""

from utils import *
from evals import *
import numpy as np
from rdkit import Chem
import pandas as pd
import argparse
import os
import time

###############################################################################

parser = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', default="all",
                    help="Which dataset from ../Datasets to use for the analysis, options: [all, specific_name]")

parser.add_argument('--mvs_a', default="yes",
                    help="Whether to use MVS-A for the run, options: [yes, no]")

parser.add_argument('--catboost', default="yes",
                    help="Whether to use CatBoost for the run, options: [yes, no]")

parser.add_argument('--score', default="yes",
                    help="Whether to use assay readouts for the run, options: [yes, no]")

parser.add_argument('--fragment_filter', default="yes",
                    help="Whether to use a fragment filter for the run, options: [yes, no]")

parser.add_argument('--filter_type', default="PAINS",
                    help="Which fragment set to use for the run, options: [PAINS, PAINS_A, PAINS_B, PAINS_C, NIH]")

parser.add_argument('--replicates', default=5, type=int,
                    help="How many replicates to use for MVS-A and CatBoost")

parser.add_argument('--filename', default="output",
                    help="Name to use when saving performance results")

parser.add_argument('--log_predictions', default="yes",
                    help="Whether to log raw predictions, only works if all algorithms are enabled, options: [yes, no]")

args = parser.parse_args()

###############################################################################

def main(dataset,
         mvs_a,
         catboost,
         score,
         fragment_filter,
         filter_type,
         replicates,
         filename,
         log_predictions):
    
    #turn log_predictions into boolean, then check if it is compatible with other params
    if log_predictions == "no":
        log_predictions = False
    else:
        if mvs_a == catboost == score == fragment_filter == "yes":
            log_predictions = True
        else:
            print("[eval]: log_predictions works only if all AIC detection algorithms are enabled, setting it to False")
            log_predictions = False
    
    #adjust dataset_names depending on if the user chose a single dataset
    if dataset != "all":
        dataset_names = [dataset]
    else:
        dataset_names = os.listdir("../Datasets")    
        dataset_names = [x[:-4] for x in dataset_names]
        
    #create boxes to store results
    mvs_a_box = np.zeros((len(dataset_names), 12))
    catboost_box = np.zeros((len(dataset_names), 12))
    filter_box = np.zeros((len(dataset_names), 12))
    score_box = np.zeros((len(dataset_names), 12))

    #print run info
    print("[eval]: Beginning eval run...")
    print("[eval]: Run parameters:")
    print(f"        dataset: {dataset_names}")
    print(f"        algorithms: mvs_a = {mvs_a}, catboost = {catboost}, score = {score}, fragment_filter = {fragment_filter} with {filter_type}")
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
        
        if score == "yes":
            print("[eval]: Running score analysis...")
            temp, filter_log = run_score(db, mols, idx, y_p,
                                         y_f, y_c, log_predictions)
            score_box = store_row(score_box, temp, fp_rate, tp_rate, i)     
            print("[eval]: Score analysis finished")
        
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
        [mvs_a_box, catboost_box, score_box, filter_box],
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
         score = args.score.lower(),
         fragment_filter = args.fragment_filter.lower(),
         filter_type = args.filter_type.upper(),
         replicates = args.replicates,
         filename = args.filename,
         log_predictions = args.log_predictions)






