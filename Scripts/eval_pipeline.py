"""Performance analysis script.

Pipeline to run the false positive and true positive detection analysis. Provides
options to control the target dataset(s), which algorithms to employ for the run, 
number of iterations and molecular features. The target dataset(s) must be stored
in the ../Datasets folder, as generated by the cleanup_pipeline.py script. The 
results for each algorithm will be saved in an independent .csv file, 
i.e. "mvsa_output.csv" 

Steps:
    1. Load dataset and compute chosen molecular feature set
    2. Create labels for primary and confirmatory data
    3. Run MVS-A analysis and store results
    4. Run CatBoost analysis and store results
    5. Run score analysis and store results
    6. Run structural alerts analysis and store results
    7. Run Isolation Forest analysis and store results
    8. Run VAE anomaly detection analysis and store results
    9. Log predictions and save

For further details regarding how each AIC analysis is run, check evals.py
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

parser.add_argument('--mvs_a', action="store_true",
                    help="Whether to use MVS-A for the run")

parser.add_argument('--catboost', action="store_true",
                    help="Whether to use CatBoost for the run")

parser.add_argument('--score', action="store_true",
                    help="Whether to use assay readouts for the run")

parser.add_argument('--fragment_filter', action="store_true",
                    help="Whether to use a fragment filter for the run")

parser.add_argument('--isoforest', action="store_true",
                    help="Whether to use Isolation Forest for the run")

parser.add_argument('--vae', action="store_true",
                    help="Whether to use VAE for the run")

parser.add_argument('--replicates', default=10, type=int,
                    help="How many replicates to use for MVS-A and CatBoost")

parser.add_argument('--feature_type', default="ECFP",
                    help="Which molecular representation to use for featurization, options: [ECFP, MACCS, 2D, none]")

parser.add_argument('--filename', default="output",
                    help="Name to use when saving performance results")

args = parser.parse_args()

###############################################################################

def main(dataset,
         mvs_a,
         catboost,
         score,
         fragment_filter,
         isoforest,
         vae,
         replicates,
         feature_type,
         filename):
    
    #adjust dataset_names depending on if the user chose a single dataset
    if dataset != "all":
        dataset_names = [dataset]
    else:
        dataset_names = os.listdir("../Datasets")    
        dataset_names = [x[:-4] for x in dataset_names]

    #make directories if they do not exist yet
    if not os.path.exists("../Results/mvsa"):
        os.makedirs("../Results/mvsa")
    if not os.path.exists("../Results/catboost"):
        os.makedirs("../Results/catboost")
    if not os.path.exists("../Results/score"):
        os.makedirs("../Results/score")
    if not os.path.exists("../Results/filter"):
        os.makedirs("../Results/filter")
    if not os.path.exists("../Results/isoforest"):
        os.makedirs("../Results/isoforest")
    if not os.path.exists("../Results/vae"):
        os.makedirs("../Results/vae")
       
    if not os.path.exists("../Logs/eval/mvsa"):
        os.makedirs("../Logs/eval/mvsa")
    if not os.path.exists("../Logs/eval/catboost"):
        os.makedirs("../Logs/eval/catboost")
    if not os.path.exists("../Logs/eval/score"):
        os.makedirs("../Logs/eval/score")
    if not os.path.exists("../Logs/eval/filter"):
        os.makedirs("../Logs/eval/filter")
    if not os.path.exists("../Logs/eval/isoforest"):
        os.makedirs("../Logs/eval/isoforest")
    if not os.path.exists("../Logs/eval/vae"):
        os.makedirs("../Logs/eval/vae")
        
    #print run info
    print("[eval]: Beginning eval run...")
    print("[eval]: Run parameters:")
    print(f"        dataset: {dataset_names}")
    print(f"        mvs_a: {mvs_a}")
    print(f"        catboost: {catboost}")
    print(f"        score: {score}")
    print(f"        fragment_filter: {fragment_filter}")
    print(f"        isoforest: {isoforest}")
    print(f"        vae: {vae}")
    print(f"        replicates: {replicates}")
    print(f"        feature type: {feature_type}")
    print(f"        file identifier: {filename}")
    
    #loop analysis over all datasets
    for i in range(len(dataset_names)):

        #load i-th dataset, get mols
        print("----------------------")
        name = dataset_names[i]
        print(f"[eval]: Processing dataset: {name}")
        db = pd.read_csv("../Datasets/" + name + ".csv")
        mols = list(db["SMILES"])
        mols = [Chem.MolFromSmiles(x) for x in mols]

        #compute representation
        if feature_type == "ECFP":
            feats = get_ECFP(mols)
        if feature_type == "MACCS":
            feats = get_MACCS(mols)
        if feature_type == "2D":
            feats = get_2D(mols)

        #get labels for the analysis and get random-guess probabilities for
        #FP and TP. y_p are the labels from the primary screen, y_c and
        #y_f are the TP/FP labels for the primary actives which also had a
        #readout in the confirmatory. idx is the position of these compounds
        #in the primary screen
        y_p, y_c, y_f, idx = get_labels(db)
        fp_rate = np.sum(y_f) / len(y_f)
        tp_rate = 1 - fp_rate
        
        #depending on user options, run analysis and store results
        if mvs_a is True:
            print("[eval]: Running MVS-A analysis...")
            output, log = run_mvsa(mols, feats, y_p, y_f, y_c, idx, replicates)
            save_output(output,
                        fp_rate,
                        tp_rate,
                        name, "mvsa", filename)
            save_log(log, name, "mvsa")
            print("[eval]: MVS-A analysis finished")

        if catboost is True:
            print("[eval]: Running CatBoost analysis...")
            output, log = run_catboost(mols, feats, y_p, y_f, y_c, idx, replicates)
            save_output(output,
                        fp_rate,
                        tp_rate,
                        name, "catboost", filename)
            save_log(log, name, "catboost")
            print("[eval]: CatBoost analysis finished")            
        
        if score is True:
            print("[eval]: Running score analysis...")
            output, log = run_score(db, mols, idx, y_p, y_f, y_c)
            save_output(output,
                        fp_rate,
                        tp_rate,
                        name, "score", filename)
            save_log(log, name, "score")
            print("[eval]: Score analysis finished")
        
        if fragment_filter is True:
            print("[eval]: Running filter analysis...")
            output, log = run_filter(mols, idx, y_p, y_f, y_c)
            save_output(output,
                        fp_rate,
                        tp_rate,
                        name, "filter", filename)
            save_log(log, name, "filter")
            print("[eval]: Filter analysis finished")
            
        if isoforest is True:
            print("[eval]: Running Isolation Forest analysis...")
            output, log = run_isoforest(mols, feats, y_p, y_f, y_c, idx, replicates)
            save_output(output,
                        fp_rate,
                        tp_rate,
                        name, "isoforest", filename)
            save_log(log, name, "isoforest")
            print("[eval]: Isoforest analysis finished")

        if vae is True:
            print("[eval]: Running VAE analysis...")
            output, log = run_vae(mols, y_p, y_f, y_c, idx, replicates)
            save_output(output,
                        fp_rate,
                        tp_rate,
                        name, "vae", filename)
            save_log(log, name, "vae")
            print("[eval]: VAE analysis finished")
    

if __name__ == "__main__":
    main(dataset = args.dataset,
         mvs_a = args.mvs_a,
         catboost = args.catboost,
         score = args.score,
         fragment_filter = args.fragment_filter,
         isoforest = args.isoforest,
         vae = args.vae,
         replicates = args.replicates,
         feature_type = args.feature_type.upper(),
         filename = args.filename,
         )
