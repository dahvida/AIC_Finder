from utils import *
import pandas as pd
import numpy as np
import argparse
import os

#####################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="all")
parser.add_argument('--filename', default="dataset_report")
args = parser.parse_args()

#####################################################################

def main(
        dataset,
        filename):
    
    #adjust names depending if user passed a specific dataset
    if dataset != "all":
        dataset_names = [dataset]
    else:
        dataset_names = os.listdir("../Datasets")
        dataset_names = [x[:-4] for x in dataset_names]
    
    print("[info]: Beginning dataset analysis...")
    print(f"[info]: Running analysis on {dataset_names}")
    print(f"[info]: Report will be saved as {filename}.csv")

    #create containers and column names for the final output
    report_box = np.zeros((len(dataset_names), 15))
    column_names = [
            "Primary - compounds",
            "Primary - actives",
            "Primary - inactives",
            "Primary - active ratio",
            "Confirmatory - compounds",
            "Confirmatory - actives",
            "Confirmatory - inactives",
            "Confirmatory - active ratio",
            "Valid - compounds",
            "Valid - TP",
            "Valid - FP",
            "Valid - TP rate",
            "Valid - FP rate",
            "Follow-up rate actives primary screen",
            "Fraction primary inactives in confirmatory",
            ]
    
    #loop over all chosen datasets
    for i in range(len(dataset_names)):
        
        #load dataset and get all labels. y_p are the labels from the primary screen, y_c and
        #y_f are the TP/FP labels for the primary actives which also had a readout in the
        #confirmatory. idx is the position of these compounds in the primary screen
        name = dataset_names[i]
        print(f"[info]: Current dataset = {name}")
        db = pd.read_csv("../Datasets/" + name + ".csv")
        primary, tp, fp, idx = get_labels(db)
        
        #get statistics on primary dataset
        report_box[i, 0] = len(primary)                                 #n_compounds
        report_box[i, 1] = np.sum(primary)                              #act
        report_box[i, 2] = report_box[i,0] - report_box[i,1]            #inact
        report_box[i, 3] = report_box[i,1] / report_box[i,0]            #act / n_compounds
        
        #get statistics on entire confirmatory dataset
        db = db[~db['Confirmatory'].isnull()]
        confirmatory = np.array(db['Confirmatory'])

        report_box[i, 4] = len(confirmatory)                            #n_compounds
        report_box[i, 5] = np.sum(confirmatory)                         #act
        report_box[i, 6] = report_box[i, 4] - report_box[i,5]           #inact
        report_box[i, 7] = report_box[i, 5] / report_box[i, 4]          #act / n_compounds
        
        #get statistics on primary active compounds that were also analyzed
        #in the confirmatory (aka the ones we can determine if they
        #are FP / TP)
        report_box[i, 8] = len(tp)                                      #primary actives which were followed up in confirmatory
        report_box[i, 9] = np.sum(tp)                                   #TPs
        report_box[i, 10] = report_box[i, 8] - report_box[i, 9]         #FPs
        report_box[i, 11] = report_box[i, 9] / report_box[i, 8]         #TPs / followed up
        report_box[i, 12] = report_box[i, 10] / report_box[i, 8]        #FPs / followed up
        
        #compare total confirmatory vs partial slice above
        report_box[i, 13] = report_box[i, 8] / report_box[i, 1]         #followed up / total primary actives
        report_box[i, 14] = 1 - (report_box[i, 8] / report_box[i, 4])   #1 - (followed up / total confirmatory compounds)
    
    #store in dataframe
    db = pd.DataFrame(
                data = report_box,
                index = dataset_names,
                columns = column_names
                )
    
    #save report in log folder
    path = "../Logs/" + filename + ".csv"
    db.to_csv(path)
    print(f"[info]: Report saved at {path}")


if __name__ == "__main__":
    main(
        dataset = args.dataset,
        filename = args.filename
        )






