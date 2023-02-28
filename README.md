# AIC_Finder
![Python 3.7](https://img.shields.io/badge/python-3.7%20%7C%203.8-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)  
Code and scripts to employ Minimum Variance Sampling Analysis (MVS-A) to find Assay Interfering Compounds (AIC) in High Throughput Screening data.  

## Repository structure
- **Datasets:** Contains all processed datasets used in this study as .csv files  
- **Raw_data:** Contains the raw HTS assays used to generate the processed datasets for the analysis  
- **Scripts:** Contains all scripts and utility functions used to reproduce the results and use the tool  
- **Results:** Contains the performance metrics calculated by the *eval_pipeline.py* script  
- **Logs:** Contains the raw predictions from each approach, dataset statistics and their change as a result of preprocessing  

## Installation  
All necessary packages can be installed via conda from the environment.yml file.  
```
git clone https://github.com/dahvida/AIC_finder
conda env create --name AIC_finder --file=environments.yml
conda activate AIC_finder
```

### Scripts tutorial
To run the analysis on all datasets saved in the */Datasets* folder from the command line, move to the */Scripts* folder and simply run the command below. The results will be saved in the */Results* folder, while the raw predictions will appear in */Logs/eval*.  
```
cd /AIC_finder/Scripts
python3 eval_pipeline.py
```
The script allows several optional arguments to customize the analysis. More details can be found in the documentation of *eval_pipeline.py*.  
Below there is an example to run it on a single dataset, only using MVS-A and PAINS_A, without logging. Keep in mind that in order to run the analysis, the file must be saved in the */Datasets* folder  
```
python3 eval_pipeline.py --dataset kinase.csv --catboost no --filter_type PAINS_A --run_logging no
```

### Jupyter notebook tutorial
MVS-A can also be easily used from a jupyter notebook. More details can be found in the documentation of *MVS_A.py*.  
 Here is an example on how to use it:  
```
#import packages
from utils import process_FP
from MVS_A import *

#create MVS_A instance to run the analysis
AIC_finder = sample_analysis(
				x = x,			#input data i.e. ECFP
                      		y = y_p, 		#binary labels from primary screen
                      		params = "default",	#could pass a dict with custom params (not recommended)
                      		verbose = True,	#option to print updates on calculation status
                      		seed = 0		#random seed for reproducibility
                      		)

#train base lightgbm model
AIC_finder.get_model()

#get raw sample importances
vals = AIC_finder.get_importance()

#convert raw scores into binary labels
true_positives, false_positives = process_FP(y_p, vals)
```

## How to cite
Link to publication  


