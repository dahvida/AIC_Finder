# AIC_Finder
![Python 3.7](https://img.shields.io/badge/python-3.7%20%7C%203.8-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)  
Code and scripts to employ Minimum Variance Sampling Analysis (MVS-A) to find Assay Interfering Compounds (AIC) in High Throughput Screening data.  

## Repository structure
- [Datasets:](Datasets) Contains all processed datasets used in this study as .csv files.    
- [Scripts:](Scripts) Contains all scripts and utility functions used to reproduce the results and use the tool.  
- [Results:](Results) Stores AIC retrieval performance metrics for each approach.  
- [Logs:](Logs) Stores the raw predictions from each approach, dataset statistics, metadata and a log of the changes from assay preprocessing.   
- [Misc:](Misc) Contains a csv file with the validated compounds described for the HTS case study, the dataset used to train the autofluorescence predictor and its classification performance.   

## Installation  
All necessary packages can be installed via conda from the `environment.yml` file.  
```
git clone https://github.com/dahvida/AIC_finder
conda env create --name AIC_finder --file=environment.yml
conda activate AIC_finder
```

## Tutorial
Here are the tutorials for using this tool via command line and in a jupyter notebook. An extensive description of all functions, classes and command line parameter options can be found in the respective file documentation.  
A summary of the contents of each python file is provided in the [Scripts](Scripts) folder.  

### Command line interface
`analyze_hts_pipeline.py` is a script that automates HTS dataset analysis to identify true positives, false positives and false negatives. In practice, the script loads the input dataset, computes MVS-A scores for each compound and checks for GSK and REOS structural alerts. To execute the script, the input file must fulfill three requirements:  
1. It must be a ".csv" file.  
2. It must contain a column with compound SMILES. By default, it is assumed to be named "smiles".  
3. It must contain a column defining whether compounds are active or not. This compound must identify active compounds with "1" and inactive ones with "0". By default, this column is assumed to be named "y".  
If these conditions are met, you can run the script using the following command:  
```
cd /AIC_finder/Scripts
python3 analyze_hts_pipeline.py --input_file ../Datasets/target.csv
```
In this toy example, the analysis results will be stored in two files, saved at the same path of the input file: `target_actives.csv` and `target_inactives.csv`. The first file will show all HTS hits, sorted from most to least likely to be a true positive. The second file will show all HTS non-hits, sorted from most to least likely to be a false negative. The "Alerts" column will report all GSK and REOS structural alerts triggered by a given compound. The script will check for the following alerts:  

- GSK problematic substructures (https://doi.org/10.1177/2472555218768497).  
- Molecular weight less than 200 or more than 500 Da (https://www.nature.com/articles/nrd1063#Sec3).  
- LogP less than -5.0 or more than 5.0 (https://www.nature.com/articles/nrd1063#Sec3).  
- Hydrogen bond donor count above 5.0 (https://www.nature.com/articles/nrd1063#Sec3).  
- Hydrogen bond acceptor count above 10 (https://www.nature.com/articles/nrd1063#Sec3).  
- Maximum absolute formal charge less than -2 or more than 2 (https://www.nature.com/articles/nrd1063#Sec3).  
- Number of rotatable bonds above 8 (https://www.nature.com/articles/nrd1063#Sec3).  
- Number of heavy atoms less than 8 or more than 50 (https://www.nature.com/articles/nrd1063#Sec3).  

If any GSK substructures are found, the matches are reported as SMARTS fragments, which you can visualize them with this tool: https://smarts.plus/. For the remaining filter, a flag will be added to "Alerts" if any property is outside the desired range. For example, if a molecule has a molecular weight above 500, the "Alerts" column will report "Molecular weight" and so forth.  

### Jupyter notebook
MVS-A can also be easily used from a jupyter notebook. More details can be found in the documentation of `MVS_A.py`.  
 Here is an example on how to use it:  
```python
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
true_positives, false_positives = process_ranking(y_p, vals)
```

### Reproducing results
To run the analysis on all datasets saved in the [Dataset](Dataset) folder from the command line, move to the [Scripts](Scripts) folder and simply run the command below. The results will be saved in the [Results](Results) folder, while the raw predictions will appear in [Logs/eval](Logs/eval).  
```
cd /AIC_finder/Scripts
python3 eval_pipeline.py
```
The script allows several optional arguments to customize the analysis. More details can be found in the documentation of `eval_pipeline.py`.  
Below there is an example to run it on a single dataset, only using MVS-A and structural alerts. Keep in mind that in order to run the analysis, the file must be saved in the [Datasets](Datasets) folder.  
```
python3 eval_pipeline.py --dataset kinase.csv --mvsa --fragment_filter 
```
To generate a new dataset for the analysis, you need two datatables from PubChem, one for the primary screen and one for the confirmatory screen. Save the two .csv files in the [Raw_data](Raw_data) folder with their original name and then move to the [Scripts](Scripts) folder. Then, run the following command:  
```
python3 cleanup_pipeline.py --AID_1 example_1 --AID_2 example_2 --filename my_new_dataset
```
The new file will be saved in the [Results](Results) folder as .csv, while a log of all dataset changes during cleanup will automatically appear in [Logs/cleanup](Logs/cleanup). Some example outputs are already present in that folder.  

## How to cite
Please refer to the following publication:  
https://pubs.acs.org/doi/10.1021/acscentsci.3c01517


