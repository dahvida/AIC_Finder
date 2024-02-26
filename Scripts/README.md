## Folder description
Code used to generate the results described in the study. Files containing "pipeline" in their name are meant to be used as command line tools.  

## Files documentation
- [cleanup_pipeline.py:](cleanup_pipeline.py) Executes the standardization and assay merging script to generate datasets ready to be analyzed. Outputs are saved in [Datasets](../Datasets), logs are saved in [Logs/cleanup](../Logs/cleanup).  

- [info_pipeline.py:](info_pipeline.py) Collects relevant statistics from all datasets in [Datasets](../Datasets). Outputs are saved in [Logs](../Logs)  

- [eval_pipeline.py:](eval_pipeline.py) Collects performance metrics for AIC retrieval algorithms for the benchmarks in [Datasets](../Datasets). Performance is evaluated in terms of Precision (K=10%), Enrichment Factor (K=10%), BEDROC (alpha=20), calculation time (s) and Murko scaffold diversity. Outputs are saved in [Results](../Results), raw predictions are saved in [/Logs/eval](../Logs/eval). Current AIC retrieval algorithms include MVS-A, CatBoost Object Importance, structural alerts (GSK and REOS filters), Isolation Forest, Variational Autoencoder and sorting by primary assay readout (top 10% most active compounds in primary screen are flagged as TPs).  

- [opt_pipeline.py:](opt_pipeline.py) Script to generate the autofluorescence predictor with Bayesian hyperparameter optimization. The 10-fold cross-validation performance is saved in [Misc](../Misc).

- [predictors_pipeline.py:](predictors_pipeline.py) Computes performance metrics from the predictions of Hit Dexter, SCAM Detective and the autofluorescence predictor. The predictions are stored in [Logs/eval](../Logs/eval), while the output is saved in [Results/fp_detectors](../Results/fp_detectors).  

- [summarize_pipeline.py:](summarize_pipeline.py) Averages performance metrics for each method across replicates and generates final summary files for performance metrics and statistical tests. The output is saved in [Results/summary](../Results/summary).  

- [case_study_pipeline.py:](case_study_pipeline.py) Script to reproduce the case study analysis reported in the publication.  

- [evals.py:](evals.py) Contains the dataset evaluation functions used to collect performance metrics in [eval_pipeline.py:](eval_pipeline.py).  

- [utils.py:](utils.py) Contains various helper functions used in all pipelines (i.e. generating ECFPs etc). One key function is `process_ranking`, which is used to convert raw importance scores into binary labels for *precision@10* calculation.  

- [fragment_filter.py:](fragment_filter.py) Contains the function for using structural alerts as decision rules to classify FPs and TPs. Currently supports GSK and REOS filters.  

- [filters.cvs:](filters.csv) SMARTS definition of the GSK structural alerts used in `fragment_filter.py`.  

- [vae.py:](vae.py) PyTorch implementation of a SMILES-based Variational Autoencoder for anomaly detection. The implementation and hyperparameters are based on https://github.com/aspuru-guzik-group/chemical_vae.  

- [MVS_A.py:](MVS_A.py) Contains the `sample_analysis` class, which contains all relevant methods for running MVS-A outside of these scripts. See the README on the main page for a jupyter notebook example.  

## Getting help
For more technical details, check the documentation and comments provided in the respective files. For the command line tools, you can also use the `--help` flag to further clarify their usage. Here is an example:  
```
python3 eval_pipeline.py --help
```


