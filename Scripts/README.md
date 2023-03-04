## Folder description
Python files containing "pipeline" in their name are meant to be used as command line tools. [evals.py](evals.py), [utils.py](utils.py) and [fragment_filter.py](fragment_filter.py) contain helper functions to execute the scripts. [MVS_A.py](MVS_A.py) contains the class for running Minimum Variance Sampling Analysis for assay interfering compound retrieval.  

## Files documentation
- [cleanup_pipeline.py:](cleanup_pipeline.py) Executes the standardization and assay merging script to generate datasets ready to be analyzed. Outputs are saved in [Datasets](../Datasets).  
- [info_pipeline.py:](info_pipeline.py) Collects relevant statistics from all datasets in [Datasets](../Datasets). Outputs are saved in [Logs](../Logs)  
- [eval_pipeline.py:](eval_pipeline.py) Collects performance metrics for AIC retrieval algorithms for the benchmarks in [Datasets](../Datasets). Outputs are saved in [Results](../Results)  
- [utils.py:](utils.py) Contains various helper functions used in all pipelines (i.e. generating ECFPs etc). One key function is *process_FP*, which is used to convert raw importance scores into binary labels for *precision@90* calculation.  
- [fragment_filter.py:](fragment_filter.py) Contains the function for using structural alerts as decision rules to classify FPs and TPs. Currently supports PAINS, PAINS-A, PAINS-B, PAINS-C and NIH filters.  
- [evals.py:](evals.py) Contains the dataset evaluation functions used to collect performance metrics in [eval_pipeline.py](eval_pipeline.py).  
- [MVS_A.py:](MVS_A.py) Contains the *sample_analysis* class, which contains all relevant methods for running MVS-A outside of these scripts. See the README on the main page for a jupyter notebook example.  

## Getting help
For more technical details, check the documentation and comments provided in the respective files. For the command line tools, you can also use the "--help" flag to further clarify their usage.  
```
python3 eval_pipeline.py --help
```


