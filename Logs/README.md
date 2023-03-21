## Folder description
Log files for the main command-line tools in the [Scripts](../Scripts) folder and general dataset information.

## Files documentation
- [cleanup:](cleanup) Contains .txt files describing dataset changes before and after executing `cleanup_pipeline.py` on a primary-confirmatory HTS pair. Each file details the change in total number of compounds as well as active ones, both for the primary and confirmatory assays. Also, the file states which AIDs were used to generate the preprocessed .csv file.  
- [eval:](eval) Collects raw binary predictions from `eval_pipeline.py` for each dataset used in the analysis. Predictions will be logged only if all AIC retrieval methods were employed.    
- [dataset_metadata.csv:](dataset_metadata.csv) Stores metadata information for each preprocessed dataset, i.e. which were the parent AIDs, publication source and so forth.  
- [dataset_statistics.csv:](dataset_statistics.csv) Stores dataset statistics such as compound number, imbalance rate and so forth for each dataset.  


