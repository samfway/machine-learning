This project provides functions to explore and evaluate supervised learning techniques on microbial data.  Initially, I've uploaded two code files, machine_learning.py and run_machine_learning.py, which serves as an interface to methods in the first file.  I've also uploaded a simple test file set, which includes an OTU table (.biom) and a mapping file.  

These scripts assume that you have the following packages installed:
+ argparse, a package for parsing command line options
+ qiime, for parsing OTU and mapping files
+ scikit-learn, which includes implementations for a variety of machine learning classifiers, evaluation measures, etc. 
+ matplotlib
+ numpy

Usage: run_machine_learning.py [-h] -i BIOM_FILE [-m MAPPING_FILE]
                               [-l LABELS_FILE] [-n NUM_FOLDS]
                               [-c METADATA_CATEGORY] [-v METADATA_VALUE]

Simple wrapper for machine_learning.py.

optional arguments:
  -h, --help            show this help message and exit
  -i BIOM_FILE, --biom-file BIOM_FILE
                        Biom OTU table
  -m MAPPING_FILE, --mapping-file MAPPING_FILE
                        Mapping table
  -l LABELS_FILE, --labels-file LABELS_FILE
                        Labels file
  -n NUM_FOLDS, --num-folds NUM_FOLDS
                        Number of folds for cross-val
  -c METADATA_CATEGORY, --metadata-category METADATA_CATEGORY
                        Metadata category
  -v METADATA_VALUE, --metadata-value METADATA_VALUE
                        Metadata value
                        
You must supply either:
Mapping file + metadata category 
-OR-
A separate labels file.

Note: If a mapping file is supplied, the entire OTU matrix is considered.  If a labels file is supplied, only rows corresponding to SampleIds present in the labels file are considered.  
