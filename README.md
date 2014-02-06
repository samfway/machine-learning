This project provides functions to explore and evaluate supervised learning techniques on microbial community data.  The main file, machine_learning.py, takes as input an OTU (.biom) matrix and extracts classification labels from either a mapping file (must supply metadata category) or from a separate labels file.  

The following packages are required by the project:
+ qiime
+ scikit-learn (requires numpy+scipy)
+ matplotlib
+ argparse

