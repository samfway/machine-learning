This project provides functions to explore and evaluate supervised learning techniques on microbial community data.  The main file, machine_learning.py, takes as input an OTU (.biom) matrix and extracts classification labels from either a mapping file (must supply metadata category) or from a separate labels file.  

The following packages are required by the project:
+ qiime
+ scikit-learn (requires numpy+scipy)
+ matplotlib
+ argparse

If you're reading this, you likely have qiime and matplotlib installed already.  Installing scikit-learn is really pretty simple.  [Click here](http://scikit-learn.org/stable/install.html) to see their installation instructions.  
