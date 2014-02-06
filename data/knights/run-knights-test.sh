#!/bin/bash

DIR="$HOME/MachineLearning"

echo "cd $DIR ; python machine_learning.py -i knights/FS_otu.biom -m knights/FS_mapping.txt -c Label -s knights/sklearn_config.txt -o knights/ML-report-FS-test.txt" | qsub -k oe -N ML-test -l pvmem=4gb -q memroute
