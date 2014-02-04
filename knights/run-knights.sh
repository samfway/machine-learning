#!/bin/bash

DIR="$HOME/machine-learning"

echo "cd $DIR ; python run_machine_learning.py -i knights/FS_otu.biom -m knights/FS_mapping.txt -c Label -s knights/sklearn_config.txt -o knights/ML-report-FS.txt" | qsub -k oe -N ML-FS -l pvmem=4gb -q memroute
echo "cd $DIR ; python run_machine_learning.py -i knights/FSH_otu.biom -m knights/FSH_mapping.txt -c Label -s knights/sklearn_config.txt -o knights/ML-report-FSH.txt" | qsub -k oe -N ML-FSH -l pvmem=4gb -q memroute
echo "cd $DIR ; python run_machine_learning.py -i knights/CBH_otu.biom -m knights/CBH_mapping.txt -c Label -s knights/sklearn_config.txt -o knights/ML-report-CBH.txt" | qsub -k oe -N ML-CBH -l pvmem=4gb -q memroute
echo "cd $DIR ; python run_machine_learning.py -i knights/CS_otu.biom -m knights/CS_mapping.txt -c Label -s knights/sklearn_config.txt -o knights/ML-report-CS.txt" | qsub -k oe -N ML-CS -l pvmem=4gb -q memroute
echo "cd $DIR ; python run_machine_learning.py -i knights/CSS_otu.biom -m knights/CSS_mapping.txt -c Label -s knights/sklearn_config.txt -o knights/ML-report-CSS.txt" | qsub -k oe -N ML-CSS -l pvmem=4gb -q memroute
