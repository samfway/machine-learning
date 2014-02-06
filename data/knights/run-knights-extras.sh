#!/bin/bash

DIR="$HOME/machine-learning"

echo "cd $DIR ; python run_machine_learning.py -i knights/FS_otu.biom -m knights/FS_mapping.txt -c Label -s knights/sklearn_config-extras.txt -o knights/ML-report-FS-extras.txt" | qsub -k oe -N ML-FS-e -l pvmem=4gb -q memroute
echo "cd $DIR ; python run_machine_learning.py -i knights/FSH_otu.biom -m knights/FSH_mapping.txt -c Label -s knights/sklearn_config-extras.txt -o knights/ML-report-FSH-extras.txt" | qsub -k oe -N ML-FSH-e -l pvmem=4gb -q memroute
echo "cd $DIR ; python run_machine_learning.py -i knights/CBH_otu.biom -m knights/CBH_mapping.txt -c Label -s knights/sklearn_config-extras.txt -o knights/ML-report-CBH-extras.txt" | qsub -k oe -N ML-CBH-e -l pvmem=4gb -q memroute
echo "cd $DIR ; python run_machine_learning.py -i knights/CS_otu.biom -m knights/CS_mapping.txt -c Label -s knights/sklearn_config-extras.txt -o knights/ML-report-CS-extras.txt" | qsub -k oe -N ML-CS-e -l pvmem=4gb -q memroute
echo "cd $DIR ; python run_machine_learning.py -i knights/CSS_otu.biom -m knights/CSS_mapping.txt -c Label -s knights/sklearn_config-extras.txt -o knights/ML-report-CSS-extras.txt" | qsub -k oe -N ML-CSS-e -l pvmem=4gb -q memroute
