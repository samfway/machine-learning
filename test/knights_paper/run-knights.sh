#!/bin/bash

DIR="$HOME/machine-learning"

echo "cd $DIR ; python run_machine_learning.py -i test/knights_paper/FS_otu.biom -m test/knights_paper/FS_mapping.txt -c Label -s test/knights_paper/sklearn_config.txt -o test/knights_paper/ML-report-FS.txt" | qsub -k oe -N ML-FS -l pvmem=4gb -q memroute
echo "cd $DIR ; python run_machine_learning.py -i test/knights_paper/FSH_otu.biom -m test/knights_paper/FSH_mapping.txt -c Label -s test/knights_paper/sklearn_config.txt -o test/knights_paper/ML-report-FSH.txt" | qsub -k oe -N ML-FSH -l pvmem=4gb -q memroute
echo "cd $DIR ; python run_machine_learning.py -i test/knights_paper/CBH_otu.biom -m test/knights_paper/CBH_mapping.txt -c Label -s test/knights_paper/sklearn_config.txt -o test/knights_paper/ML-report-CBH.txt" | qsub -k oe -N ML-CBH -l pvmem=4gb -q memroute
echo "cd $DIR ; python run_machine_learning.py -i test/knights_paper/CS_otu.biom -m test/knights_paper/CS_mapping.txt -c Label -s test/knights_paper/sklearn_config.txt -o test/knights_paper/ML-report-CS.txt" | qsub -k oe -N ML-CS -l pvmem=4gb -q memroute
echo "cd $DIR ; python run_machine_learning.py -i test/knights_paper/CSS_otu.biom -m test/knights_paper/CSS_mapping.txt -c Label -s test/knights_paper/sklearn_config.txt -o test/knights_paper/ML-report-CSS.txt" | qsub -k oe -N ML-CSS -l pvmem=4gb -q memroute
