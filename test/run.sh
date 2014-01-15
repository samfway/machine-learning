#!/bin/bash

DIR="$HOME/machine-learning"

echo "cd $DIR ; python run_machine_learning.py -i test/otu.biom -m test/mapping.txt -c HOST_INDIVIDUAL -s test/sklearn_config.txt -f -o ML-report.txt" | qsub -k oe -N ML


