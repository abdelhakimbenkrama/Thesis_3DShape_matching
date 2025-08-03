#!/bin/sh
#BATCH -N 1
#SBATCH -n 14
#SBATCH -p cpu
source $HOME/testenv/bin/activate
python  main.py