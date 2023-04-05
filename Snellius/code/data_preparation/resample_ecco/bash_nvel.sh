#!/bin/bash
#Set job requirements
#SBATCH -n 128
#SBATCH -t 00:30:00
#SBATCH -p thin

#Loading modules
module load 2021
module load Anaconda3/2021.05


source activate ecco


python ecco_prep_nvel.py  
