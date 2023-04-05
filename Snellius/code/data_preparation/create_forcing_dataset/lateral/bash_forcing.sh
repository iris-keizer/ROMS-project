#!/bin/bash
#Set job requirements
#SBATCH -n 128
#SBATCH -t 00:40:00
#SBATCH -p thin

#Loading modules
module load 2021
module load Anaconda3/2021.05


source activate esmf


python create_forcing_fix.py  
