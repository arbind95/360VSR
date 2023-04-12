#!/bin/bash
 
# Options SBATCH :
#SBATCH --job-name=S3PO_test      # Job Name
#SBATCH --gpus=1        
#SBATCH --partition=gpu          # Name of the Slurm partition used
#SBATCH --nodelist=nodename   

python3 test.py
