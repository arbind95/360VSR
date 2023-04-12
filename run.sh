#!/bin/bash
 
# Options SBATCH :
#SBATCH --job-name=S3PO_train     # Job Name
#SBATCH --gpus=1        
#SBATCH --partition=gpu          # Name of the Slurm partition used
#SBATCH --nodelist=nodename
    

python3 main.py 

# Wait for the end of the "child" processes (Steps) before finishing the parent process (Job). Make sure you have this line, otherwise steps may exit with error.
# wait