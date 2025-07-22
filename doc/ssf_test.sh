#!/bin/bash
#SBATCH -e slurm_%A_%a.err               # Error file location
#SBATCH -o slurm_%A_%a.out 
#SBATCH -p common     # Show who you are and get priority
#SBATCH -c 1
#SBATCH --mail-type=END        
#SBATCH --mail-user=am1155@duke.edu       # It will send you an email when the job is finishe$
#SBATCH --mem=10G                # Memory, keep it as 10G
#SBATCH --job-name=ssf_test_circ_level.out                # How is your output is called, you n$
python3 ssf_test.py 
