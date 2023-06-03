#!/bin/bash

#SBATCH -A snic2022-22-631
#SBATCH -p core
#SBATCH -n 6
#SBATCH -t 02:00:00 
#SBATCH --error=.log/create.middle.%J.err 
#SBATCH --output=.log/create.middle.%J.out
#SBATCH -C usage_mail
#SBATCH --mail-type=ALL

# python sample_graph.py
python create_middle_indices.py