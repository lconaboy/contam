#!/bin/bash -l
#
#SBATCH --job-name=
#SBATCH --account=
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --partition=
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=

# Need to load python3 and pynbody (+ dependencies)

python contam.py
