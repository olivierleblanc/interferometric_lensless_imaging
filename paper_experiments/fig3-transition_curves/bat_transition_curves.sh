#!/bin/bash
#
#SBATCH --job-name=transition_curves
#SBATCH --output=transition_curves_output.txt
#
#SBATCH --ntasks=10
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=1000

mpirun -n 10 python transition_curves.py