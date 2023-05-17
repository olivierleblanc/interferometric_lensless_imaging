#!/bin/bash
#
#SBATCH --job-name=experimental_rec
#SBATCH --output=experimental_rec_output.txt
#
#SBATCH --ntasks=9
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=10000

mpirun -n 9 python experimental_rec.py