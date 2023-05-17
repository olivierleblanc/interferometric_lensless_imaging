#!/bin/bash
#
#SBATCH --job-name=phase_transition
#SBATCH --output=phase_transition_output.txt
#
#SBATCH --ntasks=16
#SBATCH --time=02:30:00
#SBATCH --mem-per-cpu=1000

mpirun -n 16 python phase_transition_diagrams.py