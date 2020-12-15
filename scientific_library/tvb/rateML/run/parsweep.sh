#!/usr/bin/env bash

#SBATCH -A slns
#SBATCH --partition=develgpus
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o ./output.out
#SBATCH -e ./error.er
#SBATCH --time=00:30:00
#SBATCH -J RateML

# Run the program
srun python ./parsweep.py --model mdlrun -c couplings -s speeds


