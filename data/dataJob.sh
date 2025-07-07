#!/bin/sh
#
#SBATCH -p rpi,priority
#SBATCH -A rpi
#SBATCH --job-name=dataCreation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=4000
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

conda run -n ft python createFinDataset.py

