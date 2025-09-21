#!/bin/bash
# This script submits all generated jobs to SLURM
for script in mu_transfer_*.sh; do
  sbatch "$script"
done

echo 'Submitted all generated scripts to SLURM'
