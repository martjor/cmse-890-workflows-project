#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --mem=500M

conda init
conda activate minigraphs_env

snakemake -s Snakefile_grt --executor slurm --workflow-profile slurm --rerun-incomplete