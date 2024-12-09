#!/bin/bash
#SBATCH --output=output_%A_%a.out
#SBATCH --error=error_%A_%a.err
#SBATCH --array=1-9
#SBATCH --time=01:00:00  # Time limit
#SBATCH --mem=4G

GRAPH_NAME=$1
GRAPH_DIR="$DATA_DIR/$GRAPH_NAME/"

# Source profile
source ~/.profile_minigraphs

# Calculate miniaturization size
FRAC=$(echo "scale=3; $SLURM_ARRAY_TASK_ID * 0.1" | bc)




