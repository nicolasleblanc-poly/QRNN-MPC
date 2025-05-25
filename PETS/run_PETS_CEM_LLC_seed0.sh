#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=96:0:0    
#SBATCH --mail-user=<nicolas-3.leblanc@polymtl.ca>
#SBATCH --mail-type=ALL

# cd ~/$projects/def-bonizzat/nileb3/job-scripts/
# python /home/nileb3/projects/def-bonizzat/nileb3/job-scripts/my_script.py
module purge
module load mujoco
module load python/3.10.13 scipy-stack
#/2023a
source ~/sp_py310/bin/activate

python main_PETS_CEM_LLC_seed0.py # my_script.py