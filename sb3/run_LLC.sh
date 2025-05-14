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
module load python/3.11.9 scipy-stack
source ~/myenv/bin/activate

python run_LLC.py # my_script.py