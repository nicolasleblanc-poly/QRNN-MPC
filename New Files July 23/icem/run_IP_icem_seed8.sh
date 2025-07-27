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
# module load python/3.10.13 scipy-stack
# source ~/py310/bin/activate
module load python/3.11.9 scipy-stack
source ~/myenv/bin/activate

python InvertedPendulum_approximate_continuous_seed8.py # my_script.py

# #!/bin/bash
# #SBATCH --mem=32G
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=8
# #SBATCH --time=96:0:0    
# #SBATCH --mail-user=<nicolas-3.leblanc@polymtl.ca>
# #SBATCH --mail-type=ALL

# #SBATCH --gres=gpu:1
# #SBATCH --cpus-per-task=4

# # cd ~/$projects/def-bonizzat/nileb3/job-scripts/
# # python /home/nileb3/projects/def-bonizzat/nileb3/job-scripts/my_script.py
# module purge
# module load mujoco
# module load python/3.10.13 scipy-stack
# source ~/py310/bin/activate

# python MountainCar_approximate_continuous.py # my_script.py

# #!/bin/bash
# #SBATCH --job-name=mountaincar-gpu
# #SBATCH --mem=32G
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --gres=gpu:1
# #SBATCH --time=96:00:00    
# #SBATCH --mail-user=nicolas-3.leblanc@polymtl.ca
# #SBATCH --mail-type=ALL
# #SBATCH --output=%x-%j.out

# # Load modules
# module purge
# module load python/3.11.9 scipy-stack
# module load cuda/12.1
# module load mujoco
# module load pytorch/2.1  # Optional if using torch with GPU

# # Activate environment
# source ~/myenv/bin/activate

# # Run your Python script
# python MountainCar_approximate_continuous.py


