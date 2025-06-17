#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=<nicolas-3.leblanc@polymtl.ca>
#SBATCH --mail-type=ALL

# cd ~/$projects/def-bonizzat/nileb3/job-scripts/
# python /home/nileb3/projects/def-bonizzat/nileb3/job-scripts/my_script.py
module purge
module load python/3.11.9 scipy-stack
source ~/myenv/bin/activate

python run_LLC_MSE_AS_seed0.py # my_script.py
# echo "all done"

# #!/bin/bash
# #SBATCH --mem=32G
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=8
# #SBATCH --time=168:0:0    
# #SBATCH --mail-user=<nicolas-3.leblanc@polymtl.ca>
# #SBATCH --mail-type=ALL

# # cd ~/$projects/def-bonizzat/nileb3/job-scripts/
# # python /home/nileb3/projects/def-bonizzat/nileb3/job-scripts/my_script.py
# module purge
# module load python/3.11.9 scipy-stack
# source ~/myenv/bin/activate

# # python run.py # my_script.py
# echo "starting..."
# python -c "import torch; print('CUDA AVAILABLE: ' + str(torch.cuda.is_available()))"
# echo "all done"


# # #!/bin/bash
# # #SBATCH --time=10:00:00	    # Requests the amount of time needed for the job.
# # #SBATCH --nodes=1	    # Number of nodes to request. (Default=1).
# # #SBATCH --ntasks=1 # Number of threads/MPI process.
# # #SBATCH --gres=gpu:1              # Number of GPU(s) per node
# # #SBATCH --cpus-per-task=6         # CPU cores/threads/tasks
# # #SBATCH --mem=16000M               # memory per node (0 means entire memory)

# # module load python/3.11
# # module load scipy-stack
# # source ~/myenv/bin/activate
# # echo "starting..."
# # python -c "import torch; print('CUDA AVAILABLE: ' + str(torch.cuda.is_available()))"
# # echo "all done"


