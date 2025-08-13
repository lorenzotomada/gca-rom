#!/bin/bash

# SLURM job options
#SBATCH --partition=gpu2
#SBATCH --job-name=log_gca
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=40000
#SBATCH --time=05:00:00
#SBATCH --output=%x.o%j.%N
#SBATCH --error=%x.e%j.%N

# Print job details
NOW=`date +%H:%M-%a-%d/%b/%Y`
echo '------------------------------------------------------'
echo 'This job is allocated on '$SLURM_JOB_CPUS_PER_NODE' cpu(s)'
echo 'Job is running on node(s): '
echo  $SLURM_JOB_NODELIST
echo '------------------------------------------------------'
#
# ==== End of Info part ===== #
#

cd $SLURM_SUBMIT_DIR
export SLURM_NTASKS_PER_NODE=1

module load cuda/12.1
conda init
source ~/.bashrc
conda activate /scratch/ltomada/miniconda3/envs/gca_rom

OUTPUT_FILE_NAME=$1

if [ -z "$OUTPUT_FILE_NAME" ]; then
    echo "No file name provided. Using default."
    OUTPUT_FILE_NAME="square_advection"
fi

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python ${OUTPUT_FILE_NAME}.py
