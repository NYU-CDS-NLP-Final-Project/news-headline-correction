#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu
#SBATCH --job-name=t5_summary
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mcs9843@nyu.edu
#SBATCH --output=/scratch/mcs9834/llm_env/t5_summary.out

module purge

if [ -e /dev/nvidia0 ]; then nv="--nv"; fi

singularity exec $nv \
  --overlay /scratch/mcs9834/llm_env/my_pytorch.ext3:ro \
  /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
  /bin/bash -c "source /ext3/env.sh; python3 /scratch/mcs9834/llm_env/t5_summary.py"
