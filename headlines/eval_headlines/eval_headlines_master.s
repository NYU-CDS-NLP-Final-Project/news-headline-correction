#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=HeadlineMaster
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mcn8851@nyu.edu
#SBATCH --output=eval_headlines_master.out

module purge

if [ -e /dev/nvidia0 ]; then nv="--nv"; fi

singularity exec $nv \
  --overlay /scratch/mcn8851/LLM_env/overlay-50G-10M.ext3:ro \
  /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; python eval_headlines_master.py"


