#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=SummaryHeadlines
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mcn8851@nyu.edu
#SBATCH --output=summary_headlines_%j.out

module purge

if [ -e /dev/nvidia0 ]; then nv="--nv"; fi

singularity exec $nv \
  --overlay /scratch/mcn8851/LLM_env/overlay-50G-10M.ext3:ro \
  /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; python summary_headlines.py"


