#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=T5Base
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mcn8851@nyu.edu
#SBATCH --output=t5_headline_%j.out

module purge

if [ -e /dev/nvidia0 ]; then nv="--nv"; fi

singularity exec $nv \
  --overlay /scratch/mcn8851/LLM_env/overlay-50G-10M.ext3:ro \
  /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; python t5_base_headlines.py"



# Add these lines at the end of your Slurm script
if [ -e t5_headline_$SLURM_JOB_ID.out ]; then
    mail -s "T5 Base Job $SLURM_JOB_ID - Output" mcn8851@nyu.edu < t5_headline_$SLURM_JOB_ID.out
fi
