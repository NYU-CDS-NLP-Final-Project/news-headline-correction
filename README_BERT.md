1. Create a login on huggingface\\ 
2. Get a token from settings on your profile for your machine\\

3. On terminal:\\
singularity exec --overlay /scratch/$USER/LLM\_env/overlay-50G-10M.ext3:rw /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash
\\
source /ext3/env.sh
\\
huggingface-cli login
\\

4. Insert your token here when it asks\\

5. Make sure the requirements are installed and also do: pip install accelerate -U\\

6. Exit\\

7. Make sure to change the hard encoded file paths in the .py file and the .SBATCH file such as the model name, data, output names and the huggingface username
