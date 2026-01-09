#!/bin/bash
#SBATCH --job-name=singularity-gpu-2containers
#SBATCH --output=output/output-%j
#SBATCH --error=output/error-%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --time=80:30:00
# #SBATCH --partition=gpu

# Paths to your images
SERVER_IMAGE=/home/${USERNAME}/sif-files/llama.cpp_server-cuda.sif
CLIENT_IMAGE=/home/${USERNAME}/sif-files/synbioner_generate.sif
WORKDIR=/home/${USERNAME}/models/quantized
WAIT=60
# Optional binding
BIND_PATHS_SERVER="${WORKDIR}:/models"

echo "Current time: $(date +"%H:%M:%S")"

# Launch the server container in background
singularity exec --nv --pwd /app --no-home \
  --network-args "portmap=8484:8484/tcp" \
  --bind $BIND_PATHS_SERVER \
  $SERVER_IMAGE /app/llama-server -m /models/medgemma-27b-text-it-BF16-00001-of-00002.gguf --port 8484 --host 0.0.0.0 -n 5000 --n-gpu-layers 999 &
