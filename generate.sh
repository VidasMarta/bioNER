#!/bin/bash
#SBATCH --job-name=singularity-gpu-2containers
#SBATCH --output=output/output-%j.out
#SBATCH --error=output/error-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --time=80:30:00
# #SBATCH --partition=gpu

# Paths to your images
SERVER_IMAGE=/home/mkeber/sif-files/llama.cpp_server-cuda.sif
CLIENT_IMAGE=/home/mkeber/sif-files/synbioner_generate.sif
WORKDIR=/home/mkeber/models/quantized
WAIT=60
# Optional binding
BIND_PATHS_SERVER="${WORKDIR}:/models"

echo "Current time: $(date +"%H:%M:%S")"

# Launch the server container in background
singularity exec --nv --pwd /app --no-home \
  --network-args "portmap=8484:8484/tcp" \
  --bind $BIND_PATHS_SERVER \
  $SERVER_IMAGE /app/llama-server -m /models/medgemma-27b-text-it-BF16-00001-of-00002.gguf --port 8484 --host 0.0.0.0 -n 5000 --n-gpu-layers 999 &

echo "Waiting for llama.cpp server to load model wait time is $WAIT ..."
sleep $WAIT
echo " "
echo "STARTING translation:"
echo "Bigger models will need even more waiting time currently $WAIT seconds"

# singularity exec --nv --cleanenv $CLIENT_IMAGE python3 src/translateClinicalNotes.py --csv data/agbonnet/agbonet.csv\
#     --system_template translation --temperature 0 --max_tokens 5000\
#         --server_url http://0.0.0.0:8484 --column_to_translate full_note\
#             --output_dir data/agbonnet

singularity exec --nv --cleanenv $CLIENT_IMAGE python3 llmAnnotationGeneration.py \
  --input_file data/NCBI-Disease/test.txt \
  --input_file_type list \
  --output_directory data/synthetic_aug \
  --server_url http://0.0.0.0:8484 \
  --kshot_path data/ncbi/trf/ncbi_train_10pct.json \
  --kshot_size 5 \
  --num_sentences 2 \
  --include_pos \
  --include_dep \
  --random_seed 42 \
  --verbose --test
echo "Current time: $(date +"%H:%M:%S")"