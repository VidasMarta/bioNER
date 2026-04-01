#!/bin/bash
#SBATCH --job-name=singularity-gpu-2containers
#SBATCH --output=output/output/%j
#SBATCH --error=output/error/%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --time=120:30:00
# #SBATCH --partition=gpu


USERNAME=mkeber
CORPUS=bronco150 
# CORPUS=distemist, CORPUS=ncbi, CORPUS=quaero, CORPUS=bronco150 
SPACY=de_dep_news_trf 
# SPACY=es_dep_news_trf, SPACY=en_core_web_trf, SPACY=fr_dep_news_trf, SPACY=de_dep_news_trf

# Paths to your images
SERVER_IMAGE=/home/${USERNAME}/sif-files/llama.cpp_server-cuda.sif
CLIENT_IMAGE=/home/${USERNAME}/sif-files/synbioner_generate2.sif
WORKDIR=/home/${USERNAME}/models/quantized
WAIT=60

# Optional binding
BIND_PATHS_SERVER="${WORKDIR}:/models"
BIND_PATHS_SPACY="home/${USERNAME}/models/spacy_models:/models"


echo "Current time: $(date +"%H:%M:%S")"

# Launch the server container in background
singularity exec --nv --pwd /app --no-home \
  --network-args "portmap=8484:8484/tcp" \
  --bind $BIND_PATHS_SERVER \
  $SERVER_IMAGE /app/llama-server -m /models/medgemma-27b-text-it-BF16-00001-of-00002.gguf --port 8484 --host 0.0.0.0 -n 5000 --n-gpu-layers 999 &

echo "Waiting for llama.cpp server to load model wait time is $WAIT ..."
sleep $WAIT
echo " "
echo "STARTING generation 0 percent:"
echo "Bigger models will need even more waiting time currently $WAIT seconds, "

singularity exec --nv \
  -B $BIND_PATHS_SPACY \
  $CLIENT_IMAGE bash -c "
export PYTHONPATH=/models:\$PYTHONPATH
python3 utils/save_spacy_to_disk.py --model ${SPACY}
"

singularity exec --nv \
  -B $BIND_PATHS_SPACY \
  $CLIENT_IMAGE bash -c "
export PYTHONPATH=/models:\$PYTHONPATH
python3 -m bioNER.kshot_synthetic_generation \
  --config_file bioNER/experiments/${CORPUS}/init_generation.yml
"

singularity exec --nv \
  -B $BIND_PATHS_SPACY \
  $CLIENT_IMAGE bash -c "
export PYTHONPATH=/models:\$PYTHONPATH
python3 -m bioNER.kshot_synthetic_generation \
  --config_file bioNER/experiments/${CORPUS}/kshot_generation_10.yml
"

singularity exec --nv \
  -B $BIND_PATHS_SPACY \
  $CLIENT_IMAGE bash -c "
export PYTHONPATH=/models:\$PYTHONPATH
python3 -m bioNER.kshot_synthetic_generation \
  --config_file bioNER/experiments/${CORPUS}/kshot_generation_20.yml
"

singularity exec --nv \
  -B $BIND_PATHS_SPACY \
  $CLIENT_IMAGE bash -c "
export PYTHONPATH=/models:\$PYTHONPATH
python3 -m bioNER.kshot_synthetic_generation \
  --config_file bioNER/experiments/${CORPUS}/kshot_generation_50.yml
"
singularity exec --nv \
  -B $BIND_PATHS_SPACY \
  $CLIENT_IMAGE bash -c "
export PYTHONPATH=/models:\$PYTHONPATH
python3 -m bioNER.kshot_synthetic_generation \
  --config_file bioNER/experiments/${CORPUS}/kshot_generation_all.yml
"