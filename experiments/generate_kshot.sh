#!/bin/bash
#SBATCH --job-name=syn2gpugeneration
#SBATCH --output=output/output/%j
#SBATCH --error=output/error/%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --time=100:30:00
# #SBATCH --partition=gpu

# Paths to your images
WAIT=70
PORT=8484
USERNAME=mkeber
MODEL=${MODEL:-medgemma-27b-text-it-BF16-00001-of-00002.gguf}
# (6, 1536) (4, 2048) 
BATCH=4 
CTX_SINGLE=2304
# Set your BATCH value here

# Calculate batch-dependent variables
BATCH_SIZE=$((BATCH * 128))      # ~2048 when BATCH=3
UBATCH_SIZE=$((BATCH * 64))     # ~1024 when BATCH=3
OUTPUT_TOKENS=$((BATCH * 80))         # Small output (sentence length)
CTX_SIZE=$((BATCH*CTX_SINGLE))
SERVER_IMAGE=/home/${USERNAME}/sif-files/llama.cpp_server-cuda.sif
CLIENT_IMAGE=/home/${USERNAME}/sif-files/synbioner_generate2.sif
WORKDIR=/home/${USERNAME}/models/quantized
WAIT=60
# Optional binding
BIND_PATHS_SERVER="${WORKDIR}:/models"
BIND_PATHS_SPACY="/home/${USERNAME}/models/spacy_models:/models"
WORKDIR=/home/${USERNAME}/models/quantized

# CORPUS=distemist, CORPUS=ncbi, CORPUS=quaero, CORPUS=bronco150
CORPUS=bronco150
# SPACY=es_dep_news_trf, SPACY=en_core_web_trf, SPACY=fr_dep_news_trf, SPACY=de_dep_news_trf
SPACY=de_dep_news_trf
echo "Current time: $(date +"%H:%M:%S")"

# Launch the server container in background
singularity exec --nv --pwd /app --no-home \
  --network-args "portmap=${PORT}:${PORT}/tcp" \
  --bind ${WORKDIR}:/models \
  $SERVER_IMAGE /app/llama-server \
  -m /models/$MODEL \
  --port $PORT --host 0.0.0.0 \
  --ctx-size $CTX_SIZE \
  --batch-size $BATCH_SIZE \
  --ubatch-size $UBATCH_SIZE \
  --parallel $BATCH \
  --n-gpu-layers 999 \
  --swa-full \
  -n $OUTPUT_TOKENS &
  echo "Starting server..."
SERVER_PID=$!
echo $! > llama_server.pid

echo "Waiting for llama.cpp server to load model wait time is $WAIT ..."
sleep $WAIT
echo " "
echo "STARTING generation 0 percent:"
echo "Bigger models will need even more waiting time currently $WAIT seconds, "

# singularity exec --nv \
#   -B $BIND_PATHS_SPACY \
#   $CLIENT_IMAGE bash -c "
# export PYTHONPATH=/models:\$PYTHONPATH
# python3 -m bioNER.utils.save_spacy_to_disk --model ${SPACY}
# "
CORPORA=(ncbi distemist bronco150)
SPACY_MODELS=(en_core_web_trf es_dep_news_trf de_dep_news_trf)

for i in "${!CORPORA[@]}"; do
  CORPUS=${CORPORA[$i]}
  SPACY=${SPACY_MODELS[$i]}
  echo "Running CORPUS=$CORPUS with SPACY=$SPACY"
  singularity exec --nv \
    -B $BIND_PATHS_SPACY \
    $CLIENT_IMAGE bash -c "
  export PYTHONPATH=/models:\$PYTHONPATH
  python3 -m bioNER.kshot_synthetic_generation \
    --config_file bioNER/experiments/${CORPUS}/kshot_generation_1.yml
  "

  echo "bioNER/experiments/${CORPUS}/kshot_generation_50.yml"
  # WAIT 10
  singularity exec --nv \
    -B $BIND_PATHS_SPACY \
    $CLIENT_IMAGE bash -c "
  export PYTHONPATH=/models:\$PYTHONPATH
  python3 -m bioNER.kshot_synthetic_generation \
    --config_file bioNER/experiments/${CORPUS}/kshot_generation_5.yml
  "
done

# echo "bioNER/experiments/${CORPUS}/init_generation.yml"
# WAIT 10
# singularity exec --nv \
#   -B $BIND_PATHS_SPACY \
#   $CLIENT_IMAGE bash -c "
# export PYTHONPATH=/models:\$PYTHONPATH
# python3 -m bioNER.kshot_synthetic_generation \
#   --config_file bioNER/experiments/${CORPUS}/init_generation.yml
# "

# WAIT 10
# singularity exec --nv \
#   -B $BIND_PATHS_SPACY \
#   $CLIENT_IMAGE bash -c "
# export PYTHONPATH=/models:\$PYTHONPATH
# python3 -m bioNER.kshot_synthetic_generation \
#   --config_file bioNER/experiments/${CORPUS}/kshot_generation_all.yml
# "

# echo "bioNER/experiments/${CORPUS}/kshot_generation_10.yml"
# singularity exec --nv \
#   -B $BIND_PATHS_SPACY \
#   $CLIENT_IMAGE bash -c "
# export PYTHONPATH=/models:\$PYTHONPATH
# python3 -m bioNER.kshot_synthetic_generation \
#   --config_file bioNER/experiments/${CORPUS}/kshot_generation_10.yml
# "

