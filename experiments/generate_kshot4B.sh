#!/bin/bash
#SBATCH --job-name=27Bsyn2gpu
#SBATCH --output=output/output/%j
#SBATCH --error=output/error/%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:2
#SBATCH --time=120:30:00
# #SBATCH --partition=gpu
# Choose the corpus you want to generate synth data for
# set env to: ncbi or distemist or quaero or bronco150
export CORPUS=distemist

TEST=false 
WAIT=30 # if false use 70 seconds
BATCH=48
MODEL=medgemma-4b-it-BF16.gguf
SIZE="4B"
KSHOT_CHECK=0
K_SHOT=3
MAX_TOKENS=128
# Paths to your images
PORT=8284
USERNAME=mkeber
# MODEL=${MODEL:-medgemma-27b-text-it-BF16-00001-of-00002.gguf}
# (6, 1536) (4, 2048) 
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
# Optional binding
BIND_PATHS_SERVER="${WORKDIR}:/models"
BIND_PATHS_SPACY="/home/${USERNAME}/models/spacy_models:/models"
WORKDIR=/home/${USERNAME}/models/quantized
BIND_PATHS_SCRATCH="/scratch:/scratch"
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
  --seed 42 \
  -n $OUTPUT_TOKENS &
  echo "Starting server..."
SERVER_PID=$!
echo $! > llama_server.pid

echo "Waiting for llama.cpp server to load model wait time is $WAIT ..."
sleep $WAIT
echo " "
echo "STARTING generation 0 percent:"
echo "Bigger models will need even more waiting time currently $WAIT seconds, "

export SLURM_TMPDIR=/scratch/slurm_job_${SLURM_JOBID}
mkdir -p ${SLURM_TMPDIR}
export HOME_OUTPUT_DIR="data/synthetic/${CORPUS}"


export OUTPUT_DIR_SCRATCH="${SLURM_TMPDIR}/${CORPUS}"
export SINGULARITYENV_BATCH=$BATCH
export SINGULARITYENV_TEST=$TEST
export SINGULARITYENV_MAX_TOKENS=$MAX_TOKENS
export SINGULARITYENV_SIZE=$SIZE
export SINGULARITYENV_KSHOT_CHECK=$KSHOT_CHECK
export SINGULARITYENV_K_SHOT=$K_SHOT
export SINGULARITYENV_PORT=$PORT

export SINGULARITYENV_CORPUS=$CORPUS

# 0 shot generation
export SINGULARITYENV_K_SHOT="0"
export SINGULARITYENV_OUTPUT_DIR_SCRATCH=${OUTPUT_DIR_SCRATCH}/0shot_syn_generation_0pct_${KSHOT_CHECK}shot_check_${SIZE}

# Check if the final destination on home directory exists and copy the samples to the scratch directory.
SRC_DIR="/home/${USERNAME}/syn-bioner/${HOME_OUTPUT_DIR}/0shot_syn_generation_0pct_${KSHOT_CHECK}shot_check_${SIZE}"
if [[ -d "$SRC_DIR" ]]; then
    echo "Found existing directory: $SRC_DIR"

    mkdir -p "$SINGULARITYENV_OUTPUT_DIR_SCRATCH"

    # Copy contents (including hidden files)
    cp -a "$SRC_DIR"/. "$SINGULARITYENV_OUTPUT_DIR_SCRATCH"/

    echo "Copied contents to: $SINGULARITYENV_OUTPUT_DIR_SCRATCH"
else
    echo "Directory does not exist: $SRC_DIR"
fi
singularity exec --nv \
    -B $BIND_PATHS_SPACY \
    -B $BIND_PATHS_SCRATCH \
    $CLIENT_IMAGE bash -c "
  export PYTHONPATH=/models:\$PYTHONPATH
  python3 -m bioNER.kshot_synthetic_generation \
    --config_file bioNER/experiments/${CORPUS}/0shot_generation.yml
"
cp -r ${SINGULARITYENV_OUTPUT_DIR_SCRATCH} /home/${USERNAME}/syn-bioner/${HOME_OUTPUT_DIR}

export SINGULARITYENV_K_SHOT=$K_SHOT
PCTS=(1 5 10 20 50 100)
for i in "${!PCTS[@]}"; do
export SINGULARITYENV_PCT=${PCTS[$i]}
export SINGULARITYENV_OUTPUT_DIR_SCRATCH=${OUTPUT_DIR_SCRATCH}/${K_SHOT}shot_syn_generation_${SINGULARITYENV_PCT}pct_${KSHOT_CHECK}shot_check_${SIZE}
# Check if the final destination on home directory exists and copy the samples to the scratch directory.
SRC_DIR="/home/${USERNAME}/syn-bioner/${HOME_OUTPUT_DIR}/${K_SHOT}shot_syn_generation_${SINGULARITYENV_PCT}pct_${KSHOT_CHECK}shot_check_${SIZE}"
if [[ -d "$SRC_DIR" ]]; then
    echo "Found existing directory: $SRC_DIR"

    mkdir -p "$SINGULARITYENV_OUTPUT_DIR_SCRATCH"

    # Copy contents (including hidden files)
    cp -a "$SRC_DIR"/. "$SINGULARITYENV_OUTPUT_DIR_SCRATCH"/

    echo "Copied contents to: $SINGULARITYENV_OUTPUT_DIR_SCRATCH"
else
    echo "Directory does not exist: $SRC_DIR"
fi
singularity exec --nv \
    -B $BIND_PATHS_SPACY \
    -B $BIND_PATHS_SCRATCH \
    $CLIENT_IMAGE bash -c "
  export PYTHONPATH=/models:\$PYTHONPATH
    python3 -m bioNER.kshot_synthetic_generation \
      --config_file bioNER/experiments/${CORPUS}/kshot_generation.yml
  "
cp -r ${SINGULARITYENV_OUTPUT_DIR_SCRATCH} /home/${USERNAME}/syn-bioner/${HOME_OUTPUT_DIR}

done