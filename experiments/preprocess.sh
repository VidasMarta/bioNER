#!/bin/bash
#SBATCH --job-name=data-cpu-preprocess
#SBATCH --output=output/output/%j
#SBATCH --error=output/error/%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=04:30:00
# #SBATCH --partition=gpu

# Paths to your images
USERNAME=mkeber

# Matching pairs (same order!)
CORPORA=(ncbi)
# CORPORA=(ncbi distemist quaero bronco150)
SPACY_MODELS=(en_core_web_trf)
# SPACY_MODELS=(en_core_web_trf es_dep_news_trf fr_dep_news_trf de_dep_news_trf)

CLIENT_IMAGE=/home/${USERNAME}/sif-files/synbioner_generate2.sif
BIND_PATHS_SPACY=/home/${USERNAME}/models/spacy_models:/models

echo "Start time: $(date +"%H:%M:%S")"

# Loop through indices
for i in "${!CORPORA[@]}"; do
  CORPUS=${CORPORA[$i]}
  SPACY=${SPACY_MODELS[$i]}

  echo "Running CORPUS=$CORPUS with SPACY=$SPACY"

  singularity exec --nv -B $BIND_PATHS_SPACY \
    $CLIENT_IMAGE bash -c "
    export PYTHONPATH=/models:\$PYTHONPATH && \
    python3 -m bioNER.data_wranglig.${CORPUS}_to_json \
      --config_file bioNER/experiments/data/${CORPUS}.yml
    "
done

echo "End time: $(date +"%H:%M:%S")"

# singularity exec --nv \
#   -B $BIND_PATHS_SPACY \
#   $CLIENT_IMAGE bash -c "
# export PYTHONPATH=/models:\$PYTHONPATH
# python3 utils/save_spacy_to_disk.py --model ${SPACY}
# "
