#!/bin/bash
#SBATCH --job-name=data-cpu-preprocess
#SBATCH --output=output/output/%j
#SBATCH --error=output/error/%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
# #SBATCH --partition=gpu

# Paths to your images
USERNAME=mkeber
CORPUS=ncbi
# CORPUS=distemist, CORPUS=ncbi, CORPUS=quaero, CORPUS=bronco150 
SPACY=en_core_web_trf 
# SPACY=es_dep_news_trf, SPACY=en_core_web_trf, SPACY=fr_dep_news_trf, SPACY=de_dep_news_trf

CLIENT_IMAGE=/home/${USERNAME}/sif-files/synbioner_generate2.sif
# Optional binding

BIND_PATHS_SPACY=/home/${USERNAME}/models/spacy_models:/models
echo "Current time: $(date +"%H:%M:%S")"

# singularity exec --nv \
#   -B $BIND_PATHS_SPACY \
#   $CLIENT_IMAGE bash -c "
# export PYTHONPATH=/models:\$PYTHONPATH
# python3 utils/save_spacy_to_disk.py --model ${SPACY}
# "
singularity exec --nv -B $BIND_PATHS_SPACY \
  $CLIENT_IMAGE bash -c "
export PYTHONPATH=/models:\$PYTHONPATH && \
python3 -m bioNER.data_wranglig.${CORPUS}_to_json \
  --config_file bioNER/experiments/data/${CORPUS}.yml
"

# singularity exec --nv -B $BIND_PATHS_SPACY \
#   $CLIENT_IMAGE bash -c "
# export PYTHONPATH=/models:\$PYTHONPATH && \
# python3 -m bioNER.data_wranglig.bronco150_to_json \
#   --config_file bioNER/experiments/data/bronco150.yml
# "

# singularity exec --nv -B $BIND_PATHS_SPACY  \
#   $CLIENT_IMAGE bash -c "
# export PYTHONPATH=/models:\$PYTHONPATH && \
# python3 -m bioNER.data_wranglig.quaero_to_json \
#   --config_file bioNER/experiments/data/quaero.yml
# "

# export PYTHONPATH=/models/spacy_models:\$PYTHONPATH
# python3 -m bioNER.utils.parsing_v2 \
#     --input_file /home/mkeber/syn-bioner/data/processed/distemist/trf/distemist_train.json \
#     --spacy_model fr_dep_news_trf \
#     --output_path_features data/processed/distemist/trf/syntax_features_sent_tree_head.json \
#     --gen_pipeline
# "


# singularity exec --nv -B /home/${USERNAME}/models/spacy_models:/models \
#   $CLIENT_IMAGE bash -c "
# export PYTHONPATH=/models/spacy_models:\$PYTHONPATH
# python3 -m bioNER.data_wranglig.quaero_to_json \
#   --config_file bioNER/experiments/data/quaero.yml
# "

# singularity exec --nv \
#   -B /home/${USERNAME}/models/spacy_models:/models \
#   $CLIENT_IMAGE \
#   pip install \
#   https://github.com/explosion/spacy-models/releases/download/de_dep_news_trf-3.8.0/de_dep_news_trf-3.8.0-py3-none-any.whl \
#   -t /models  


#https://github.com/explosion/spacy-models/releases/download/fr_dep_news_trf-3.8.0/fr_dep_news_trf-3.8.0-py3-none-any.whl \
#   -t /models

# "
# #  https://github.com/explosion/spacy-models/releases/download/fr_dep_news_trf-3.8.0/fr_dep_news_trf-3.8.0-py3-none-any.whl