#!/bin/bash
#SBATCH --job-name=distance_matrix
#SBATCH --output=output/distance_matrix/output-%j
#SBATCH --error=output/distance_matrix/error-%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
# #SBATCH --partition=gpu

# Paths to your images
USERNAME=mkeber
CLIENT_IMAGE=/home/${USERNAME}/sif-files/synbioner_generate2.sif

singularity exec --nv --cleanenv $CLIENT_IMAGE python3 -m bioNER.utils.distance_matrix \
    --input /home/mkeber/syn-bioner/data/processed/ncbi/trf/ncbi_ner_train.json \
    --input_synth /home/mkeber/syn-bioner/data/synthetic/ncbi/kshot_syn_generation_all/joined.jsonl


# singularity exec --nv --cleanenv $CLIENT_IMAGE python3 -m bioNER.utils.bar_plot_cluster_sizes \
#     --synth /home/mkeber/syn-bioner/data/synthetic/ncbi/kshot_syn_generation_10pct/distance_matrix_nm_5646_14204.npz \
#     --p 30 --strategy mean \
#     --base /home/mkeber/syn-bioner/data/processed/ncbi/trf/syntax_features_sent_tree_head_D.jsonl \
#     --dist /home/mkeber/syn-bioner/data/processed/ncbi/trf/distance_matrix_syntax_features_sent_tree_head.jsonl_5646.npz


# singularity exec --nv --cleanenv $CLIENT_IMAGE python3 -m bar_plot_cluster_sizes \
#     --synth /home/mkeber/syn-bioner/data/synthetic/init_syn_generation_all_new/distance_matrix_nm.npz \
#     /home/mkeber/syn-bioner/data/synthetic/kshot_syn_generation_10pct/distance_matrix_nm.npz \
#     /home/mkeber/syn-bioner/data/synthetic/kshot_syn_generation_20pct/distance_matrix_nm.npz \
#     /home/mkeber/syn-bioner/data/synthetic/kshot_syn_generation_50pct/distance_matrix_nm.npz \
#     /home/mkeber/syn-bioner/data/synthetic/kshot_syn_generation_all/distance_matrix_nm.npz\
#     --p 30 --strategy mean --base /home/mkeber/syn-bioner/data/processed/ncbi/trf/ncbi_ner_train_10pct_D.jsonl \
#     --dist /home/mkeber/syn-bioner/data/processed/ncbi/trf/distance_matrix_ncbi_ner_train_10pct.jsonl.csv

# singularity exec --nv --cleanenv $CLIENT_IMAGE python3 -m bioNER.utils.distance_matrix \
#     --input /home/mkeber/syn-bioner/data/processed/ncbi/trf/syntax_features_sent_tree_head.jsonl \
#     --input_synth /home/mkeber/syn-bioner/data/synthetic/kshot_syn_generation_20pct/corrected_generated_sentences_20260303.jsonl
    
# singularity exec --nv --cleanenv $CLIENT_IMAGE python3 -m bioNER.utils.distance_matrix \
#     --input /home/mkeber/syn-bioner/data/processed/ncbi/trf/syntax_features_sent_tree_head.jsonl \
#     --input_synth /home/mkeber/syn-bioner/data/synthetic/kshot_syn_generation_50pct/corrected_generated_sentences_20260304.jsonl

# singularity exec --nv --cleanenv $CLIENT_IMAGE python3 -m bioNER.utils.distance_matrix \
#     --input /home/mkeber/syn-bioner/data/processed/ncbi/trf/syntax_features_sent_tree_head.jsonl \
#     --input_synth /home/mkeber/syn-bioner/data/synthetic/kshot_syn_generation_all/corrected_generated_sentences_20260304.jsonl

# singularity exec --nv --cleanenv $CLIENT_IMAGE python3 -m bioNER.utils.distance_matrix \
#     --input /home/mkeber/syn-bioner/data/processed/ncbi/trf/syntax_features_sent_tree_head.jsonl \
#     --input_synth /home/mkeber/syn-bioner/data/synthetic/init_syn_generation_10pct/corrected_generated_sentences_20260211.jsonl
# # for plot distance matrix needs to generate the D and the distances between base sentences and all other sentences
# singularity exec --nv --cleanenv $CLIENT_IMAGE python3 -m bioNER.utils.distance_matrix \
#     --input /home/mkeber/syn-bioner/data/processed/ncbi/trf/syntax_features_sent_tree_head.jsonl \
#     --input_synth /home/mkeber/syn-bioner/data/synthetic/init_syn_generation_20pct/corrected_generated_sentences_20260212.jsonl

# singularity exec --nv --cleanenv $CLIENT_IMAGE python3 -m bioNER.utils.distance_matrix \
#     --input /home/mkeber/syn-bioner/data/processed/ncbi/trf/syntax_features_sent_tree_head.jsonl \
#     --input_synth /home/mkeber/syn-bioner/data/synthetic/init_syn_generation_20pct/corrected_generated_sentences_20260212.jsonl


# singularity exec --nv --cleanenv $CLIENT_IMAGE python3 -m bioNER.utils.distance_matrix \
#     --input /home/mkeber/syn-bioner/data/processed/ncbi/trf/syntax_features_sent_tree_head.jsonl \
#     --input_synth /home/mkeber/syn-bioner/data/synthetic/init_syn_generation_all_new/corrected_generated_sentences_20260214.jsonl

# singularity exec --nv --cleanenv $CLIENT_IMAGE python3 -m bar_plot_cluster_sizes --synth /home/mkeber/syn-bioner/data/synthetic/init_syn_generation_all_new/distance_matrix_nm.npz /home/mkeber/syn-bioner/data/synthetic/kshot_syn_generation_50pct/distance_matrix_nm.npz /home/mkeber/syn-bioner/data/synthetic/kshot_syn_generation_all/distance_matrix_nm.npz\
#     --p 30 --strategy mean --base /home/mkeber/syn-bioner/data/processed/ncbi/trf/syntax_features_sent_tree_head_D.jsonl \
#     --dist /home/mkeber/syn-bioner/data/processed/ncbi/trf/distance_matrices/distance_matrix_simple_D.csv



# singularity exec --nv --cleanenv $CLIENT_IMAGE python3 -m bar_plot_cluster_sizes --synth /home/mkeber/syn-bioner/data/synthetic/init_syn_generation_all_new/distance_matrix_nm.npz /home/mkeber/syn-bioner/data/synthetic/kshot_syn_generation_10pct/distance_matrix_nm.npz /home/mkeber/syn-bioner/data/synthetic/kshot_syn_generation_20pct/distance_matrix_nm.npz /home/mkeber/syn-bioner/data/synthetic/kshot_syn_generation_50pct/distance_matrix_nm.npz /home/mkeber/syn-bioner/data/synthetic/kshot_syn_generation_all/distance_matrix_nm.npz\
#     --p 30 --strategy mean --base /home/mkeber/syn-bioner/data/processed/ncbi/trf/ncbi_ner_train_10pct_D.jsonl \
#     --dist /home/mkeber/syn-bioner/data/processed/ncbi/trf/distance_matrix_ncbi_ner_train_10pct.jsonl.csv

# singularity exec --nv --cleanenv $CLIENT_IMAGE python3 -m bioNER.utils.distance_matrix \
#     --input /home/mkeber/syn-bioner/data/processed/ncbi/trf/ncbi_ner_train_10pct.json

# singularity exec --nv --cleanenv $CLIENT_IMAGE python3 -m bioNER.utils.distance_matrix \
#     --input /home/mkeber/syn-bioner/data/processed/distemist/trf/distemist_ner_train.json \
#     --input_synth /home/mkeber/syn-bioner/data/synthetic/distemist/0shot_syn_generation/corrected_generated_sentences_20260316.jsonl
    