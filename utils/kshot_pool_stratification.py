import argparse
import json
import os
import random
import numpy as np
import yaml
import parsing_embedding as pe
from sklearn.metrics.pairwise import cosine_similarity

import kmeans_params

def compute_coverage(data, selected_abstract_ids, total_size):
        return sum(1 for item in data if item["abstract_id"] in selected_abstract_ids) / total_size

def cluster_data(args):
    with open(args.parsed_features, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    data = [entry for entry in data if entry.get("corpus") == "NCBI_train"]
    graphs, _ = pe.build_dependency_graphs(data)
    emb, _ = pe.get_graph_embedding(graphs)
    if not os.path.exists(args.cluster_dir):
        kmeans_params.main(args, emb)  # Call the kmeans_params script to compute clusters
    
    centroids_real = np.load(os.path.join(args.cluster_dir, "cluster_centroids.npy"))
    
    similarities = cosine_similarity(emb, centroids_real)
    labels = np.argmax(similarities, axis=1)
    with open(args.clustered_file, "w", encoding="utf-8") as f:
        for sample, label in zip(data, labels):
            sample["cluster_id"] = int(label)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return len(data)

def extract_data_size(folder):
    with open(folder, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    return len(data)


def extract_abstracts_from_clusters(input_file, output_file, sample_ratio, ncbi_size, seed=42):
    # Load the full NCBI dataset (that contains cluster classes)
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    np.random.seed(seed)
    total_size = len(data)

    cluster_labels = sorted({item["cluster_id"] for item in data})
    print(f"Total clusters: {len(cluster_labels)}")
    all_abstract_ids = sorted({item["abstract_id"] for item in data})
    selected_abstract_ids = set()

    for cluster_label in cluster_labels: # U svakom klasteru uzmi jedan abstract_id
        cluster_abstract_ids = [item["abstract_id"] for item in data if item["cluster_id"] == cluster_label]
        np.random.shuffle(cluster_abstract_ids)
        for abstract_id in cluster_abstract_ids:
            if abstract_id not in selected_abstract_ids:
                selected_abstract_ids.add(abstract_id)
                break

    np.random.shuffle(all_abstract_ids)

    i = 0
    while compute_coverage(data, selected_abstract_ids, total_size) < sample_ratio and i < len(all_abstract_ids):
        aid = all_abstract_ids[i]
        if aid not in selected_abstract_ids:
            selected_abstract_ids.add(aid)
        i += 1

    filtered_data = [item for item in data if item["abstract_id"] in selected_abstract_ids]

    final_ratio = len(filtered_data) / ncbi_size

    # Save to new JSON file
    filtered_abstracts = output_file + f"ncbi_ner_train_{final_ratio*100:.0f}pct.json"
    with open(filtered_abstracts, "w", encoding="utf-8") as f:
        for item in filtered_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(filtered_data)} sentences to {filtered_abstracts}")

    return filtered_abstracts
    

def argparse_args():
    parser = argparse.ArgumentParser(description="LLM-based text Generator iteration pipeline with k-shot.")
    parser.add_argument('--config_file', type=str, default='/home/${USERNAME}/syn-bioner/bioNER/experiments/pool_stratification.yml', help='Path to config file with all arguments.')

    return parser.parse_args()

def load_yaml_with_env(path):
    with open(path) as f:
        content = os.path.expandvars(f.read())
    return yaml.safe_load(content)


'''singularity exec --nv --cleanenv $CLIENT_IMAGE /opt/conda/envs/gen/bin/python3 /home/${USERNAME}/syn-bioner/bioNER/utils/kshot_pool_stratification.py'''
if __name__ == "__main__":
    init_args = argparse_args()
    yaml_args = load_yaml_with_env(init_args.config_file,)
    args = argparse.Namespace(**yaml_args)
        
    subset_pcts = sorted(args.pcts, reverse=True) #make sure pcts go from bigger to smaller
    if not os.path.exists(args.clustered_file):
        ncbi_size = cluster_data(args)
    else:
        ncbi_size = extract_data_size(args.clustered_file)
    available_abstracts = args.clustered_file
    previous_pct = 1

    for pct in subset_pcts:
        samples_pct = float(pct) / float(previous_pct) # so that it contains given % from train dataset and not subset it is being extracted from
        filtered_abstracts = extract_abstracts_from_clusters(available_abstracts, args.output_path, samples_pct, ncbi_size)
        available_abstracts = filtered_abstracts
        previous_pct = pct
