import argparse
import json
import os
import random
import numpy as np
import parsing_embedding as pe

def compute_coverage(data, selected_abstract_ids, total_size):
        return sum(1 for item in data if item["abstract_id"] in selected_abstract_ids) / total_size


def extract_abstracts_from_clusters(input_file, output_file, cluster_dir, sample_ratio, seed=42):
    # Load the full NCBI dataset (that contains cluster classes)
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    cluster_labels = np.load(os.path.join(cluster_dir, "cluster_labels.npy"))
    clustered_data = []
    for sample, label in zip(data, cluster_labels):
        sample['cluster_id'] = label
        print(sample)
        clustered_data.append(sample)
    data = clustered_data

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

    final_ratio = len(filtered_data) / total_size

    # Save to new JSON file
    filtered_abstracts = output_file + f"ncbi_ner_train_{final_ratio*100:.0f}pct.json"
    with open(filtered_abstracts, "w", encoding="utf-8") as f:
        for item in filtered_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(filtered_data)} sentences to {filtered_abstracts}")

    return filtered_abstracts
    
def parse_args():
    parser = argparse.ArgumentParser(description="Parsing")
    parser.add_argument('--pcts', type=float, nargs="+", required=True, help='Percentages...')  
    parser.add_argument('--filtered_parsed_mesh_file', type=str, required=False, help='Directory to where to save filtered parsed mesh NCBI train json', default="data/MeSH_NCBI/sm/")
    parser.add_argument('--output_path_features', type=str, required=False, help='Path where to save output features', default="data/MeSH_NCBI/sm/syntax_features_sent_tree_head.json")
    parser.add_argument('--cluster_dir', type=str, required=False, help='Path where kmeans centroids are saved', default="/home/mvidas/syn-bioner/data/generation_pipeline/kmeans_clusters/")
    return parser.parse_args()

'''singularity exec --nv --cleanenv $CLIENT_IMAGE /opt/conda/envs/gen/bin/python3 /home/mvidas/syn-bioner/bioNER/utils/kshot_pool_stratification.py --pcts 0.5 0.2 0.1 --filtered_parsed_mesh_file /home/mvidas/syn-bioner/data/ncbi/trf/ncbi_ner_train.json --output_path_features /home/mvidas/syn-bioner/data/ncbi/trf/syntax_features_sent_tree_head.json --cluster_dir /home/mvidas/syn-bioner/data/generation_pipeline/kmeans_clusters'''

if __name__ == "__main__":
    args = parse_args()
    for path in [args.filtered_parsed_mesh_file, args.output_path_features]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
    subset_pcts = sorted(args.pcts, reverse=True) #make sure pcts go from bigger to smaller
    available_abstracts = args.output_path_features
    previous_pct = 1

    for pct in subset_pcts:
        samples_pct = pct / previous_pct # so that it contains given % from train dataset and not subset it is being extracted from
        filtered_abstracts = extract_abstracts_from_clusters(available_abstracts, args.filtered_parsed_mesh_file, args.cluster_dir, samples_pct)
        available_abstracts = filtered_abstracts
        previous_pct = pct
