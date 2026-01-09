import argparse
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from typing import List, Dict, Any
import os

import os
import numpy as np
import networkx as nx
from karateclub import Graph2Vec
from karateclub import GL2Vec
import json
from tqdm import tqdm

def build_dependency_graphs(data):
    """
    Build directed dependency graphs encoding only syntactic structure (POS + dependency).
    Graphs are directed from parent -> child.
    
    Args:
    data (list[dict]): list of dependency-parsed sentence dicts.
    
    Returns:
    dict[int, nx.DiGraph]: mapping from sentence ID to NetworkX graph.
    """
    graphs = []
    sent_ids_list = []
    for entry in data:
        sent_ids_list.append(entry["id"])
        sent_id = entry["id"]
        sentence = entry["sentence"]
        pos_tags = entry["pos"]
        dep_labels = entry["dep"]
        parents = entry["parents"]
        G = nx.DiGraph(id=sent_id, sentence=sentence)
        n = len(pos_tags)
        for i in range(0, n):
            node_label = f"{pos_tags[i]}"
            G.add_node(i, feature=node_label, is_root=False)
    
        # --- Add edges (parent -> child) ---
        for child_idx, parent_idx in enumerate(parents):
            G.add_edge(parent_idx, child_idx, feature=f'{dep_labels[child_idx]}')
        
        graphs.append(G)
    return graphs, sent_ids_list
 
def get_graph_embedding(graphs, model=None, embedding_type="gl2vec", wl_iterations=1, 
                        dimensions=16, workers=24, learning_rate=0.05,
                        min_count=1, epochs=20):

    if model is None:
        if embedding_type == "gl2vec" :
            model = GL2Vec(wl_iterations=wl_iterations, dimensions=dimensions, 
                        workers=workers, learning_rate=learning_rate, 
                        min_count=min_count, epochs=epochs)
        elif embedding_type == "graph2vec":
            model = Graph2Vec(wl_iterations=wl_iterations, dimensions=dimensions, 
                            workers=workers, learning_rate=learning_rate,
                            min_count=min_count, epochs=epochs)
        else:
            raise Exception(f"No such embedding type {embedding_type}!")
    model.fit(graphs) #list(graphs.values()))
    embeddings = model.get_embedding()
    embeddings_data = np.array(embeddings)
    return embeddings_data, model


def load_syntax_data(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def compute_embeddings(parsed_data: List[Dict[str, Any]]) -> np.ndarray:
    graphs, _ = build_dependency_graphs(parsed_data)
    return get_graph_embedding(graphs)

def evaluate_k(real_emb: np.ndarray, k: int) -> Dict[str, float]:
    real_emb = normalize(real_emb)
    km = KMeans(n_clusters=k, random_state=42)
    
    labels = km.fit_predict(real_emb)

    inertia = km.inertia_
    silhouette = silhouette_score(real_emb, labels)

    return {
        "inertia": inertia,
        "silhouette": silhouette,
        "labels": labels,
        "centroids": km.cluster_centers_,
    }

def main(args, emb):
    os.makedirs(args.cluster_dir, exist_ok=True)

    if emb is None:
        print("[INFO] Loading parsed syntax data...")
        data = load_syntax_data(args.parsed_features)

        print("[INFO] Computing embeddings...")
        emb, _ = compute_embeddings(data)

    results = []
    best_k = None
    best_score = -np.inf

    print("[INFO] Evaluating cluster sizes...")
    for k in range(args.k_min, args.k_max + 1):
        print(f" → K = {k}")

        res = evaluate_k(emb, k)
        silhouette = res["silhouette"]

        results.append({
            "k": k,
            "inertia": str(res["inertia"]),
            "silhouette": str(silhouette),
        })
        print(results)
        # Choose best K based on silhouette
        if silhouette > best_score:
            best_score = silhouette
            best_k = k
            best_labels = res["labels"]
            best_centroids = res["centroids"]

    # Save evaluation table
    eval_path = os.path.join(args.cluster_dir, "cluster_evaluation.json")
    with open(eval_path, "w") as f:
        for res in results:
            f.write(json.dumps(res) + '\n')
    print(f"[INFO] Saved clustering evaluation → {eval_path}")

    # Save best cluster assignments
    np.save(os.path.join(args.cluster_dir, "cluster_labels.npy"), best_labels)
    np.save(os.path.join(args.cluster_dir, "cluster_centroids.npy"), best_centroids)

    # Save config
    config = {
        "best_k": best_k,
        "silhouette": str(best_score)
    }
    with open(os.path.join(args.cluster_dir, "cluster_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"[SUCCESS] Best K = {best_k} (silhouette={best_score:.4f})")
    print(f"[INFO] Labels and centroids saved in {args.cluster_dir}")


"""python3 utils/kmeans_params.py --parsed_features /home/${USERNAME}/syn-bioner/data/ncbi/trf/ncbi_ner_train_10pct.json \
--cluster_dir /home/${USERNAME}/syn-bioner/data/clustering --k_min 2 --k_max 4 """

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Determine optimal number of syntax clusters.")
    parser.add_argument("--parsed_features", type=str, required=True,
                        help="Path to syntax_features.jsonl")
    parser.add_argument("--cluster_dir", type=str, required=True)
    parser.add_argument("--k_min", type=int, default=2)
    parser.add_argument("--k_max", type=int, default=20)
    args = parser.parse_args()
    main(args, None)