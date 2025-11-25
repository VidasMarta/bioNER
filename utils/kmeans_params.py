import argparse
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from typing import List, Dict, Any
import parsing_embedding as pe
import os

def load_syntax_data(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def compute_embeddings(parsed_data: List[Dict[str, Any]]) -> np.ndarray:
    graphs = pe.build_dependency_graphs(parsed_data)
    return pe.get_graph_embedding(graphs)

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

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("[INFO] Loading parsed syntax data...")
    data = load_syntax_data(args.parsed_features)

    print("[INFO] Computing embeddings...")
    emb = compute_embeddings(data)

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
            "inertia": res["inertia"],
            "silhouette": silhouette,
        })

        # Choose best K based on silhouette
        if silhouette > best_score:
            best_score = silhouette
            best_k = k
            best_labels = res["labels"]
            best_centroids = res["centroids"]

    # Save evaluation table
    eval_path = os.path.join(args.output_dir, "cluster_evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Saved clustering evaluation → {eval_path}")

    # Save best cluster assignments
    np.save(os.path.join(args.output_dir, "cluster_labels.npy"), best_labels)
    np.save(os.path.join(args.output_dir, "cluster_centroids.npy"), best_centroids)

    # Save config
    config = {
        "best_k": best_k,
        "silhouette": best_score
    }
    with open(os.path.join(args.output_dir, "cluster_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"[SUCCESS] Best K = {best_k} (silhouette={best_score:.4f})")
    print(f"[INFO] Labels and centroids saved in {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Determine optimal number of syntax clusters.")
    parser.add_argument("--parsed_features", type=str, required=True,
                        help="Path to syntax_features.jsonl")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--k_min", type=int, default=2)
    parser.add_argument("--k_max", type=int, default=20)
    args = parser.parse_args()
    main()