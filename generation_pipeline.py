import argparse
import json
import os
import random
from typing import Any, Dict, List
import yaml
from src_generate.llmAnnotationGenerationLatest import *
from src_generate.utils import setup_logger
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import utils.parsing_embedding as pe
import utils.kmeans_params as kmeans_params
import umap
import subprocess
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

def argparse_args():
    parser = argparse.ArgumentParser(description="LLM-based text Generator iteration pipeline with k-shot.")
    parser.add_argument('--config_file', type=str, default='/home/mvidas/syn-bioner/bioNER/experiments/default_generate.yml', help='Path to config file with all arguments.')

    return parser.parse_args()

def compute_cluster_coverage(
    real_emb: np.ndarray,
    synth_emb: np.ndarray,
    real_labels: np.ndarray,
    centroids_real: np.ndarray,
    overlap_threshold: float,
    iter: int,
    output_dir: str,
    min_samples_per_cluster: int = 10,
):
    """
    Compute per-cluster cosine overlap and weighted global coverage.
    Args:
        real_emb (np.ndarray): NCBI syntax embeddings.
        synth_emb (np.ndarray): Generated syntax embeddings.
        real_labels (np.ndarray): Cluster labels for NCBI embeddings.
        centroids_real (np.ndarray): Centroids of NCBI clusters.
        overlap_threshold (float): Minimum cosine similarity to consider cluster covered.
        min_samples_per_cluster (int): Minimum number of synthetic samples per cluster.
    """
    n_clusters = len(np.unique(real_labels))
    print(f"[DEBUG] emb length {len(synth_emb)}")
    # Assign synthetic embeddings to nearest real cluster center
    similarities = cosine_similarity(synth_emb, centroids_real)
    synth_labels = np.argmax(similarities, axis=1)
    print(f"[DEBUG] labels length {len(synth_labels)}")


    cluster_overlaps = []
    uncovered_clusters = []
    cluster_sizes = []
    synth_counts = []

    log = {"per_cluster": []}

    for i in range(n_clusters):
        real_cluster_emb = real_emb[real_labels == i] #take all embeddings from real data that belong to cluster i
        synth_cluster_emb = synth_emb[synth_labels == i] #take all embeddings from synthetic data that belong to cluster i

        cluster_sizes.append(len(real_cluster_emb))
        synth_counts.append(len(synth_cluster_emb))

        overlap_i = 0.0
        if len(synth_cluster_emb) == 0: #no synthetic samples in this cluster
            cluster_overlaps.append(0.0) 
            uncovered_clusters.append(i)
            continue
        
        #compute how similar the average syntactic embedding of generated sentences is to the real NCBI sentences within that cluster
        else:
            real_centroid = normalize(real_cluster_emb).mean(axis=0, keepdims=True)  #compute average embedding for real and synthetic cluster (centroids)
            syntax_centroid = normalize(synth_cluster_emb).mean(axis=0, keepdims=True)
            overlap_i = cosine_similarity(
                real_centroid,
                syntax_centroid,
            )[0, 0]
            cluster_overlaps.append(overlap_i) #add 

            if overlap_i < overlap_threshold or len(synth_cluster_emb) < min_samples_per_cluster:
                uncovered_clusters.append(i)

        log["per_cluster"].append({
            "cluster_id": i,
            "cluster_centroid_overlap": float(overlap_i),  
            "real_cluster_size": int(cluster_sizes[-1]),
            "synth_cluster_size": int(synth_counts[-1]),
            "is_uncovered": i in uncovered_clusters
        })

    # Weighted average (by cluster size)
    cluster_sizes = np.array(cluster_sizes)
    weighted_coverage = np.sum(cluster_sizes * np.array(cluster_overlaps)) / np.sum(cluster_sizes)
    
    log["iteration_summary"] = {
        "iteration": iter,
        "uncovered_clusters": uncovered_clusters,
        "weighted_coverage": weighted_coverage
    }
    logger_file = os.path.join(output_dir, "logger.jsonl")
    with open(logger_file, "a") as f:
        f.write(json.dumps(log) + "\n")

    return cluster_overlaps, uncovered_clusters, weighted_coverage, cluster_sizes, synth_labels

def get_cluster_specific_kshot(ncbi_clustered_path, uncovered_clusters):
    """Return a list of NCBI examples sampled from uncovered clusters."""
    with open(ncbi_clustered_path, "r") as f:
        data = [json.loads(line) for line in f]
    filtered = [ex for ex in data if ex.get("cluster_id") in uncovered_clusters]
    
    if not filtered:
        print("[WARN] No NCBI examples found for uncovered clusters, using random fallback.") #TODO dodati da se "sakriju" entiteti
        filtered = data
    
    return filtered


def visualize_embeddings_clusterwise(
    real_emb, real_labels, synth_emb, synth_labels, cluster_overlaps, save_path
):
    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42
    )
    both = np.vstack([real_emb, synth_emb])
    coords = reducer.fit_transform(both)
    labels = np.concatenate([real_labels, synth_labels + real_labels.max() + 1])

    plt.figure(figsize=(8, 7))
    plt.scatter(
        coords[: len(real_emb), 0],
        coords[: len(real_emb), 1],
        c=real_labels,
        cmap="tab10",
        s=15,
        alpha=0.6,
        label="NCBI_train",
    )
    plt.scatter(
        coords[len(real_emb) :, 0],
        coords[len(real_emb) :, 1],
        c=synth_labels,
        cmap="tab10",
        s=15,
        alpha=0.6,
        marker="x",
        label="Generated",
    )
    plt.title("Syntax Cluster Alignment (UMAP)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"[FINAL] Cluster overlaps: {[round(x, 3) for x in cluster_overlaps]}")

    os.makedirs(save_path, exist_ok=True)
    fname = f"umap_overlap.png"
    full_path = os.path.join(save_path, fname)
    plt.savefig(full_path, dpi=200)
    print(f"[INFO] Saved UMAP visualization to {full_path}")

def delete_excess_synht(clusters, synth_labels, real_labels, synth_data, real_data, max_synthetic_ratio):
    ids_to_delete = []
    synth_labels = np.asarray(synth_labels)
    real_labels = np.asarray(real_labels)
    for cluster_id in clusters:
        synth_idxs = np.where(synth_labels == cluster_id)[0]
        real_idxs = np.where(real_labels == cluster_id)[0]

        cluster_syntax = [synth_data[i] for i in synth_idxs]
        cluster_real = [real_data[i] for i in real_idxs]

        if len(cluster_real) == 0:
            # No real samples → skip or delete synth
            continue

        ratio = len(cluster_syntax) / len(cluster_real)
        if ratio > max_synthetic_ratio: #delete random syntax sentences until ratio as wanted
            to_delete = abs(args.max_synthetic_ratio - len(cluster_syntax))*len(cluster_real)
            selected = random.sample(cluster_syntax, to_delete)
            ids_to_delete.extend([item.get("id") for item in selected])

    if ids_to_delete:
        print(f"[DEBUG] I need to delete {len(ids_to_delete)} sentences")
        print(f"[DEBUG] Len synth_data before {len(synth_data)}")
        synth_data = [item for item in synth_data if item.get("id") not in ids_to_delete]
        print(f"[DEBUG] Len synth_data after {len(synth_data)}")

def adaptive_syntax_generation(
    args,
    real_data: List[Dict[str, Any]],
    kshot_data: List[Dict[str, Any]],
    initial_synth_data: List[Dict[str, Any]],
)-> tuple[List[Dict[str, Any]], Any]:
    """
    Iteratively generate synthetic sentences until GL2Vec embedding overlap
    with NCBI syntax distribution exceeds threshold or max_iterations reached.
    """
    iteration = 0
    synth_data = initial_synth_data
    # Precompute real embeddings once
    print("[INFO] Building real NCBI dependency graphs...")
    # KARATE ENV
    real_graphs, _ = pe.build_dependency_graphs(real_data)
    real_emb, model = pe.get_graph_embedding(real_graphs, embedding_type="gl2vec",
                                              wl_iterations=args.wl_iterations, dimensions=args.dimensions, 
                                              workers=args.workers, learning_rate= args.learning_rate, 
                                              min_count = args.min_count, epochs = args.epochs)

    print(f"[INFO] Clustering real embeddings into syntax clusters...") 
    if not os.path.exists(args.cluster_dir):
        kmeans_params.main(args, real_emb)  # Call the kmeans_params script to compute clusters
    
    real_labels = np.load(os.path.join(args.cluster_dir, "cluster_labels.npy"))
    centroids_real = np.load(os.path.join(args.cluster_dir, "cluster_centroids.npy"))
    clusters = [range(0, max(real_labels)+1)]

    # Save NCBI examples with cluster labels for later k-shot selection
    # KARATE ENV
    ncbi_clustered_path = os.path.join(args.output_directory, "kshot_ncbi_clustered.jsonl")
    if not os.path.exists(ncbi_clustered_path):
        kshot_graphs, _ = pe.build_dependency_graphs(kshot_data)
        kshot_emb, _ = pe.get_graph_embedding(kshot_graphs, model)
        similarities = cosine_similarity(kshot_emb, centroids_real)
        kshot_labels = np.argmax(similarities, axis=1)
        with open(ncbi_clustered_path, "w") as f:
            for sample, label in zip(kshot_data, kshot_labels):
                sample["cluster_id"] = int(label)
                f.write(json.dumps(sample) + "\n")
        print(f"[INFO] Saved NCBI examples with cluster IDs → {ncbi_clustered_path}")

    term_list = generate_term_list(args) #TODO potencijalno dodati nešto da se iskoriste svi termovi

    if args.test:
        args.max_iterations = 2
        synth_data = synth_data[:10000]
        term_list = term_list[:1000]

    while True:
        '''print(f"\n[ITERATION {iteration}] Computing synthetic embeddings...")
        iter_output = os.path.join(args.output_directory, f"iteration_{iteration}/")
        os.makedirs(iter_output, exist_ok=True)
        # KARATE ENV
        synth_graphs, _ = pe.build_dependency_graphs(synth_data)
        synth_emb, _ = pe.get_graph_embedding(synth_graphs, model) 

        (
            cluster_overlaps,
            uncovered_clusters,
            weighted_coverage,
            cluster_sizes,
            synth_labels,
        ) = compute_cluster_coverage(
            real_emb,
            synth_emb,
            real_labels,
            centroids_real,
            args.overlap_threshold,
            iteration,
            args.output_directory,
            args.min_samples_per_cluster
        )

        print(f"[INFO] Per-cluster overlaps: {[round(x, 3) for x in cluster_overlaps]}")
        print(f"[INFO] Weighted coverage = {weighted_coverage:.3f}")
        print(f"[INFO] Uncovered clusters: {uncovered_clusters}")

        # Stopping conditions
        if weighted_coverage >= args.weighted_threshold and not uncovered_clusters:
            print(
                f"[STOP] Coverage target reached: {weighted_coverage:.3f} "
                f"(all clusters sufficiently represented)."
            )
            delete_excess_synht(clusters, synth_labels, real_labels, synth_data, real_data, args.max_synthetic_ratio)
            visualize_embeddings_clusterwise(real_emb, real_labels, synth_emb, synth_labels, cluster_overlaps, args.output_directory)
            break

        elif iteration == args.max_iterations:
            print("[STOP] Max iteration reached target reached.")
            delete_excess_synht(clusters, synth_labels, real_labels, synth_data, real_data, args.max_synthetic_ratio)
            visualize_embeddings_clusterwise(real_emb, real_labels, synth_emb, synth_labels, cluster_overlaps, args.output_directory)
            break

        # Adaptive regeneration: guided by uncovered clusters
        terms_for_gen = []
        terms_per_cluster = []
        
        start = 0
        for cluster_id in uncovered_clusters:            
            synth_idxs = np.where(synth_labels == cluster_id)[0]
            real_idxs = np.where(real_labels == cluster_id)[0]

            cluster_syntax = [synth_data[i] for i in synth_idxs]
            cluster_real = [real_data[i] for i in real_idxs]

            if args.test:
                num_new = 3
            else:
                if args.max_synthetic_ratio > len(cluster_syntax)/len(cluster_real): 
                    num_new = abs(args.max_synthetic_ratio - len(cluster_syntax))*len(cluster_real)
                else:
                    print(f"[DEBUG] No need for generation in this cluster {cluster_id}!")
                    num_new = 0 #TODO onda ne generiramo, ali ako je cluster neprekriven do ovog ne bi trebalo doći?

            end = start + num_new
            terms_for_gen.append(term_list[start:end]) 
            terms_per_cluster.append(num_new)
            start = end
            if start >= len(term_list): #fallback ako baš iskoristimo sve termove
                start = 0


        # Select cluster-specific k-shot examples for uncovered clusters
        kshot_examples = get_cluster_specific_kshot(
            os.path.join(args.output_directory, "kshot_ncbi_clustered.jsonl"),
            uncovered_clusters
        )

        # Save temporarily for prompt conditioning
        kshot_file = os.path.join(iter_output, f"kshot_iter_{iteration}.jsonl")
        with open(kshot_file, "w") as f:
            for ex in kshot_examples:
                f.write(json.dumps(ex) + "\n")

        # Generate new samples from uncovered clusters using LLM
        # SPACY ENV
        new_path = generate_sentences_per_cluster(args, iter_output, kshot_file, terms_for_gen, uncovered_clusters, terms_per_cluster)

        print(f"[INFO] Generation finished...")
        print(f"[DEBUG] regen term example: {terms_for_gen[0]}")

        starting_id = synth_data[-1].get("id")+1
        postprocessed = os.path.join(iter_output, f"syntax_features_iter_{iteration}.jsonl")
        sub_results = subprocess.run([
            "/opt/conda/bin/python3", "/home/mvidas/syn-bioner/bioNER/generation_postprocessing.py",
            "--generated", new_path,
            "--generated_postprocessed", postprocessed,
            "--starting_id", str(starting_id),
            "--config_file", args.config_file
        ], capture_output=True, text=True)
        if args.verbose:
            print("STDOUT:\n", sub_results.stdout)
            print("STDERR:\n", sub_results.stderr)

        with open(postprocessed, "r") as f:
            newly_parsed_sentences = [json.loads(line) for line in f]

        # Add newly generated ones
        synth_data.extend(newly_parsed_sentences)
        print(f"[INFO] Added {len(newly_parsed_sentences)} parsed sentences to synthetic corpus.")
        print(f"[INFO] Now I have {len(synth_data)} parsed sentences in synthetic corpus.")'''

        iter_output = os.path.join(args.output_directory, f"iteration_{iteration}/")
        if args.ner_model_eval:
            train_path = os.path.join(iter_output, "train_synth_iter.jsonl")

            with open(train_path, "w") as f:
                for ex in synth_data:
                    f.write(json.dumps(ex) + "\n")

            print("[NER] Started spacy NER model training.")
            sub_results = subprocess.run([
                "/opt/conda/bin/python3", "/home/mvidas/syn-bioner/bioNER/utils/train_spacy_ner.py",
                "--train", train_path,
                "--output", f"{iter_output}ner_model",
                "--n_iter", str(args.num_train_iter),
            ], capture_output=True, text=True)

            #print(sub_results)

            print("[NER] Spacy NER model evaluating.")
            sub_results = subprocess.run([
                "/opt/conda/bin/python3", "/home/mvidas/syn-bioner/bioNER/utils/eval_spacy_ner.py",
                "--model", f"{iter_output}ner_model",
                "--test", args.ncbi_dev_set,
                "--logger", f"{iter_output}ner_model/logger.jsonl",
            ], capture_output=True, text=True, check=True)
            print(sub_results.stdout)
            print(sub_results.stderr)
        
        iteration += 1
        return


    return synth_data, weighted_coverage

def main(args: argparse.Namespace):    
    ''' 
    iterative generation pipeline (LLM generation -(parsing)-> UMAP/clustering -> condition 
    (while % uncovered NCBI specific clusters) -(k-shot on extracted NCBI specific cluster)-> LLM generation
    '''
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    parsed_data_path = os.path.join(args.output_directory, "syntax_features.jsonl")
    os.makedirs(args.output_directory, exist_ok=True)
    # Load and parse data
    # SPACY ENV

    if not os.path.exists(parsed_data_path):
        print("[INFO] started SYNTAX FEATURES GENERATION!")
    
        sub_results = subprocess.run([
                "/opt/conda/bin/python3", "/home/mvidas/syn-bioner/bioNER/utils/parsing_v2.py",
                "--parsed_mesh_file", args.NCBI_train,
                "--gen_train_path", args.Generated_train,
                "--output_path_features", parsed_data_path,
                "--rewrite",
                "--gen_pipeline",
                "--spacy_model", args.spacy_model
            ], capture_output=True, text=True)
        if args.verbose:
            print("STDOUT:\n", sub_results.stdout)
            print("STDERR:\n", sub_results.stderr)

    with open(parsed_data_path, "r") as f:
        all_data = [json.loads(line) for line in f] 
    ncbi_data = [entry for entry in all_data if entry.get("corpus") == "NCBI_train"]
    synth_data = [entry for entry in all_data if entry.get("corpus") == "generated_train"]
    print(f"[INFO] SAVED: SYNTAX FEATURES GENERATION! {parsed_data_path}")
    with open(args.NCBI_kshot, ) as f:
        kshot_data = [json.loads(line) for line in f]

    final_data, final_overlap = adaptive_syntax_generation(
        args,
        real_data=ncbi_data,
        kshot_data = kshot_data,
        initial_synth_data=synth_data
    )

    print(f"[RESULT] Final syntax overlap: {final_overlap:.3f}")
    final_data_path = os.path.join(args.output_directory, "final_synthetic_data.jsonl")
    with open(final_data_path, "w") as f:
        for entry in final_data:
            f.write(json.dumps(entry) + "\n")


'''
singularity exec --nv --cleanenv $CLIENT_IMAGE /opt/conda/envs/gen/bin/python3 \
    /home/mvidas/syn-bioner/bioNER/generation_pipeline.py
'''


if __name__ == "__main__":
    init_args = argparse_args()
    with open(init_args.config_file, 'r') as file:
        yaml_args = yaml.safe_load(file)
    args = argparse.Namespace(**yaml_args)
    logger = setup_logger(args.output_directory, args.verbose)
    args.logger = logger
    args.config_file = init_args.config_file
    logger.info(f"Output directory: {args.output_directory}")
    main(args)