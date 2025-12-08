import argparse
import json
import os
import random
from typing import Any, Dict, List
import yaml
from src_generate.llmAnnotationGenerationLatest import *
from src_generate.llmAnnotationGenerationLatest import setup_logger
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import utils.parsing_embedding as pe
import utils.kmeans_params as kmeans_params
import umap
import subprocess
import matplotlib.pyplot as plt

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
    # Assign synthetic embeddings to nearest real cluster center
    similarities = cosine_similarity(synth_emb, centroids_real)
    synth_labels = np.argmax(similarities, axis=1)


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
            overlap_i = cosine_similarity(
                real_cluster_emb.mean(axis=0, keepdims=True), #compute average embedding for real and synthetic cluster (centroids)
                synth_cluster_emb.mean(axis=0, keepdims=True),
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
        print("[WARN] No NCBI examples found for uncovered clusters, using random fallback.")
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

    # Save NCBI examples with cluster labels for later k-shot selection
    # KARATE ENV
    ncbi_clustered_path = os.path.join(args.output_directory, "kshot_ncbi_clustered.jsonl")
    kshot_graphs, _ = pe.build_dependency_graphs(kshot_data)
    kshot_emb, _ = pe.get_graph_embedding(kshot_graphs, model)
    similarities = cosine_similarity(kshot_emb, centroids_real)
    kshot_labels = np.argmax(similarities, axis=1)
    with open(ncbi_clustered_path, "w") as f:
        for sample, label in zip(kshot_data, kshot_labels):
            sample["cluster_id"] = int(label)
            f.write(json.dumps(sample) + "\n")
    print(f"[INFO] Saved NCBI examples with cluster IDs → {ncbi_clustered_path}")

    if args.test:
        args.max_iterations = 3

    while True:
        print(f"\n[ITERATION {iteration}] Computing synthetic embeddings...")
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

        # Stopping conditionsS
        if weighted_coverage >= args.weighted_threshold and not uncovered_clusters:
            print(
                f"[STOP] Coverage target reached: {weighted_coverage:.3f} "
                f"(all clusters sufficiently represented)."
            )
            visualize_embeddings_clusterwise(real_emb, real_labels, synth_emb, synth_labels, cluster_overlaps, args.output_directory)
            break

        elif iteration == args.max_iterations:
            print("[STOP] Max iteration reached target reached.")
            visualize_embeddings_clusterwise(real_emb, real_labels, synth_emb, synth_labels, cluster_overlaps, args.output_directory)
            break

        # Adaptive regeneration: guided by uncovered clusters
        regen_terms = []
        terms_per_cluster = []

        for cluster_id in uncovered_clusters:
            # Determine regeneration strength (0–1): the lower the overlap, the higher the regen weight
            regen_weight = 1.0 - cluster_overlaps[cluster_id]

            # Determine how many new samples to generate for this cluster
            # TODO jel nam ovo ok?
            # You can tune scaling constant (args.min_samples_per_cluster acts as base target)
            num_new = max(1, int(regen_weight * args.min_samples_per_cluster))

            # Collect synthetic terms already assigned to this cluster
            cluster_mask = synth_labels == cluster_id
            indices_in_cluster = [i for i in range(len(synth_data)) if cluster_mask[i] and "entities" in synth_data[i]] 
            cluster_terms = [synth_data[i].get("entities") for i in indices_in_cluster]
            

            # If not enough terms in this cluster, sample some from the global synthetic pool as backup
            if len(cluster_terms) < num_new:
                print(f"[INFO] found {len(cluster_terms)} for cluster {cluster_id}, will get more globally.")
                global_terms = [d["entities"] for d in synth_data if "entities" in d]

                if len(global_terms) == 0:
                    print("[WARN] No global terms available for regeneration at all.")
                else:
                    needed = num_new - len(cluster_terms)
                    extra_terms = list(np.random.choice(
                        global_terms,
                        size=min(needed, len(global_terms)),
                        replace=False
                    ))
                    cluster_terms.extend(extra_terms)

            #Some sentences have multiple terms (entities)
            clean_cluster_terms = []
            for t in cluster_terms:
                if isinstance(t, list):
                    clean_cluster_terms.extend(t)
                elif isinstance(t, str):
                    clean_cluster_terms.append(t)


            # Randomly pick terms for this cluster’s regeneration quota
            selected_terms = np.random.choice(clean_cluster_terms, size=num_new, replace=False)
            regen_terms.extend(selected_terms)
            terms_per_cluster.append(num_new)

        print(f"[INFO] Total new terms to regenerate across clusters: {len(regen_terms)}")

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
        if args.test:
            regen_terms = regen_terms[:3]
        #new_path = generate_sentence_samples(args, kshot_file, regen_terms, method="a")
        new_path = generate_sentences_per_cluster(args, kshot_file, regen_terms, uncovered_clusters, terms_per_cluster)

        postprocessed = os.path.join(iter_output, f"syntax_features_iter_{iteration}.jsonl")
        sub_results = subprocess.run([
            "/opt/conda/bin/python3", "/home/mvidas/syn-bioner/bioNER/generation_postprocessing.py",
            "--generated", new_path,
            "--generated_postprocessed", postprocessed,
            "--starting_id", str(len(synth_data) + 1),
            "--spacy_model", args.spacy_model
        ], capture_output=True, text=True)
        if args.verbose:
            print("STDOUT:\n", sub_results.stdout)
            print("STDERR:\n", sub_results.stderr)

        with open(postprocessed, "r") as f:
            newly_parsed_sentences = [json.loads(line) for line in f]

        if clean_cluster_terms:  # remove sentences from this cluster whose terms were selected for regeneration
            terms_to_replace = set(clean_cluster_terms)
            indices_to_replace = [
                i for i, entry in enumerate(synth_data)
                if any(ent in terms_to_replace for ent in entry.get("entities", []))
                and synth_labels[i] == cluster_id
            ]
    
    # Remove in reverse order to preserve indexing
    for idx in sorted(indices_to_replace, reverse=True):
        del synth_data[idx]


        # Add new regenerated ones
        synth_data.extend(newly_parsed_sentences)
        print(f"[INFO] Added {len(newly_parsed_sentences)} parsed sentences to synthetic corpus.")

        if args.ner_model_eval:
            train_path = os.path.join(iter_output, "train_synth_iter.jsonl")

            with open(train_path, "w") as f:
                for ex in synth_data:
                    f.write(json.dumps(ex) + "\n")

            print("[NER] Started spacy NER model training.")
            sub_results = subprocess.run([
                "/opt/conda/bin/python3", "/home/mvidas/syn-bioner/bioNER/utils/train_spacy_ner.py",
                "--train", postprocessed,
                "--output", f"{iter_output}/ner_model",
                "--n_iter", args.num_train_iter,
            ], capture_output=True, text=True)

            print("[NER] Spacy NER model evaluating.")
            logger_file = os.path.join(args.output_directory, "logger.jsonl")
            sub_results = subprocess.run([
                "/opt/conda/bin/python3", "/home/mvidas/syn-bioner/bioNER/utils/eval_spacy_ner.py",
                "--model", f"{iter_output}/ner_model",
                "--test", args.ncbi_dev_set,
                "--logger", logger_file,
            ], capture_output=True, text=True)

        iteration += 1

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
singularity exec --nv --cleanenv $CLIENT_IMAGE /opt/conda/envs/gen/bin/python3 /home/mvidas/syn-bioner/bioNER/generation_pipeline.py
'''


if __name__ == "__main__":
    init_args = argparse_args()
    with open(init_args.config_file, 'r') as file:
        yaml_args = yaml.safe_load(file)
    args = argparse.Namespace(**yaml_args)
    logger = setup_logger(args)
    args.logger = logger
    logger.info(f"Output directory: {args.output_directory}")
    main(args)