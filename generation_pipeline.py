import argparse
import json
import os
import random
from typing import Any, Dict, List
from sklearn.cluster import KMeans
from src_generate.llmAnnotationGenerationLatest import main as generate_sentence_samples
from src_generate.llmAnnotationGenerationLatest import setup_logger
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import utils.parsing_embedding as pe
import utils.parsing_v2 as parser
import utils.kmeans_params as kmeans_params
import umap
import src_generate.utils as utils
import subprocess
import matplotlib.pyplot as plt

def argparse_args():
    parser = argparse.ArgumentParser(description="LLM-based text Generator iteration pipeline with k-shot.")
    parser.add_argument('--NCBI_train', type=str, default='',
                        help='Path to the NCBI_train.')
    parser.add_argument('--NCBI_kshot', type=str, default='',
                        help='Path to the NCBI subset (10, 20 or 50 pct) to use as k-shot pool.')
    parser.add_argument('--Generated_train', type=str, default='', #TODO: možda cijeli generated, ne samo train
                        help='Path to the Generated_train.')
    parser.add_argument('--output_directory', type=str, required=True, 
                        help='Directory where sample sentences are outputed using JSON format.')
    parser.add_argument('--server_url', type=str, default="http://127.0.0.1:8080", 
                        help='URL of the llama.cpp inference server (default: http://127.0.0.1:8080).')
    parser.add_argument('--num_sentences', type=int, default=1, 
                        help='Number of sentences to produce by LLM.')
    parser.add_argument('--system_prompt_key', type=str, default='generation', 
                        help="Key for selecting system prompt from predefined prompt templates (default: 'generation').")
    parser.add_argument('--temperature', type=float, default=0.2, 
                        help='Sampling temperature for the LLM. (default: 0.2).')
    parser.add_argument('--max_tokens', type=int, default=2000, 
                        help='Maximum number of tokens to generate in LLM response (default: 2000).')
    parser.add_argument('--verbose', action='store_true', 
                        help='If set, print detailed debug output including LLM responses.')
    parser.add_argument('--use_context', action='store_true', 
                        help='If set, include context of the ontology term')    
    parser.add_argument('--obo_file_path', type=str, default='', 
                        help='Directory where sample sentences are outputed using JSON format.')
    parser.add_argument('--reprocess', action='store_true', 
                        help="""If set, recalculate the generation for allready existing 
                        sentences using terms.""")
    parser.add_argument('--test', action='store_true', 
                        help="""If set, recalculate the generation for allready existing 
                        sentences using terms.""")
    parser.add_argument('--spacy_model', type=str, default='en_core_web_sm', 
                        help='spaCy model to use for tokenization (default: en_core_web_sm).')
    parser.add_argument('--kshot_size', type=int, default=3, 
                        help='Number of examples to use for each prompt.')
    parser.add_argument('--random_seed', type=int, default=42, 
                        help='Random seed for reproducibility.')
    parser.add_argument('--include_pos', action='store_true', 
                        help='Include POS tags in k-shot examples.')
    parser.add_argument('--include_dep', action='store_true', 
                        help='Include dependency tags in k-shot examples.')
    parser.add_argument('--no_entity_ratio', type=float, default=0.25, #TODO ovo izračunati iz NCBI traina
                        help='Ratio of sentences without entities.')
    parser.add_argument('--n_clusters', type=int, default=5,
                        help='Number of clusters for KMeans clustering of syntax embeddings.')
    parser.add_argument('--min_samples_per_cluster', type=int, default=10,
                        help='Minimum number of synthetic samples required per cluster to consider it covered.')
    parser.add_argument('--weighted_threshold', type=float, default=0.75,
                        help='Weighted coverage threshold to stop iterations (global overlap).')
    parser.add_argument('--max_iterations', type=int, default=5,
                        help='Maximum number of iterations for adaptive generation.')
    parser.add_argument('--cluster_dir', type=str, default='',
                        help='Directory containing precomputed cluster centroids and labels.')
    parser.add_argument('--overlap_threshold', type=float, default=0.75, help='Wanted cluster overlap treshold per cluster.')


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

def get_cluster_specific_kshot(ncbi_clustered_path, uncovered_clusters, kshot_size=5):
    """Return a list of NCBI examples sampled from uncovered clusters."""
    with open(ncbi_clustered_path, "r") as f:
        data = [json.loads(line) for line in f]
    filtered = [ex for ex in data if ex.get("cluster_id") in uncovered_clusters]
    
    if not filtered:
        print("[WARN] No NCBI examples found for uncovered clusters, using random fallback.")
        filtered = data
    
    np.random.shuffle(filtered)
    return filtered[:kshot_size]


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
    output_dir = args.output_directory
    #TODO:json graph embeddings hiperparameters
    # Precompute real embeddings once
    print("[INFO] Building real NCBI dependency graphs...")
    # KARATE ENV
    real_graphs, _ = pe.build_dependency_graphs(real_data)
    real_emb, model = pe.get_graph_embedding(real_graphs) #TODO ovdje podesiti parametre za gl2vec model

    print(f"[INFO] Clustering real embeddings into {args.n_clusters} syntax clusters...") 
    cluster_dir = args.cluster_dir
    if not os.path.exists(cluster_dir):
        args.parsed_features = os.path.join(output_dir, "syntax_features.jsonl")
        args.output_dir = os.path.join(output_dir, "kmeans_clusters")   
        args.k_min = 2
        args.k_max = 10
        kmeans_params.main(args)  # Call the kmeans_params script to compute clusters

    real_labels = np.load(os.path.join(cluster_dir, "cluster_labels.npy"))
    centroids_real = np.load(os.path.join(cluster_dir, "cluster_centroids.npy"))

    # Save NCBI examples with cluster labels for later k-shot selection
    # KARATE ENV
    ncbi_clustered_path = os.path.join(output_dir, "kshot_ncbi_clustered.jsonl")
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

    while  iteration < args.max_iterations:
        print(f"\n[ITERATION {iteration}] Computing synthetic embeddings...")
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
            output_dir,
            args.min_samples_per_cluster
        )

        print(f"[INFO] Per-cluster overlaps: {[round(x, 3) for x in cluster_overlaps]}")
        print(f"[INFO] Weighted coverage = {weighted_coverage:.3f}")
        print(f"[INFO] Uncovered clusters: {uncovered_clusters}")

        # Stopping condition
        if weighted_coverage >= args.weighted_threshold and not uncovered_clusters:
            print(
                f"[STOP] Coverage target reached: {weighted_coverage:.3f} "
                f"(all clusters sufficiently represented)."
            )
            break

        # Adaptive regeneration: guided by uncovered clusters
        regen_terms = []

        for cluster_id in uncovered_clusters:
            # Determine regeneration strength (0–1): the lower the overlap, the higher the regen weight
            regen_weight = 1.0 - cluster_overlaps[cluster_id]

            # Determine how many new samples to generate for this cluster
            # TODO jel nam ovo ok?
            # You can tune scaling constant (args.min_samples_per_cluster acts as base target)
            num_new = max(1, int(regen_weight * args.min_samples_per_cluster))

            # Collect synthetic terms already assigned to this cluster
            cluster_mask = synth_labels == cluster_id
            indices_in_cluster = [i for i in range(len(synth_data)) if cluster_mask[i] and "term" in synth_data[i]] 
            cluster_terms = [synth_data[i].get("term") for i in indices_in_cluster]

            # If not enough terms in this cluster, sample some from the global synthetic pool as backup
            if len(cluster_terms) < num_new:
                extra_terms = [synth_data[i].get("term") for i in np.random.choice(range(len(synth_data)), num_new - len(cluster_terms), replace=False) 
                               if "term" in synth_data[i]]
                cluster_terms.extend(extra_terms)

            # Randomly pick terms for this cluster’s regeneration quota
            selected_terms = np.random.choice(cluster_terms, size=num_new, replace=False)
            regen_terms.extend(selected_terms)

        print(f"[INFO] Total new terms to regenerate across clusters: {len(regen_terms)}")

        # Select cluster-specific k-shot examples for uncovered clusters
        kshot_examples = get_cluster_specific_kshot(
            os.path.join(output_dir, "kshot_ncbi_clustered.jsonl"),
            uncovered_clusters,
            kshot_size=args.kshot_size
        )

        # Save temporarily for prompt conditioning
        kshot_file = os.path.join(output_dir, f"kshot_iter_{iteration}.jsonl")
        with open(kshot_file, "w") as f:
            for ex in kshot_examples:
                f.write(json.dumps(ex) + "\n")

        # Generate new samples from uncovered clusters using LLM
        iter_output = os.path.join(args.output_directory, f"iteration_{iteration}")
        os.makedirs(iter_output, exist_ok=True)
        # SPACY ENV
        if args.test:
            regen_terms = regen_terms[:3]
        new_path = generate_sentence_samples(args, kshot_file, regen_terms, method="a")

        #TODO new, check if it works  (SPACY_ENV)
        postprocessed = os.path.join(iter_output, f"syntax_features_iter_{iteration}.jsonl")
        subprocess.run([
            "~/conda/bin/python3", "generation_postprocessing.py",
            "--generated", new_path,
            "--generated_postprocessed", postprocessed,
            "--starting_id", str(len(synth_data) + 1),
            "--spacy_model", args.spacy_model
        ])

        '''if os.path.exists(new_path):
            with open(new_path, "r") as f:
                new_sentences = [json.loads(line) for line in f]
        else:
            print("[WARN] No new synthetic data found, stopping.")
            break
        # parse new sentences for syntax features
        new_ids = list(range(len(synth_data), len(synth_data) + len(new_sentences)))
        new_texts = [entry["sentence"] for entry in new_sentences]
        new_entities = [entry.get("entities", []) for entry in new_sentences]
        new_labels = ["generated_train"] * len(new_sentences)

        parser.extract_syntax_features(
            SPACY_NLP,
            new_ids,
            new_texts,
            new_entities,
            new_labels,
            os.path.join(iter_output, f"syntax_features_iter_{iteration}.json")
        )'''

        with open(postprocessed, "r") as f:
            newly_parsed_sentences = [json.loads(line) for line in f]

        if cluster_terms: #remove sentences form this cluster whose terms were selected for regeneration
            terms_to_replace = set(cluster_terms)
            indices_to_replace = [i for i, entry in enumerate(synth_data)
                                if entry.get("term") in terms_to_replace
                                and synth_labels[i] == cluster_id] 
            # Remove in reverse order to preserve indexing
            for idx in sorted(indices_to_replace, reverse=True):
                del synth_data[idx]

        # Add new regenerated ones
        synth_data.extend(newly_parsed_sentences)
        print(f"[INFO] Added {len(newly_parsed_sentences)} parsed sentences to synthetic corpus.")

        iteration += 1
        
    #TODO potencijalno problem jer se cluster_overlap i to definira u while petlji
    visualize_embeddings_clusterwise(real_emb, real_labels, synth_emb, synth_labels, cluster_overlaps, output_dir)

    return synth_data, weighted_coverage

def load_and_parse_data(ncbi_path: str, gen_path: str, output_path_features: str, model: Any):
    quadruple = parser.load_corpuses(ncbi_path, gen_path)
    ids, sentences, entities, corpus_labels = zip(*quadruple)
    print(f"Loaded {len(sentences)} sentences: "
        f"{corpus_labels.count('NCBI_train')} from NCBI_train and "
        f"{corpus_labels.count('generated_train')} from Generated.")
    
    parser.extract_syntax_features(
        SPACY_NLP,
        ids, 
        sentences,
        entities,
        corpus_labels,
        output_path_features
    )

def main(args: argparse.Namespace):    
    ''' 
    iterative generation pipeline (LLM generation -(parsing)-> UMAP/clustering -> condition 
    (while % uncovered NCBI specific clusters) -(k-shot on extracted NCBI specific cluster)-> LLM generation
    '''
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    parsed_data_path = os.path.join(args.output_directory, "syntax_features.jsonl")

    # Load and parse data
    # SPACY ENV
    load_and_parse_data(
        args.NCBI_train,
        args.Generated_train,
        parsed_data_path,
        SPACY_NLP)
    
    with open(parsed_data_path, "r") as f:
        all_data = [json.loads(line) for line in f] 
    ncbi_data = [entry for entry in all_data if entry.get("corpus") == "NCBI_train"]
    synth_data = [entry for entry in all_data if entry.get("corpus") == "generated_train"]

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
python3 bioNER/generation_pipeline.py \
    --NCBI_train data/NCBI-Disease/ \
    --NCBI_kshot data/NCBI-Disease/lg/ncbi_ner_train_10pct.json \
    --Generated_train data/NCBI-Disease/synthetic_10_trial/generated_10_3000.josn \
    --output_directory data/generation_pipeline/ \
    --server_url http://172.20.0.2:8484 \
    --num_sentences 3
    --system_prompt_key generation \
    --temperature 0 \
    --max_tokens 500 \
    --input_file_type list\
    --spacy_model en_core_web_lg \
    --kshot_size 3 \
    --embedding_hiperparameters bioNER/experiments/gl2vec_hiperparams.json \
    --min_samples_per_cluster 10 \
    --weighted_threshold 0.75 \
    --max_iterations 5 \
    --test \
    --cluster_dir data/generation_pipeline/kmeans_clusters \
    --overlap_threshold 0.75
    
    --server_url http://172.17.0.1:8484 \
'''

if __name__ == "__main__":
    args = argparse_args()
    SPACY_NLP = utils.get_spacy_model(args.spacy_model)
    args.logger = setup_logger(args)
    args.logger.info(f"Output directory: {args.output_directory}")
    main(args)