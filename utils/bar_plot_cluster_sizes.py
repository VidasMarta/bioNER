"""
compare_clusters.py

Builds a hierarchical clustering dendrogram from a base JSONL distance-matrix
file and compares the resulting cluster distribution against one or more
synthetic .npz distance matrices, producing a normalised bar-plot.

Usage
-----
python compare_clusters.py \\
    --base /home/mkeber/syn-bioner/data/processed/ncbi/trf/distance_matrices/syntax_features_sent_tree_head_D.jsonl \\
    --synth /path/to/distance_matrix_1.npz /path/to/distance_matrix_2.npz \\
    [--p 30] [--strategy mean]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')          # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from tqdm import tqdm
from .distance_matrix import DistanceMatrixGenerator

# ── helpers ──────────────────────────────────────────────────────────────────

def get_cluster_members(Z, n_samples):
    """Return a dict mapping every linkage node → list of original sample indices."""
    cluster_members = {i: [i] for i in range(n_samples)}
    for i, (c1, c2, _, _) in enumerate(Z):
        c1, c2 = int(c1), int(c2)
        cluster_members[n_samples + i] = cluster_members[c1] + cluster_members[c2]
    return cluster_members


def assign_new_samples(new_sample_distances, cluster_members, leaves, strategy='mean'):
    """
    Assign a single new sample to the nearest leaf cluster.

    Parameters
    ----------
    new_sample_distances : 1-D array, shape (n_original,)
        Distances from the new sample to every original sample.
    cluster_members : dict  {node_idx: [original_sample_indices]}
    leaves : list of int  — the p leaf node indices from the dendrogram
    strategy : 'mean' | 'min' | 'max'

    Returns
    -------
    assigned_cluster : int
    cluster_distances : dict {leaf_idx: float}
    """
    agg = {'mean': np.mean, 'min': np.min, 'max': np.max}[strategy]
    cluster_distances = {
        idx: agg(new_sample_distances[cluster_members[idx]])
        for idx in leaves
    }
    assigned_cluster = min(cluster_distances, key=cluster_distances.get)
    return assigned_cluster, cluster_distances


# ── core function ─────────────────────────────────────────────────────────────
def load_npz_path(path):
    if path.endswith('.npz'):
        raw_dist, sid = DistanceMatrixGenerator.load_distance_matrix(path)
    else:
        # Find the npz file in the path assuming the path is directory
        npz_files = [f for f in os.listdir(path) if f.endswith('.npz')]
        if not npz_files:
            raise FileNotFoundError(f"No .npz file found in directory {path}")
        npz_path = os.path.join(path, npz_files[0])
        raw_dist, sid = DistanceMatrixGenerator.load_distance_matrix(npz_path)
    return raw_dist, sid

def make_comparison_original_synth(
    dist_path,
    leaves,
    leaf_label_dict,
    cluster_members,
    n_words,
    out_dir,
    strategy='mean',
    png_name='normalized_bar_plot.png',
    original_data_name='train',
):
    """
    Parameters
    ----------
    dist_path : str | list[str]
        Path(s) to .npz files, each containing 'distance_matrix' and
        'sentence_ids'.
    leaves, leaf_label_dict, cluster_members, n_words
        Built from the base JSONL file (see build_base_structures).
    out_dir : str
        Directory where normalised_bar_plot.png is written.
    strategy : str
        Aggregation strategy for cluster assignment.
    """
    if not isinstance(dist_path, list):
        dist_path = [dist_path]

    # ── load & orient every distance matrix ──────────────────────────────────
    all_distance_matrices = {}
    all_sentence_ids      = {}

    for path in dist_path: # shape: (n_original, m_new) or (m_new, n_original)
        raw_dist, sid = load_npz_path(path)
        n_original = len(n_words)
        if raw_dist.shape[0] == n_original:       # rows are original → transpose
            dm = np.log(raw_dist.T + 1)           # → (m_new, n_original)
        elif raw_dist.shape[1] == n_original:     # columns are original → keep
            dm = np.log(raw_dist + 1)             # → (m_new, n_original)
        else:
            raise ValueError(
                f"[{path}] Neither dimension matches n_original={n_original}, "
                f"got shape {raw_dist.shape}"
            )

        all_distance_matrices[path] = dm
        all_sentence_ids[path]      = sid
        print(f"Loaded {os.path.basename(path)}: distance matrix shape {dm.shape}")

    # ── baseline counts ───────────────────────────────────────────────────────
    before_counts = {idx: len(cluster_members[idx]) for idx in leaves}
    before_total  = sum(before_counts.values())

    # ── per-path assignment & counts ─────────────────────────────────────────
    path_after_counts = {}
    path_totals       = {}
    number_of_samples = []
    for path, dm in all_distance_matrices.items():
        assignments = []
        number_of_samples.append((dm.shape[0], dm.shape[1]))

        for i, row in tqdm(
            enumerate(dm),
            total=dm.shape[0],
            desc=os.path.basename(path),
        ):
            assigned, _ = assign_new_samples(row, cluster_members, leaves, strategy=strategy)
            assignments.append({'new_sample_idx': i, 'assigned_cluster': assigned})

        after_counts = before_counts.copy()
        for a in assignments:
            after_counts[a['assigned_cluster']] += 1

        path_after_counts[path] = after_counts
        path_totals[path]       = sum(after_counts.values())

    # ── print comparison tables ───────────────────────────────────────────────
    for path, after_counts in path_after_counts.items():
        comparison = pd.DataFrame({
            'cluster_label': [leaf_label_dict[idx] for idx in leaves],
            'before':        [before_counts[idx]   for idx in leaves],
            'after':         [after_counts[idx]    for idx in leaves],
            'added':         [after_counts[idx] - before_counts[idx] for idx in leaves],
        }).sort_values('added', ascending=False)
        print(f"\n── {os.path.basename(path)} ──")
        print(comparison.to_string(index=False))

    # ── normalised bar plot ───────────────────────────────────────────────────
    n_paths   = len(dist_path)
    n_groups  = len(leaves)
    total_bars = 1 + n_paths          # original + one bar per synthetic file
    width      = 0.8 / total_bars
    x          = np.arange(n_groups)
    labels     = [leaf_label_dict[idx] for idx in leaves]

    fig, ax = plt.subplots(figsize=(16, 6))

    # Original bars (normalised by their own total)
    before_norm = [before_counts[idx] / before_total for idx in leaves]
    ax.bar(
        x + (-total_bars / 2 + 0.5) * width,
        before_norm,
        width,
        label=f'{original_data_name} {number_of_samples[0][1]}',
        color='steelblue',
    )

    colors = plt.cm.tab10(np.linspace(0.1, 0.9, n_paths))
    for i, (path, after_counts) in enumerate(path_after_counts.items()):
        after_norm = [after_counts[idx] / path_totals[path] for idx in leaves]
        offset = (-total_bars / 2 + 1.5 + i) * width
        ax.bar(
            x + offset,
            after_norm,
            width,
            label=f'{os.path.basename(os.path.dirname(path)) if path.endswith(".npz") else os.path.basename(path)}: {number_of_samples[i][0]}',
            color=colors[i],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=9)
    ax.set_ylabel('Fraction of sentences (normalised)')
    ax.set_title('Cluster distribution: original vs synthetic sets')
    ax.legend(fontsize=8)
    plt.tight_layout()
    out_dir = out_dir if len(dist_path) > 1 else os.path.dirname(dist_path[0])
    out_path = os.path.join(out_dir, png_name)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"\nSaved bar plot → {out_path}")


# ── build base structures from JSONL ─────────────────────────────────────────

def build_base_structures(jsonl_path, distance_matrix_path, p=30):
    """
    Read the base JSONL file, build the distance matrix, run hierarchical
    clustering, draw + save the dendrogram, and return everything needed for
    cluster assignment.

    Returns
    -------
    leaves, leaf_label_dict, cluster_members, n_words, Z, out_dir
    """
    out_dir  = os.path.dirname(os.path.abspath(distance_matrix_path))
    basename = os.path.splitext(os.path.basename(jsonl_path))[0]

    # ── load JSONL ────────────────────────────────────────────────────────────
    records = []
    n_words = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            records.append(data)
            n_words.append(data['n_words'])

    n_words = np.array(n_words)
    distance_matrices = dict()
    
    sample, sentence_ids = load_npz_path(distance_matrix_path)
    # df = pd.read_csv(distance_matrix_path, index_col=0)
    # sample = df.values
    sample = np.log(sample + 1)

    print(f"Distance matrix shape: {sample.shape}")
    # ── hierarchical clustering ───────────────────────────────────────────────

    condensed = sch.distance.squareform(sample)
    Z = sch.linkage(condensed, method='average')

    n_samples = len(n_words)
    ddata     = sch.dendrogram(Z, truncate_mode='lastp', p=p, no_plot=True)
    leaves    = ddata['leaves']          # length == p

    cluster_members = get_cluster_members(Z, n_samples)

    leaf_label_dict = {
        idx: (
            f"{len(cluster_members[idx])}: "
            f"{np.mean(n_words[cluster_members[idx]]):.1f}"
            f"±{np.std(n_words[cluster_members[idx]]):.1f}"
        )
        for idx in leaves
    }

    # ── save dendrogram ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    sch.dendrogram(
        Z,
        truncate_mode='lastp',
        p=p,
        leaf_rotation=60.0,
        leaf_font_size=12.0,
        leaf_label_func=lambda idx: leaf_label_dict.get(idx, str(idx)),
        show_contracted=True,
        ax=ax,
    )
    ax.set_title(f'Dendrogram — {basename}')
    plt.tight_layout()

    dendro_path = os.path.join(out_dir, f'{basename}.png')
    fig.savefig(dendro_path, dpi=150)
    plt.close(fig)
    print(f"Saved dendrogram → {dendro_path}")
    png_name = f'bar_plot_{basename}.png' 
    return leaves, leaf_label_dict, cluster_members, n_words, Z, out_dir, png_name


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description=(
            'Build cluster dendrogram from a base JSONL file and compare '
            'cluster distributions against synthetic .npz distance matrices.'
        )
    )
    parser.add_argument(
        '--base',
        required=True,
        metavar='JSONL_PATH',
        help=(
            'Path to the base JSONL file used to build leaves / cluster labels. '
            'Example: syntax_features_sent_tree_head_D.jsonl'
        ),
    )
    parser.add_argument(
        '--dist',
        required=True,
        metavar='CSV_PATH',
        help=(
            'Path to the base distance matrix file used to build leaves / cluster labels. '
            'Example: /home/mkeber/syn-bioner/data/processed/ncbi/trf/distance_matrices/distance_matrix_simple_D.csv'
        ),
    )
    parser.add_argument(
        '--synth',
        required=True,
        nargs='+',
        metavar='NPZ_PATH',
        help=(
            'One or more .npz distance matrix files to compare against the '
            'original clustering. Each must contain "distance_matrix" and '
            '"sentence_ids" arrays.'
        ),
    )
    parser.add_argument(
        '--p',
        type=int,
        default=30,
        metavar='INT',
        help='Number of leaf clusters for the truncated dendrogram (default: 30).',
    )
    parser.add_argument(
        '--strategy',
        choices=['mean', 'min', 'max'],
        default='mean',
        help='Aggregation strategy for cluster assignment (default: mean).',
    )
    parser.add_argument(
        '--output_path',
        help='Path to the output directory where plots will be saved.',
    )
    return parser.parse_args(args)

"""
python3 -m bar_plot_cluster_sizes --synth /home/mkeber/syn-bioner/data/synthetic/init_syn_generation_all_new/distance_matrix_nm.npz /home/mkeber/syn-bioner/data/synthetic/kshot_syn_generation_10pct/distance_matrix_nm.npz \
    --p 30 --strategy mean --base /home/mkeber/syn-bioner/data/processed/ncbi/trf/syntax_features_sent_tree_head_D.jsonl \
    --dist /home/mkeber/syn-bioner/data/processed/ncbi/trf/distance_matrices/distance_matrix_simple_D.csv
"""

def main(args=None):
    args = parse_args(args)

    print(f"Base JSONL : {args.base}")
    print(f"Distance matrix path: {args.dist}")
    print(f"Synth files: {args.synth}")
    print(f"p={args.p}, strategy={args.strategy}\n")
    print(f"Output path: {args.output_path}\n")

    leaves, leaf_label_dict, cluster_members, n_words, Z, out_dir, png_name = \
        build_base_structures(args.base, args.dist, p=args.p)
    original_data_name = os.path.basename(args.base)

    make_comparison_original_synth(
        dist_path=args.synth,
        leaves=leaves,
        leaf_label_dict=leaf_label_dict,
        cluster_members=cluster_members,
        n_words=n_words,
        out_dir=out_dir,
        strategy=args.strategy,
        png_name=png_name,
        original_data_name=original_data_name

    )


if __name__ == '__main__':
    main()