"""
Distance Matrix Generator from D (Actual Distance) Values

Calculate distances between sentences based on their syntactic complexity (D).
Load results from JSONL and generate distance matrices.

Distance functions:
1. Simple D difference: |D_sent1 - D_sent2|
2. Normalized by word count: |D_sent1/n_words_sent1 - D_sent2/n_words_sent2|
3. Custom distance metric
"""

import json
from turtle import distance
import numpy as np
from typing import List, Dict, Callable, Tuple, Optional
from pathlib import Path
import pandas as pd
import random
import argparse
import os
from . import actual_D
from . import bar_plot_cluster_sizes


class SentenceDistance:
    """Calculate distance between sentences based on syntactic features"""
    
    @staticmethod
    def simple_D_distance(d1: float, d2: float) -> float:
        """
        Simple absolute difference between D values
        
        Args:
            d1: D value of sentence 1
            d2: D value of sentence 2
            
        Returns:
            Distance: |d1 - d2|
        """
        return abs(d1 - d2)
    

    @staticmethod
    def normalized_D_distance(d1: float, n_words1: int, 
                             d2: float, n_words2: int) -> float:
        """
        D normalized by number of words (average distance per word)
        
        Args:
            d1: D value of sentence 1
            n_words1: Number of words in sentence 1
            d2: D value of sentence 2
            n_words2: Number of words in sentence 2
            
        Returns:
            Distance: |D1/n_words1 - D2/n_words2|
        """
        if n_words1 == 0 or n_words2 == 0:
            return float('inf')
        
        norm_d1 = d1 / n_words1
        norm_d2 = d2 / n_words2
        
        return abs(norm_d1 - norm_d2)
    
    @staticmethod
    def avg_distance_difference(avg_dist1: float, avg_dist2: float) -> float:
        """
        Distance based on average edge distance (D / n_edges)
        
        Args:
            avg_dist1: Average distance per edge in sentence 1
            avg_dist2: Average distance per edge in sentence 2
            
        Returns:
            Distance: |avg_dist1 - avg_dist2|
        """
        return abs(avg_dist1 - avg_dist2)


class DistanceMatrixGenerator:
    """Generate distance matrices from sentence D values"""
    
    @staticmethod
    def load_results_from_jsonl(jsonl_path: str) -> List[Dict]:
        """
        Load results from JSONL file
        
        Args:
            jsonl_path: Path to JSONL file with D calculation results
            
        Returns:
            List of sentence result dictionaries
        """
        results = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                results.append(json.loads(line))
        return results
    
    @staticmethod
    def create_distance_matrix_nm(resultsn: List[Dict], resultsm: List[Dict],
                              distance_func: Callable = None,
                              verbose: bool = True) -> Tuple[np.ndarray, List[int]]:
        """
        Create distance matrix from results
        
        Args:
            results: List of sentence result dicts with D, n_words, avg_distance
            distance_func: Distance function to use (default: simple_D_distance)
            verbose: Print progress
            
        Returns:
            Tuple of (distance_matrix, sentence_ids)
        """
        if distance_func is None:
            distance_func = SentenceDistance.simple_D_distance
        
        n = len(resultsn)
        m = len(resultsm)
        distance_matrix = np.zeros((n, m))
        sentence_ids = [r['id'] for r in resultsn]
        
        if verbose:
            print(f"Creating distance matrix for {n} sentences...")
        
        for i in range(n):
            for j in range(m):
                if distance_func == SentenceDistance.simple_D_distance:
                    # Simple D difference
                    dist = distance_func(resultsn[i]['D'], resultsm[j]['D'])
                distance_matrix[i, j] = dist
                
            if verbose and (i + 1) % max(1, n // 10) == 0:
                print(f"  Processed {i + 1}/{n} sentences")
            
        if verbose:
            print(f"✓ Distance matrix created: {n}x{m}")
        
        return distance_matrix, sentence_ids

    @staticmethod
    def create_distance_matrix(results: List[Dict],
                              distance_func: Callable = None,
                              verbose: bool = True) -> Tuple[np.ndarray, List[int]]:
        """
        Create distance matrix from results
        
        Args:
            results: List of sentence result dicts with D, n_words, avg_distance
            distance_func: Distance function to use (default: simple_D_distance)
            verbose: Print progress
            
        Returns:
            Tuple of (distance_matrix, sentence_ids)
        """
        if distance_func is None:
            distance_func = SentenceDistance.simple_D_distance
        
        n = len(results)
        distance_matrix = np.zeros((n, n))
        sentence_ids = [r.get('id', i) for i, r in enumerate(results)]
        
        if verbose:
            print(f"Creating distance matrix for {n} sentences...")
        
        for i in range(n):
            for j in range(i, n):
                if distance_func == SentenceDistance.simple_D_distance:
                    # Simple D difference
                    dist = distance_func(results[i]['D'], results[j]['D'])
                
                elif distance_func == SentenceDistance.normalized_D_distance:
                    # Normalized D
                    dist = distance_func(
                        results[i]['D'], results[i]['n_words'],
                        results[j]['D'], results[j]['n_words']
                    )
                
                elif distance_func == SentenceDistance.avg_distance_difference:
                    # Average distance
                    dist = distance_func(
                        results[i]['avg_distance'],
                        results[j]['avg_distance']
                    )
                
                else:
                    # Generic function call
                    dist = distance_func(results[i], results[j])
                
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist  # Symmetric
            
            if verbose and (i + 1) % max(1, n // 10) == 0:
                print(f"  Processed {i + 1}/{n} sentences")
        
        if verbose:
            print(f"✓ Distance matrix created: {n}x{n}")
        
        return distance_matrix, sentence_ids
    
    @staticmethod
    def save_distance_matrix(distance_matrix: np.ndarray,
                            sentence_ids: List[int],
                            output_path: str,
                            format: str = 'csv') -> None:
        """
        Save distance matrix to file
        
        Args:
            distance_matrix: The distance matrix
            sentence_ids: List of sentence IDs
            output_path: Where to save
            format: 'csv', 'npy', or 'txt'
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            # Save as CSV with row/column labels
            df = pd.DataFrame(
                distance_matrix,
                index=sentence_ids,
                columns=sentence_ids
            )
            df.to_csv(output_path)
            print(f"✓ Saved to {output_path} (CSV format)")
        
        elif format == 'npy':
            # NumPy binary format
            np.save(output_path, distance_matrix)
            print(f"✓ Saved to {output_path} (NumPy binary format)")
        
        elif format == 'txt':
            # Plain text format
            np.savetxt(output_path, distance_matrix, delimiter=',')
            print(f"✓ Saved to {output_path} (Text format)")
        
        elif format == 'npz':
            # Compressed NumPy format (can include sentence_ids as metadata)
            np.savez_compressed(output_path, distance_matrix=distance_matrix, sentence_ids=sentence_ids)
            print(f"✓ Saved to {output_path} (Compressed NumPy format)")
            
    @staticmethod
    def load_distance_matrix(input_path: str) -> Tuple[np.ndarray, List[int]]:
        """
        Load distance matrix from file. Format is inferred from file extension.

        Args:
            input_path: Path to the saved distance matrix file.
                        Supported extensions: .csv, .npy, .txt, .npz

        Returns:
            Tuple of (distance_matrix: np.ndarray, sentence_ids: List[int])
            Note: .npy and .txt formats do not store sentence_ids —
                a 0-based integer range is returned in that case.
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Distance matrix file not found: {input_path}")

        suffix = input_path.suffix.lower()

        if suffix == '.csv':
            df = pd.read_csv(input_path, index_col=0)
            sentence_ids = list(df.index.astype(int))
            distance_matrix = df.values.astype(float)

        elif suffix == '.npy':
            distance_matrix = np.load(input_path)
            sentence_ids = list(range(len(distance_matrix)))  # no metadata stored

        elif suffix == '.txt':
            distance_matrix = np.loadtxt(input_path, delimiter=',')
            sentence_ids = list(range(len(distance_matrix)))  # no metadata stored

        elif suffix == '.npz':
            data = np.load(input_path, allow_pickle=True)
            if 'distance_matrix' not in data:
                raise KeyError(f"'distance_matrix' array not found in {input_path}. "
                            f"Available keys: {list(data.keys())}")
            distance_matrix = data['distance_matrix']
            sentence_ids = list(data['sentence_ids']) if 'sentence_ids' in data \
                        else list(range(len(distance_matrix)))

        else:
            raise ValueError(f"Unsupported file extension '{suffix}'. "
                            f"Expected one of: .csv, .npy, .txt, .npz")

        print(f"✓ Loaded from {input_path} — "
            f"shape {distance_matrix.shape}, {len(sentence_ids)} sentence IDs")

        return distance_matrix, sentence_ids
    

    @staticmethod
    def print_distance_matrix(distance_matrix: np.ndarray,
                             sentence_ids: List[int],
                             max_display: int = 10) -> None:
        """
        Print distance matrix in readable format
        
        Args:
            distance_matrix: The distance matrix
            sentence_ids: List of sentence IDs
            max_display: Max rows/cols to show (0 for all)
        """
        n = len(distance_matrix)
        
        if max_display == 0 or n <= max_display:
            # Show all
            display_matrix = distance_matrix
            display_ids = sentence_ids
        else:
            # Show subset
            display_matrix = distance_matrix[:max_display, :max_display]
            display_ids = sentence_ids[:max_display]
        
        # Create DataFrame for nice display
        df = pd.DataFrame(
            display_matrix,
            index=display_ids,
            columns=display_ids
        )
        
        print("\nDistance Matrix:")
        print("=" * (len(display_ids) * 12 + 10))
        print(df.to_string(float_format=lambda x: f'{x:7.2f}'))
        print("=" * (len(display_ids) * 12 + 10))
        
        if n > max_display:
            print(f"\n(Showing first {max_display} of {n} sentences)")


class DistanceAnalyzer:
    """Analyze distance matrix and find similar/dissimilar sentences"""
    
    @staticmethod
    def find_similar_sentences(results: List[Dict],
                              distance_matrix: np.ndarray,
                              query_idx: int,
                              n_similar: int = 5) -> List[Tuple[int, float, str]]:
        """
        Find most similar sentences to a query sentence
        
        Args:
            results: Original sentence results
            distance_matrix: Distance matrix
            query_idx: Index of query sentence
            n_similar: Number of similar sentences to return
            
        Returns:
            List of (sentence_id, distance, sentence_text) tuples
        """
        distances = distance_matrix[query_idx]
        
        # Sort by distance (ascending), excluding the sentence itself
        sorted_indices = np.argsort(distances)
        sorted_indices = sorted_indices[1:]  # Skip distance to itself (0)
        
        similar = []
        for idx in sorted_indices[:n_similar]:
            similar.append((
                results[idx]['id'],
                distances[idx],
                results[idx]['sentence']
            ))
        
        return similar
    
    @staticmethod
    def find_dissimilar_sentences(results: List[Dict],
                                 distance_matrix: np.ndarray,
                                 query_idx: int,
                                 n_dissimilar: int = 5) -> List[Tuple[int, float, str]]:
        """
        Find most dissimilar sentences to a query sentence
        
        Args:
            results: Original sentence results
            distance_matrix: Distance matrix
            query_idx: Index of query sentence
            n_dissimilar: Number of dissimilar sentences to return
            
        Returns:
            List of (sentence_id, distance, sentence_text) tuples
        """
        distances = distance_matrix[query_idx]
        
        # Sort by distance (descending)
        sorted_indices = np.argsort(-distances)
        
        dissimilar = []
        for idx in sorted_indices[:n_dissimilar]:
            dissimilar.append((
                results[idx]['id'],
                distances[idx],
                results[idx]['sentence']
            ))
        
        return dissimilar
    
    @staticmethod
    def print_similar_analysis(results: List[Dict],
                              distance_matrix: np.ndarray,
                              query_idx: int) -> None:
        """Print analysis of similar and dissimilar sentences"""
        query = results[query_idx]
        
        print("\n" + "=" * 80)
        print(f"Query Sentence (ID {query['id']}):")
        print(f"  {query['sentence']}")
        print(f"  D={query['D']}, n_words={query['n_words']}, avg_distance={query['avg_distance']:.2f}")
        print("=" * 80)
        
        # Similar sentences
        print("\nMost Similar Sentences:")
        similar = DistanceAnalyzer.find_similar_sentences(results, distance_matrix, query_idx, 5)
        for sent_id, dist, text in similar:
            print(f"\n  Distance: {dist:.4f}")
            print(f"  ID: {sent_id}")
            print(f"  Text: {text[:80]}..." if len(text) > 80 else f"  Text: {text}")
        
        # Dissimilar sentences
        print("\n\nMost Dissimilar Sentences:")
        dissimilar = DistanceAnalyzer.find_dissimilar_sentences(results, distance_matrix, query_idx, 5)
        for sent_id, dist, text in dissimilar:
            print(f"\n  Distance: {dist:.4f}")
            print(f"  ID: {sent_id}")
            print(f"  Text: {text[:80]}..." if len(text) > 80 else f"  Text: {text}")
        print("\n" + "=" * 80)
    
    @staticmethod
    def print_statistics(distance_matrix: np.ndarray) -> None:
        """Print statistics about the distance matrix"""
        # Get upper triangle (avoid counting diagonal and duplicates)
        upper_triangle = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
        
        print("\nDistance Matrix Statistics:")
        print("=" * 50)
        print(f"  Minimum distance:  {upper_triangle.min():.4f}")
        print(f"  Maximum distance:  {upper_triangle.max():.4f}")
        print(f"  Mean distance:     {upper_triangle.mean():.4f}")
        print(f"  Median distance:   {np.median(upper_triangle):.4f}")
        print(f"  Std deviation:     {upper_triangle.std():.4f}")
        print("=" * 50)

def plot_hierarchical_clustering(distance_matrix, labels=None, title="Hierarchical Clustering",
                                  method="average", ax=None):
    """
    Plot a dendrogram from a precomputed distance matrix.

    Parameters
    ----------
    distance_matrix : np.ndarray  shape (N, N), symmetric, zero diagonal
    labels          : list of str or None  (sentence IDs or indices)
    title           : str
    method          : linkage method ('average', 'ward', 'complete', 'single')
                      NOTE: 'ward' requires Euclidean distances; use 'average'
                      for arbitrary distance matrices.
    ax              : matplotlib Axes or None (creates new figure if None)
    """
    condensed = sch.distance.squareform(distance_matrix)
    linkage   = sch.linkage(condensed, method=method)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(8, len(labels or []) * 0.6), 5))

    sch.dendrogram(
        linkage,
        labels=labels,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=9,
    )
    ax.set_title(title)
    ax.set_ylabel("Distance")
    return ax

def make_word_count_colormap(word_counts, max_bins=25):
    """
    Assign each graph a bin index and a colour based on its word count,
    using Freedman-Diaconis to choose the number of bins (capped at max_bins).

    Freedman-Diaconis bin width:  h = 2 * IQR * n^(-1/3)
    Falls back to max_bins if IQR = 0 (e.g. all sentences same length).

    Parameters
    ----------
    word_counts : array-like of int   len(g.nodes) for each graph in sample
    max_bins    : int                 upper cap on bin count (default 25)

    Returns
    -------
    bin_indices  : np.ndarray[int]    shape (N,)  bin index for each graph (0-based)
    colors       : np.ndarray         shape (N, 4) RGBA colour per graph
    n_bins       : int                actual number of bins used
    boundaries   : np.ndarray         shape (n_bins+1,) bin edges
    cmap         : matplotlib Colormap
    norm         : matplotlib Normalize
    """
    wc = np.asarray(word_counts, dtype=float)
    n  = len(wc)

    # ── Freedman-Diaconis bin width ──────────────────────────────────────────
    q75, q25 = np.percentile(wc, [75, 25])
    iqr = q75 - q25

    if iqr > 0:
        h = 2.0 * iqr * (n ** (-1.0 / 3.0))
        fd_bins = int(np.ceil((wc.max() - wc.min()) / h))
        n_bins  = max(1, min(fd_bins, max_bins))
    else:
        # All values identical or IQR=0 — fall back to max_bins
        n_bins = max_bins

    boundaries  = np.linspace(wc.min(), wc.max() + 1e-9, n_bins + 1)
    bin_indices = np.digitize(wc, boundaries[1:])          # 0-based bin per graph
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    cmap = plt.cm.get_cmap("viridis", n_bins)
    norm = plt.Normalize(vmin=0, vmax=n_bins - 1)
    colors = cmap(norm(bin_indices))

    return bin_indices, colors, n_bins, boundaries, cmap, norm


def plot_umap_color(distance_matrix, labels=None, title="UMAP", n_neighbors=5,
              min_dist=0.3, ax=None, random_state=42, n_jobs=8,
              word_counts=None, max_bins=25):
    """
    Run UMAP on a precomputed distance matrix and plot the 2-D embedding.

    Parameters
    ----------
    distance_matrix : np.ndarray  shape (N, N)
    labels          : list of str or None
    title           : str
    n_neighbors     : int or None  — defaults to max(2, N // 3), capped at N-1
    min_dist        : float        — UMAP min_dist parameter
    ax              : matplotlib Axes or None
    random_state    : int
    n_jobs          : int
    word_counts     : array-like of int or None
                      len(g.nodes) per graph; when provided dots are coloured
                      by Freedman-Diaconis bins (capped at max_bins)
    max_bins        : int  upper cap on bin count (default 25)
    """
    n = len(distance_matrix)
    if n_neighbors is None:
        n_neighbors = max(2, min(n - 1, n // 3))

    reducer = umap.UMAP(
        metric="precomputed",
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        n_jobs=n_jobs,
    )
    embedding = reducer.fit_transform(distance_matrix)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    # ── Colouring ────────────────────────────────────────────────────────────
    if word_counts is not None:
        _, colors, n_bins, boundaries, cmap, norm = make_word_count_colormap(
            word_counts, max_bins=max_bins
        )
        sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                        c=colors, s=60, alpha=0.8)
        # Colorbar: ticks at bin centres, labelled with word-count range
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Word count (bin)", fontsize=8)
        # Label every other tick to avoid crowding
        tick_step  = max(1, n_bins // 10)
        tick_pos   = np.arange(0, n_bins, tick_step)
        tick_labels = [
            f"{int(boundaries[i])}-{int(boundaries[i+1])}" for i in tick_pos
        ]
        cbar.set_ticks(tick_pos)
        cbar.set_ticklabels(tick_labels, fontsize=7)
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=60, alpha=0.8)

    if labels is not None:
        for i, lbl in enumerate(labels):
            ax.annotate(str(lbl), (embedding[i, 0], embedding[i, 1]),
                        fontsize=8, textcoords="offset points", xytext=(4, 4))

    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    return ax


def plot_hierarchical_clustering_color(distance_matrix, labels=None,
                                  title="Hierarchical Clustering",
                                  method="average", ax=None, truncate_p=None,
                                  word_counts=None, max_bins=25, show_contracted=True):
    """
    Plot a dendrogram from a precomputed distance matrix, with an optional
    colour strip below showing word-count bins for each leaf.

    Parameters
    ----------
    distance_matrix : np.ndarray  shape (N, N), symmetric, zero diagonal
    labels          : list of str or None
    title           : str
    method          : linkage method ('average', 'ward', 'complete', 'single')
                      NOTE: 'ward' requires Euclidean distances; use 'average'
                      for arbitrary distance matrices.
    ax              : matplotlib Axes or None
                      When word_counts is given this ax receives the dendrogram;
                      the colour strip is drawn on a new thin axis created
                      automatically below it.  Pass None to create both.
    truncate_p      : int or None  — collapse to last p merges if set
    word_counts     : array-like of int or None
                      len(g.nodes) per graph in the SAME ORDER as
                      distance_matrix rows; when provided a colour strip is
                      drawn under the dendrogram, aligned to the leaves. Colors based on word count 
                      bined to create a color legend
    max_bins        : int  upper cap on bin count (default 25)
    """
    try:
        import fastcluster
        _linkage_fn = fastcluster.linkage
    except ImportError:
        _linkage_fn = sch.linkage

    condensed = sch.distance.squareform(distance_matrix)
    Z         = _linkage_fn(condensed, method=method)
    n         = len(distance_matrix)
    MAX_WIDTH_IN = 40  # sane upper bound
    # ── Figure / axes setup ─────────────────────────────────────────────────
    if ax is None:
        width = min(
            MAX_WIDTH_IN,
            max(8, (truncate_p or n) * 0.6)
        )
        if word_counts is not None:
            # Two-row layout: tall dendrogram + thin colour strip
            fig, (ax, ax_strip) = plt.subplots(
                nrows=2, ncols=1,
                figsize=(width, 6),
                gridspec_kw={"height_ratios": [10, 1.5]},
            )
            # Create a dedicated colorbar axis below the strip
            strip_pos = ax_strip.get_position()
            cbar_h = 0.025
            ax_cbar = fig.add_axes([
                strip_pos.x0,
                strip_pos.y0 - cbar_h - 0.01,
                strip_pos.width,
                cbar_h
            ])
            fig.subplots_adjust(hspace=0.05)
        else:
            fig, ax = plt.subplots(figsize=(width, 5))
            ax_strip = None
    else:
        # Caller supplied ax — create strip axis manually below it if needed
        if word_counts is not None:
            fig      = ax.get_figure()
            pos      = ax.get_position()          # [x0, y0, w, h] in figure coords
            strip_h  = 0.03                       # fraction of figure height
            ax_strip = fig.add_axes(
                [pos.x0, pos.y0 - strip_h - 0.01, pos.width, strip_h]
            )
            # Create a dedicated colorbar axis below the strip
            strip_pos = ax_strip.get_position()
            cbar_h = 0.025

            ax_cbar = fig.add_axes([
                strip_pos.x0,
                strip_pos.y0 - cbar_h - 0.01,
                strip_pos.width,
                cbar_h
            ])
        else:
            ax_strip = None

    # ── Dendrogram ───────────────────────────────────────────────────────────
    dend_kwargs = dict(ax=ax, leaf_rotation=60, leaf_font_size=12, no_labels=True, )
    if truncate_p is not None:
        dend_kwargs.update(truncate_mode="lastp", p=truncate_p,
                           show_contracted=show_contracted, labels=None, no_labels=True)
    else:
        dend_kwargs["labels"] = labels

    dend = sch.dendrogram(Z, **dend_kwargs)

    ax.set_title(title)
    ax.set_ylabel("Distance")

    # ── Colour strip ─────────────────────────────────────────────────────────
    if word_counts is not None and ax_strip is not None and truncate_p is None:
        _, colors, n_bins, boundaries, cmap, norm = make_word_count_colormap(
            word_counts, max_bins=max_bins
        )
        print('IAM DOOING COLOR LINE.')
        # Leaf order as drawn
        leaf_order = dend["leaves"]

        # Map bins → actual RGBA colors
        leaf_bins = colors[np.array(leaf_order)]
        ordered_colors = cmap(norm(leaf_bins))

        # Leaf x-positions (scipy invariant: 5, 15, 25, ...)
        leaf_xs = np.array(sorted({
            x for icoord in dend["icoord"] for x in icoord
            if x == int(x) and x % 10 == 5
        }))

        # Draw one bar per leaf
        bar_width = 1
        print(leaf_xs[:20])
        print(leaf_xs[-20:])
        
        for lx, col in zip(leaf_xs, ordered_colors):
            ax_strip.bar(
                lx, 10,
                width=bar_width,
                color=col,
                align="center",
                linewidth=0
            )
        print('The length of LEAVES : ', len (ordered_colors))
        print('The length of LEAVES : ', len (leaf_xs))
        

        ax_strip.set_xlim(leaf_xs.min() - 5, leaf_xs.max() + 5)

        ax_strip.set_ylim(0, 1)
        ax_strip.axis("off")

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = plt.colorbar(
            sm,
            cax=ax_cbar,
            orientation="horizontal"
        )

        cbar.set_label("Word count (bin)", fontsize=8)

        tick_step = max(1, n_bins // 10)
        tick_pos = np.arange(0, n_bins, tick_step)
        cbar.set_ticks(tick_pos)
        cbar.set_ticklabels(
            [f"{int(boundaries[i])}-{int(boundaries[i+1])}" for i in tick_pos],
            fontsize=7
        )

    elif word_counts is not None and truncate_p is not None:
        # Truncated dendrograms collapse leaves — colour strip is meaningless
        ax.annotate("Colour strip unavailable with truncate_p",
                    xy=(0.5, -0.08), xycoords="axes fraction",
                    ha="center", fontsize=8, color="grey")

    return ax


def argparse_args():
    parser = argparse.ArgumentParser(description="Dependency Distance Matrix Generator")
    parser.add_argument("--input", help="Input JSONL file")
    parser.add_argument("--input_synth", help="Input synthetic JSONL file")
    parser.add_argument("--output_dir", help="Output directory for CSV files and jsonl files")
    return parser.parse_args()
"""
python3 -m distance_matrix \
    --input /home/mkeber/syn-bioner/data/processed/ncbi/trf/syntax_features_sent_tree_head.jsonl \
    --input_synth /home/mkeber/syn-bioner/data/synthetic/kshot_syn_generation_10pct/corrected_generated_sentences_20260302.jsonl
"""

def main():
    """Example usage"""
    args = argparse_args()
    # Load results from JSONL
    INPUT_FILE = args.input
    basename = os.path.basename(INPUT_FILE).replace('.*', '.csv')
    # OUTPUT_DIR = args.output_dir
    # OUTPUT_FILE = f"{OUTPUT_DIR}" + os.path.basename(INPUT_FILE).replace('.jsonl', '_D.jsonl')
    OUTPUT_FILE = INPUT_FILE + '_D.jsonl'
    
    OUTPUT_DIR = os.path.dirname(OUTPUT_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading {OUTPUT_FILE}...")
    if os.path.isfile(OUTPUT_FILE):
        results = DistanceMatrixGenerator.load_results_from_jsonl(OUTPUT_FILE)
    else:
        results = actual_D.CorpusWithActualD.load_and_calculate_D(
                        INPUT_FILE,
                        output_path=OUTPUT_FILE,
                        verbose=True
                        )
    # results = random.sample(results,100)
    print(f"✓ Loaded {len(results)} sentences\n")
    
    # Create distance matrix using different distance functions
    print("\n" + "=" * 80)
    print("1. Simple D Distance (|D1 - D2|)")
    print("=" * 80)
    dist_matrix_simple, ids = DistanceMatrixGenerator.create_distance_matrix(
        results,
        distance_func=SentenceDistance.simple_D_distance,
        verbose=True
    )
    DISTANCE_MATRIX_NN = f"{OUTPUT_DIR}/distance_matrix_{basename}_{dist_matrix_simple.shape[0]}.npz"
    DistanceMatrixGenerator.save_distance_matrix(
        dist_matrix_simple, ids, DISTANCE_MATRIX_NN, format='npz'
    )
    DistanceAnalyzer.print_statistics(dist_matrix_simple)
    # Dz distance (z statistics)

    print("\n" + "=" * 80)
    print("nm distances original to synthetic data")
    print("=" * 80)

    if args.input_synth:
        INPUT_FILE_SYNTH = args.input_synth
        OUTPUT_FILE_SYNTH = INPUT_FILE_SYNTH.replace('.jsonl', '_D.jsonl')
        OUTPUT_DIR_SYNTH = os.path.dirname(OUTPUT_FILE_SYNTH)
        if os.path.isfile(OUTPUT_FILE_SYNTH):
            results_synth = DistanceMatrixGenerator.load_results_from_jsonl(OUTPUT_FILE_SYNTH)
        else:
            results_synth = actual_D.CorpusWithActualD.load_and_calculate_D(
            INPUT_FILE_SYNTH,
            output_path=OUTPUT_FILE_SYNTH,
            verbose=True
            )
        dist, sentence_ids = DistanceMatrixGenerator.create_distance_matrix_nm(results, results_synth)
        DISTANCE_MATRIX_NM_FILE = f'{OUTPUT_DIR_SYNTH}/distance_matrix_nm_{dist.shape[0]}_{dist.shape[1]}.npz'
        DistanceMatrixGenerator.save_distance_matrix(dist, sentence_ids, DISTANCE_MATRIX_NM_FILE, format='npz')
        print(f'OUTPUT_FILE_SYNTH: {OUTPUT_FILE_SYNTH}')
        print(f'Distance matrix saved to: {DISTANCE_MATRIX_NM_FILE}')

        bar_plot_cluster_sizes.main([
            "--synth", DISTANCE_MATRIX_NM_FILE,
            "--p", "30",
            "--strategy", "mean",
            "--base", OUTPUT_FILE,
            "--dist", DISTANCE_MATRIX_NN
        ])

    print("\n✓ All distance matrices saved!")


if __name__ == "__main__":
    main()
