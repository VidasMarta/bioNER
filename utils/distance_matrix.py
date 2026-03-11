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
    def omega_distance(d_1: float, n_words1: int, d_min_1: float, 
                       d_2: float, n_words2: int, d_min_2: float) -> float:
        """
        Simple absolute difference between D values
        
        Args:
            d1: D value of sentence 1
            d2: D value of sentence 2
            D1_min: minimum theoretical value
            D2_min: minimum theoretical value
            Drla1 = 1/3*(n_words1**2-1): uniformly random linear arrangement of a certain tree
            Drla1 = 1/3*(n_words2**2-1): uniformly random linear arrangement of a certain tree
            
        Returns:
            Distance: |omega1 - omega2|
        """
        D_rla_1 = 1/3*(n_words1**2-1)
        D_rla_2 = 1/3*(n_words2**2-1)
        omega_1 = ( D_rla_1 - d_1 ) / ( D_rla_1 - d_min_1)
        omega_2 = ( D_rla_2 - d_2 ) / ( D_rla_2 - d_min_2)
        
        return abs(omega_1 - omega_2)
    
    @staticmethod
    def dz_distance(d_1: float, n_words1: int, 
                    d_2: float, n_words2: int, 
                    d_min_2: float = None, d_min_1: float = None) -> float:
        var1 = (1 / 180 ) * (n_words1**2 - 1) * (n_words1**2 - 4) 
        var2 = (1 / 180 ) * (n_words2**2 - 1) * (n_words2**2 - 4) 
        D_rla_1 = 1/3 * (n_words1**2-1)
        D_rla_2 = 1/3 * (n_words2**2-1)
        return abs(( d_1 - D_rla_1) / np.sqrt(var1) - (d_2 - D_rla_2) / np.sqrt(var2))

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
    
    @staticmethod
    def euclidean_distance(d1: float, n_words1: int, avg_dist1: float,
                          d2: float, n_words2: int, avg_dist2: float) -> float:
        """
        Euclidean distance in 3D space: (D, n_words, avg_distance)
        
        Args:
            d1, n_words1, avg_dist1: Features of sentence 1
            d2, n_words2, avg_dist2: Features of sentence 2
            
        Returns:
            Euclidean distance
        """
        return np.sqrt((d1 - d2)**2 + (n_words1 - n_words2)**2 + (avg_dist1 - avg_dist2)**2)
    
    @staticmethod
    def custom_distance(d1: float, n_words1: int, avg_dist1: float,
                       d2: float, n_words2: int, avg_dist2: float,
                       d_weight: float = 0.5,
                       norm_d_weight: float = 0.3,
                       avg_dist_weight: float = 0.2) -> float:
        """
        Weighted combination of multiple distance metrics
        
        Args:
            d1, n_words1, avg_dist1: Features of sentence 1
            d2, n_words2, avg_dist2: Features of sentence 2
            d_weight: Weight for raw D difference
            norm_d_weight: Weight for normalized D difference
            avg_dist_weight: Weight for average distance difference
            
        Returns:
            Weighted distance (normalized to 0-1 range roughly)
        """
        # Normalize weights
        total_weight = d_weight + norm_d_weight + avg_dist_weight
        d_weight /= total_weight
        norm_d_weight /= total_weight
        avg_dist_weight /= total_weight
        
        # Calculate components
        d_diff = SentenceDistance.simple_D_distance(d1, d2)
        norm_d_diff = SentenceDistance.normalized_D_distance(d1, n_words1, d2, n_words2)
        avg_diff = SentenceDistance.avg_distance_difference(avg_dist1, avg_dist2)
        
        # Weighted sum
        distance = (d_weight * d_diff + 
                   norm_d_weight * norm_d_diff + 
                   avg_dist_weight * avg_diff)
        
        return distance


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
        sentence_ids = [r['id'] for r in results]
        
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
                
                elif distance_func == SentenceDistance.euclidean_distance:
                    # Euclidean in 3D
                    dist = distance_func(
                        results[i]['D'], results[i]['n_words'], results[i]['avg_distance'],
                        results[j]['D'], results[j]['n_words'], results[j]['avg_distance']
                    )
                
                elif distance_func == SentenceDistance.custom_distance:
                    # Custom weighted
                    dist = distance_func(
                        results[i]['D'], results[i]['n_words'], results[i]['avg_distance'],
                        results[j]['D'], results[j]['n_words'], results[j]['avg_distance']
                    )
                elif distance_func == SentenceDistance.dz_distance:
                    dist = distance_func(
                        results[i]['D'], results[i]['n_words'], 
                        results[j]['D'], results[j]['n_words'], 
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


def argparse_args():
    parser = argparse.ArgumentParser(description="Dependency Distance Matrix Generator")
    parser.add_argument("--input", help="Input JSONL file")
    parser.add_argument("--input_synth", help="Input synthetic JSONL file")
    parser.add_argument("--output_dir", help="Output directory for CSV files and jsonl files")
    return parser.parse_args()
"""
~/syn-bioner/DDm/src$ python distance_matrix.py \
    --input /home/mkeber/syn-bioner/data/processed/ncbi/trf/syntax_features_sent_tree_head.jsonl \
    --input_synth /home/mkeber/syn-bioner/data/synthetic/kshot_syn_generation_10pct/corrected_generated_sentences_20260302.jsonl
    python distance_matrix.py \
    --input /home/mkeber/syn-bioner/data/processed/ncbi/trf/syntax_features_sent_tree_head.jsonl \
    --input_synth /home/mkeber/syn-bioner/data/synthetic/kshot_syn_generation_20pct/corrected_generated_sentences_20260303.jsonl
    
    python distance_matrix.py \
    --input /home/mkeber/syn-bioner/data/processed/ncbi/trf/syntax_features_sent_tree_head.jsonl \
    --input_synth /home/mkeber/syn-bioner/data/synthetic/0shot_syn_generation_all/corrected_generated_sentences_20260214.jsonl"""

def main():
    """Example usage"""
    args = argparse_args()
    # Load results from JSONL
    INPUT_FILE = args.input
    INPUT_FILE_SYNTH = args.input_synth
    # OUTPUT_DIR = args.output_dir
    # OUTPUT_FILE = f"{OUTPUT_DIR}" + os.path.basename(INPUT_FILE).replace('.jsonl', '_D.jsonl')
    OUTPUT_FILE = INPUT_FILE.replace('.jsonl', '_D.jsonl')
    OUTPUT_FILE_SYNTH = INPUT_FILE_SYNTH.replace('.jsonl', '_D.jsonl')
    OUTPUT_DIR = os.path.dirname(OUTPUT_FILE)
    OUTPUT_DIR_SYNTH = os.path.dirname(OUTPUT_FILE_SYNTH)

    if INPUT_FILE_SYNTH:
        if os.path.isfile(OUTPUT_FILE_SYNTH):
            results_synth = DistanceMatrixGenerator.load_results_from_jsonl(OUTPUT_FILE_SYNTH)
        else:
            results_synth = actual_D.CorpusWithActualD.load_and_calculate_D(
            INPUT_FILE_SYNTH,
            output_path=OUTPUT_FILE_SYNTH,
            verbose=True
            )
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
    DistanceMatrixGenerator.save_distance_matrix(
        dist_matrix_simple, ids, f"{OUTPUT_DIR}/distance_matrix_simple_D.csv", format='csv'
    )
    DistanceAnalyzer.print_statistics(dist_matrix_simple)
    # Dz distance (z statistics)

    print("\n" + "=" * 80)
    print("nm distances original to synthetic data")
    print("=" * 80)

    dist, sentence_ids = DistanceMatrixGenerator.create_distance_matrix_nm(results, results_synth)
    DistanceMatrixGenerator.save_distance_matrix(dist, sentence_ids, f"{OUTPUT_DIR_SYNTH}/distance_matrix_nm.npz", format='npz')
    print("\n✓ All distance matrices saved!")


if __name__ == "__main__":
    main()
