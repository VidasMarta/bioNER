"""
Calculate ACTUAL D from syntactic dependency trees
D = sum of distances between syntactically related words in their original positions

Distance definition:
- Words at positions i and j have distance |i - j|
- Consecutive words (adjacent positions): distance = 1
- Words separated by 1 word: distance = 2
- etc.

This is the ACTUAL D for the original sentence word order, not the minimal D_min.
"""

import json
import networkx as nx
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import spacy


class ActualDCalculator:
    """Calculate actual D (distance) from dependency trees"""
    
    @staticmethod
    def calculate_D_from_positions(edges: List[Tuple[int, int]], 
                                   edge_features: Optional[Dict] = None) -> Tuple[int, List[int]]:
        """
        Calculate actual D from edges in original sentence positions
        
        Args:
            edges: List of (source_idx, target_idx) tuples (0-indexed word positions)
            edge_features: Optional dict mapping edges to feature names
            
        Returns:
            Tuple of (total_D, list_of_edge_distances)
        """
        total_D = 0
        edge_distances = []
        
        for source, target in edges:
            # Calculate distance: |position_source - position_target|
            distance = abs(source - target)
            total_D += distance
            edge_distances.append(distance)
        
        return total_D, edge_distances
    
    @staticmethod
    def calculate_D_from_parents(parents: List[int]) -> Tuple[int, List[int]]:
        """
        Calculate actual D from parent array
        
        Args:
            parents: List where parents[i] is parent index of word i (0-indexed)
                    Root typically has parents[root] = root or None
                    
        Returns:
            Tuple of (total_D, list_of_edge_distances)
        """
        total_D = 0
        edge_distances = []
        
        for child_idx, parent_idx in enumerate(parents):
            # Skip root (self-loop or None)
            if parent_idx is None or parent_idx == child_idx:
                continue
            
            # Distance between parent and child in original positions
            distance = abs(parent_idx - child_idx)
            total_D += distance
            edge_distances.append(distance)
        
        return total_D, edge_distances
    
    @staticmethod
    def calculate_D_from_nx_graph(G: nx.DiGraph) -> Tuple[int, List[Tuple[int, int, int]]]:
        """
        Calculate actual D from NetworkX dependency graph
        
        Args:
            G: NetworkX DiGraph where nodes are word positions (0-indexed)
               Edges represent syntactic dependencies
               
        Returns:
            Tuple of (total_D, list_of_edge_info)
            Each edge_info is (source, target, distance)
        """
        total_D = 0
        edge_info = []
        
        for source, target in G.edges():
            distance = abs(int(source) - int(target))
            total_D += distance
            
            # Get edge feature if available
            feature = G[source][target].get('feature', 'unknown')
            edge_info.append({
                'source': int(source),
                'target': int(target),
                'distance': distance,
                'feature': feature
            })
        
        return total_D, edge_info
    
    @staticmethod
    def analyze_sentence(sentence: str,
                        parents: List[int],
                        pos_tags: List[str],
                        dep_labels: List[str], 
                        tokens: List[str] = None) -> Dict:
        """
        Comprehensive analysis of a sentence with D calculation
        
        Args:
            sentence: The sentence string
            parents: Parent indices for each word
            pos_tags: POS tags for each word
            dep_labels: Dependency labels for each word
            
        Returns:
            Dictionary with:
            - 'sentence': original sentence
            - 'words': list of words
            - 'D': actual distance sum
            - 'n_words': number of words
            - 'edges': list of (parent, child) pairs
            - 'edge_distances': list of distances
            - 'edge_info': detailed info per edge
            - 'avg_distance': average edge distance
        """
        if tokens is None: 
            tokens = pos_tags
        
        n = len(tokens)
        
        # Validate inputs
        if len(parents) != n or len(pos_tags) != n or len(dep_labels) != n:
            raise ValueError(f"Length mismatch: words={n}, parents={len(parents)}, "
                           f"pos={len(pos_tags)}, deps={len(dep_labels)}")
        
        # Calculate D
        total_D, edge_distances = ActualDCalculator.calculate_D_from_parents(parents)
        
        # Build edge list with details
        edges = []
        edge_info = []
        
        for child_idx, parent_idx in enumerate(parents):
            if parent_idx is None or parent_idx == child_idx:
                continue
            
            distance = abs(parent_idx - child_idx)
            edges.append((parent_idx, child_idx))
            
            edge_info.append({
                'parent_word': tokens[parent_idx],
                'parent_idx': parent_idx,
                'parent_pos': pos_tags[parent_idx],
                'child_word': tokens[child_idx],
                'child_idx': child_idx,
                'child_pos': pos_tags[child_idx],
                'dep_label': dep_labels[child_idx],
                'distance': distance
            })
        
        avg_distance = total_D / len(edges) if edges else 0
        
        return {
            'sentence': sentence,
            'words': tokens,
            'D': total_D,
            'n_words': n,
            'n_edges': len(edges),
            'edges': edges,
            'edge_distances': edge_distances,
            'edge_info': edge_info,
            'avg_distance': avg_distance,
            'max_distance': max(edge_distances) if edge_distances else 0,
            'min_distance': min(edge_distances) if edge_distances else 0
        }


class CorpusWithActualD:
    """Load NCBI corpus and calculate ACTUAL D (not D_min)"""
    
    @staticmethod
    def load_and_calculate_D(jsonl_path: str, 
                            output_path: Optional[str] = None,
                            verbose: bool = True, spacy_model: str = "en_core_web_sm") -> List[Dict]:
        """
        Load NCBI JSONL and calculate actual D for each sentence
        
        Args:
            jsonl_path: Path to JSONL file
            output_path: Optional output file for results
            verbose: Print progress
            spacy_model: spaCy model to use for NLP processing
            
        Returns:
            List of analysis results
        """
        if verbose:
            print("=" * 70)
            print("NCBI Corpus - Actual D Calculation")
            print("=" * 70)
            print(f"\nLoading {jsonl_path}...")
        
        results = []
        count = 0
        skipped = 0
        nlp = spacy.load(spacy_model)
        
        with open(jsonl_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line)
                    
                    # Filter by corpus
                    # if entry.get("corpus") != "NCBI_train":
                    #     continue
                    
                    sentence = entry["sentence"]
                    if not getattr(entry, "pos", None) or not getattr(entry, "dep", None) or not getattr(entry, "parents", None):
                        # Use spacy to calculate pos dep and parents if not provided
                        doc = nlp(sentence)
                        pos_tags = [token.pos_ for token in doc]
                        dep_labels = [token.dep_ for token in doc]
                        parents = [token.head.i for token in doc]
                    else:
                        pos_tags = entry["pos"]
                        dep_labels = entry["dep"]
                        parents = entry["parents"]
                    sent_id = getattr(entry, "id", None)
                    
                    n = len(pos_tags)
                    
                    # Skip very short sentences
                    if n <= 2:
                        skipped += 1
                        continue
                    
                    # Calculate actual D
                    analysis = ActualDCalculator.analyze_sentence(
                        sentence, parents, pos_tags, dep_labels
                    )
                    
                    # Add metadata
                    analysis['id'] = sent_id
                    analysis['corpus'] = entry.get("corpus")
                    
                    results.append(analysis)
                    count += 1
                    
                    if verbose and count % 500 == 0:
                        print(f"  Processed {count} sentences...")
                
                except Exception as e:
                    if verbose:
                        print(f"  Error on line {line_num}: {e}")
                    continue
        
        if verbose:
            print(f"\n✓ Successfully processed {count} sentences")
            print(f"  Skipped {skipped} sentences (too short)")
        
        # Save results
        if output_path:
            CorpusWithActualD._save_results(results, output_path, verbose)
        
        return results
    
    @staticmethod
    def _save_results(results: List[Dict], output_path: str, verbose: bool = True):
        """Save results to JSONL file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for result in results:
                output_obj = {
                    'id': result['id'],
                    'sentence': result['sentence'],
                    'D': result['D'],
                    'n_words': result['n_words'],
                    'n_edges': result['n_edges'],
                    'avg_distance': round(result['avg_distance'], 2),
                    'max_distance': result['max_distance'],
                    'min_distance': result['min_distance']
                }
                f.write(json.dumps(output_obj) + '\n')
        
        if verbose:
            print(f"✓ Results saved to {output_path}")
    
    @staticmethod
    def print_statistics(results: List[Dict]):
        """Print statistics about D values"""
        if not results:
            print("No results to analyze")
            return
        
        D_values = [r['D'] for r in results]
        avg_distances = [r['avg_distance'] for r in results]
        n_edges_list = [r['n_edges'] for r in results]
        
        print("\n" + "=" * 70)
        print("STATISTICS - ACTUAL D")
        print("=" * 70)
        
        print(f"\nTotal Sentences: {len(results)}")
        
        print(f"\nD (Total Distance):")
        print(f"  Min:    {min(D_values)}")
        print(f"  Max:    {max(D_values)}")
        print(f"  Mean:   {sum(D_values) / len(D_values):.2f}")
        print(f"  Median: {sorted(D_values)[len(D_values)//2]}")
        
        print(f"\nAverage Distance per Edge:")
        print(f"  Min:    {min(avg_distances):.2f}")
        print(f"  Max:    {max(avg_distances):.2f}")
        print(f"  Mean:   {sum(avg_distances) / len(avg_distances):.2f}")
        
        print(f"\nEdges per Sentence:")
        print(f"  Min:    {min(n_edges_list)}")
        print(f"  Max:    {max(n_edges_list)}")
        print(f"  Mean:   {sum(n_edges_list) / len(n_edges_list):.2f}")
        
        print("\n" + "=" * 70)
    
    @staticmethod
    def find_sentences_by_D(results: List[Dict], 
                           min_D: Optional[int] = None,
                           max_D: Optional[int] = None) -> List[Dict]:
        """Filter results by D range"""
        filtered = []
        
        for result in results:
            D = result['D']
            if min_D is not None and D < min_D:
                continue
            if max_D is not None and D > max_D:
                continue
            filtered.append(result)
        
        return filtered
    
    @staticmethod
    def most_distance_heavy(results: List[Dict], n: int = 10) -> List[Dict]:
        """Get n sentences with highest D"""
        sorted_results = sorted(results, key=lambda r: r['D'], reverse=True)
        return sorted_results[:n]
    
    @staticmethod
    def least_distance_heavy(results: List[Dict], n: int = 10) -> List[Dict]:
        """Get n sentences with lowest D"""
        sorted_results = sorted(results, key=lambda r: r['D'])
        return sorted_results[:n]


def main():
    """Example usage"""
    
    # EDIT THESE PATHS FOR YOUR DATA
    INPUT_FILE = "/home/mkeber/syn-bioner/data/synthetic/kshot_syn_generation_10pct/corrected_generated_sentences_20260302.jsonl"
    OUTPUT_FILE = "/home/mkeber/syn-bioner/data/synthetic/kshot_syn_generation_10pct/corrected_generated_sentences_20260302_D.jsonl"
    
    # Load and calculate D
    results = CorpusWithActualD.load_and_calculate_D(
        INPUT_FILE,
        output_path=OUTPUT_FILE,
        verbose=True,
        spacy_model="en_core_web_sm"  # Change if you want a different spaCy model
    )
    
    if not results:
        print("No results")
        return
    
    # Print statistics
    CorpusWithActualD.print_statistics(results)
    
    # Show examples
    print("\nMost Distance-Heavy (Longest Dependencies):")
    for i, result in enumerate(CorpusWithActualD.most_distance_heavy(results, 3), 1):
        print(f"\n{i}. D={result['D']} (avg={result['avg_distance']:.2f})")
        print(f"   {result['sentence']}")
    
    print("\n\nLeast Distance-Heavy (Shortest Dependencies):")
    for i, result in enumerate(CorpusWithActualD.least_distance_heavy(results, 3), 1):
        print(f"\n{i}. D={result['D']} (avg={result['avg_distance']:.2f})")
        print(f"   {result['sentence']}")
    
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
