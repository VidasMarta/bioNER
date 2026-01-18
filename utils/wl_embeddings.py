import networkx as nx
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

import hashlib

from sklearn.feature_extraction.text import CountVectorizer


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
            is_root = (parents[i] == i)
            node_label = f"{pos_tags[i]}|ROOT" if is_root else pos_tags[i]
            G.add_node(i, feature=node_label, is_root=is_root)

    
        # --- Add edges (parent -> child) ---
        for child_idx, parent_idx in enumerate(parents):
            G.add_edge(parent_idx, child_idx, feature=f'{dep_labels[child_idx]}')
        
        graphs.append(G)
    return graphs, sent_ids_list

def hash_to_vector(h, dim):
    v = np.zeros(dim, dtype=np.float32)
    idx = int(hashlib.md5(h.encode()).hexdigest(), 16) % dim
    v[idx] = 1.0
    return v


def get_graph_embedding(graphs, wl_iterations=2, dimensions=16):
    embeddings = []
    for G in graphs:
        h = nx.weisfeiler_lehman_graph_hash(
            G,
            node_attr='feature',
            iterations=wl_iterations
        )
        embeddings.append(hash_to_vector(h, dimensions))
    return np.vstack(embeddings)

'''
def wl_relabel(graph, node_labels, h):
    labels = dict(node_labels)
    all_labels = []

    for depth in range(h):
        new_labels = {}
        for node in sorted(graph.nodes()):
            out_neighbors = sorted(graph.successors(node))
            in_neighbors = sorted(graph.predecessors(node))

            neighbor_labels = (
                [
                    f"OUT_{labels[n]}:{graph.edges[node, n].get('feature', '')}"
                    for n in out_neighbors
                ] +
                [
                    f"IN_{labels[n]}:{graph.edges[n, node].get('feature', '')}"
                    for n in in_neighbors
                ]
            )

            raw = labels[node] + "_" + "_".join(neighbor_labels)
            hashed = hashlib.md5(raw.encode()).hexdigest()
            new_labels[node] = hashed
            all_labels.append(f"h{depth}_{hashed}")

        labels = new_labels

    return all_labels


def graph_to_wl_document(G, wl_iterations):
    """
    Convert graph to WL subtree multiset string.
    """
    # initial node labels
    node_labels = {
        n: str(G.nodes[n].get("feature", "X"))
        for n in sorted(G.nodes())
    }

    features = []
    features.extend(node_labels.values())

    features.extend(
        wl_relabel(G, node_labels, wl_iterations)
    )

    return " ".join(features)

class WLTFIDFEmbedder:
    def __init__(self, wl_iterations=2, max_features=10000):
        self.wl_iterations = wl_iterations
        self.vectorizer = TfidfVectorizer(
            lowercase=False,
            token_pattern=r"[^ ]+",
            max_features=max_features,
            norm="l2",
        )

    def fit(self, graphs):
        docs = [graph_to_wl_document(G, self.wl_iterations) for G in graphs]
        self.vectorizer.fit(docs)

    def transform(self, graphs):
        docs = [graph_to_wl_document(G, self.wl_iterations) for G in graphs]
        return self.vectorizer.transform(docs).toarray()

    def fit_transform(self, graphs):
        docs = [graph_to_wl_document(G, self.wl_iterations) for G in graphs]
        return self.vectorizer.fit_transform(docs).toarray()


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
        
        G = nx.convert_node_labels_to_integers(G, ordering="sorted")
        graphs.append(G)
    return graphs, sent_ids_list

def get_graph_embedding(
    graphs,
    model=None,
    wl_iterations=2,
    max_features=10000,
):
    if model is None:
        model = WLTFIDFEmbedder(
            wl_iterations=wl_iterations,
            max_features=max_features,
        )
        embeddings = model.fit_transform(graphs)
    else:
        embeddings = model.transform(graphs)

    return embeddings, model
'''