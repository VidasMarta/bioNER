import os
import typing
import pandas as pd
import numpy as np
import spacy
from pprint import pprint
import networkx as nx
from karateclub import Graph2Vec
from karateclub import GL2Vec
import json
 
def build_dependency_graphs(data):
    """
    Build directed dependency graphs encoding only syntactic structure (POS + dependency).
    Graphs are directed from parent -> child.
    
    Args:
    data (list[dict]): list of dependency-parsed sentence dicts.
    
    Returns:
    dict[int, nx.DiGraph]: mapping from sentence ID to NetworkX graph.
    """
    graphs = {}
    
    for entry in data:
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
        
        graphs[sent_id] = G
    return graphs
 
def get_graph_embedding(graphs, embedding_type="gl2vec", wl_iterations=1, dimensions=64, workers=32, learning_rate=0.1,
    min_count=2, epochs=50):
    if embedding_type == "gl2vec":
        model = GL2Vec(wl_iterations=wl_iterations, dimensions=dimensions, workers=workers, learning_rate=learning_rate, min_count=min_count, epochs=epochs)
        model.fit(list(graphs.values()))
        embeddings = model.get_embedding()
        embeddings_data = np.array(embeddings)
    elif embedding_type == "graph2vec":
        model = Graph2Vec(wl_iterations=wl_iterations, dimensions=dimensions, workers=workers, learning_rate=learning_rate, min_count=min_count, epochs=epochs)
        model.fit(list(graphs.values()))
        embeddings = model.get_embedding()
        embeddings_data = np.arraz(embeddings)
    else:
        raise Exception(f"No such embedding type {embedding_type}!")


    return embeddings_data