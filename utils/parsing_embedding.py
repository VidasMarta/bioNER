import os
import typing
import numpy as np
import itertools
from pprint import pprint
import networkx as nx
from karateclub import Graph2Vec
from karateclub import GL2Vec
import json
from tqdm import tqdm

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
        
        graphs.append(G)
    return graphs, sent_ids_list
 
def get_graph_embedding(graphs, model=None, embedding_type="gl2vec", wl_iterations=1, 
                        dimensions=16, workers=24, learning_rate=0.05,
                        min_count=1, epochs=20):

    if model is None:
        if embedding_type == "gl2vec" :
            model = GL2Vec(wl_iterations=wl_iterations, dimensions=dimensions, 
                        workers=workers, learning_rate=learning_rate, 
                        min_count=min_count, epochs=epochs)
        elif embedding_type == "graph2vec":
            model = Graph2Vec(wl_iterations=wl_iterations, dimensions=dimensions, 
                            workers=workers, learning_rate=learning_rate,
                            min_count=min_count, epochs=epochs)
        else:
            raise Exception(f"No such embedding type {embedding_type}!")
    model.fit(graphs) #list(graphs.values()))
    embeddings = model.get_embedding()
    embeddings_data = np.array(embeddings)
    return embeddings_data, model
