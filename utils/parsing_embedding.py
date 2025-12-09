import os
import typing
import numpy as np
import itertools
from pprint import pprint
import networkx as nx
from karateclub import Graph2Vec
from karateclub import GL2Vec
import json
from kmeans_params import evaluate_k
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
    graphs = {}
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
        
        graphs[sent_id] = G
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
    model.fit(list(graphs.values()))
    embeddings = model.get_embedding()
    embeddings_data = np.array(embeddings)
    return embeddings_data, model

if __name__ == '__main__':
    # graphs = ... your graphs dictionary
    # filename = os.path.basename('/home/mkeber/syn-bioner/data/ncbi/trf/syntax_features_sent_tree_head.json')[-10:-5]
    
    with open('/home/mkeber/syn-bioner/data/ncbi/trf/syntax_features_sent_tree_head.json', 'r') as file:
        real_data = [json.loads(line) for line in file]
    # Define the grid of hyperparameters
    param_grid = {
        "wl_iterations": [1, 2],
        "dimensions": [16, 32, 64],
        "workers": [24],  # fixed
        "learning_rate": [0.04, 0.05, 0.06, 0.1],
        "min_count": [1, 2, 3],
        "epochs": [20, 25, 30]
    }

    # Generate all combinations of hyperparameters
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    output_path = "/home/mkeber/syn-bioner/data/clustering/grid_search_embeddings_train.jsonl"
    output_path_kmeans = "/home/mkeber/syn-bioner/data/clustering/grid_search_embeddings_train_kmeans.jsonl"
    
    if os.path.isfile(output_path): f = open(output_path, "a")   
    else: f = open(output_path, "w")
    for i, combo in tqdm(enumerate(param_combinations)):
        print(f"Running GL2Vec with params: {combo}")
        
        # Initialize and fit model
        real_graphs, sent_ids_list = build_dependency_graphs(real_data)
        real_emb, _ = get_graph_embedding(real_graphs, **combo)
        embeddings_gl_data = np.array(real_emb)
        
        # Save each embedding with its hyperparameters
        entry = {
            "id": int(i),
            # "sent": sent_ids_list,
            "hyperparameters": combo,
            # "embedding": [emb.tolist() for emb in embeddings_gl_data],
        }

        for k in tqdm(range(5, 25)):
            kmeans_result = evaluate_k(embeddings_gl_data, k)
            entry[str(k)] = {
                'inertia': str(kmeans_result['inertia']),
                'silhouette': str(kmeans_result['silhouette']),
                # 'centroids': kmeans_result['centroids'].tolist(),
                # 'labels': kmeans_result['labels'].tolist() 
            }
        with open(output_path_kmeans, "a") as f:
            f.write(json.dumps(entry) + "\n")