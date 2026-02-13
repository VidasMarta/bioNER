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
from itertools import combinations

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

def node_edge_node_subgrfs(G):
    subgrfs = []
    for nodes in combinations(G.nodes, 2):
        G_sub = G.subgraph(nodes) # Create subgraph induced by nodes
        # Check for weak connectivity
        if nx.is_weakly_connected(G_sub):
            SG = G.__class__()
            for node in G_sub.nodes:
                SG.add_node(node, feature = G_sub.nodes[node]['feature'], is_root= G_sub.nodes[node]['is_root'])
            for edge in G_sub.edges:
                SG.add_edge(edge[0], edge[1], feature = G_sub.edges[edge]['feature'])
            subgrfs.append(SG)
    return subgrfs


def parent_children_subgrfs(G):
    subgrfs = []
    for node in G.nodes:
        edges_out = G.out_edges(node)
        if len(edges_out) >= 2:
            for e1, e2 in combinations(edges_out, 2):
                SG = G.__class__()
                SG.add_node(node, feature = G.nodes[node]['feature'], is_root = G.nodes[node]['is_root'])
                SG.add_node(e1[1], feature = G.nodes[e1[1]]['feature'], is_root = G.nodes[e1[1]]['is_root'])
                SG.add_node(e2[1], feature = G.nodes[e2[1]]['feature'], is_root = G.nodes[e2[1]]['is_root'])
                SG.add_edge(e1[0], e1[1], feature = G.edges[e1]['feature'])
                SG.add_edge(e2[0], e2[1], feature = G.edges[e2]['feature'])
                subgrfs.append(SG)
    return subgrfs

def edge_node_edge_subgrfs(G):
    subgrfs = []
    for node in G.nodes:
        edges_in = G.in_edges(node)
        edges_out = G.out_edges(node)
        if len(edges_in) >= 1 and len(edges_out) >= 1:
            for e_in in edges_in:
                for e_out in edges_out:
                    SG = G.__class__()
                    SG.add_node(e_in[0], feature = G.nodes[e_in[0]]['feature'], is_root = G.nodes[e_in[0]]['is_root'])
                    SG.add_node(node, feature = G.nodes[node]['feature'], is_root = G.nodes[node]['is_root'])
                    SG.add_node(e_out[1], feature = G.nodes[e_out[1]]['feature'], is_root = G.nodes[e_out[1]]['is_root'])
                    SG.add_edge(e_in[0], e_in[1], feature = G.edges[e_in]['feature'])
                    SG.add_edge(e_out[0], e_out[1], feature = G.edges[e_in]['feature'])
                    subgrfs.append(SG)   
    return subgrfs

 
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
