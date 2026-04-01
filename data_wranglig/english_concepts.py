# %%
import os
import typing
import pandas as pd
import numpy as np
from pprint import pprint

# %%
df_ancestor = pd.read_csv('/home/mkeber/syn-bioner/data/KB/SNOMEDCT/CONCEPT_ANCESTOR.csv', delimiter='\t', on_bad_lines='skip')
df_concepts = pd.read_csv('/home/mkeber/syn-bioner/data/KB/SNOMEDCT/CONCEPT.csv', delimiter='\t', on_bad_lines='skip')


# %%
df_concepts.head()

# %%


# Filter by ancestor
descendants = df_ancestor[df_ancestor['ancestor_concept_id'].isin([4274025])]['descendant_concept_id'].values

# Filter by language and descendants
# sp_df = df_concepts[df_concepts['language_concept_id'] == args.lang_id]
output = df_concepts[df_concepts['concept_id'].isin(descendants)]

# %%
print(output.concept_class_id.unique(),
output.domain_id.unique())


# %%
output= output[output.concept_class_id!='Event']

# %%
output=output[output.concept_class_id!='Clinical Finding']

# %%
output=output[output.concept_class_id!='Procedure']


# %%
output.head()

# %%
# Save Output
v[['concept_name', 'concept_id']].to_csv('/home/mkeber/syn-bioner/data/KB/SNOMEDCT/concepts_disease_filtered_en.csv', index=False)
print(f"File saved successfully to: ")

# %%
output.concept_id.value_counts()

# %%
df_sp = pd.read_csv('/home/mkeber/syn-bioner/data/KB/SNOMEDCT/concepts_disease_filtered_sp.csv')
df_sp= df_sp[df_sp['concept_id'].isin(output['concept_id'])]

# %%
df_sp.concept_id.value_counts()

# %%
df_sp[df_sp['concept_id'] == 4042934]

# %%
df_ge = pd.read_csv('/home/mkeber/syn-bioner/data/KB/ICD10GE/icd10gm2026syst_kodes.txt', on_bad_lines="skip", delimiter=';')

# %%
df_ge.shape

# %%
df = pd.read_csv('/home/mkeber/syn-bioner/data/KB/SNOMEDCT/CONCEPT_SYNONYM.csv', on_bad_lines="skip", delimiter='\t')
df_sp = df[df['language_concept_id'] == 4182511]

# %%
df_sp.drop(columns='language_concept_id', inplace=True)

# %%
df_sp = df_sp[[df_sp.columns[1], df_sp.columns[0]]]
df_sp.to_csv("concepts_filtered_sp.csv", index=False)

# %%
corpus_list = []
id = 0
with open('data/processed/ncbi/trf/ncbi_ner_train.json', "r", encoding="utf-8") as f:
    json_data = json.load(f)
    for entry in json_data:
        id += 1
        entities = entry["entities"]
        corpus_list.append((id, entry["sentence"], entities, corpus_name, entry["abstract_id"], [None]*len(entities)))       
    

# %%
import json

path = "/home/mkeber/syn-bioner/data/init_synthetic_snomed_sent_type/init_synthetic_snomed_sent_type/generated_sentences_20260109.jsonl"

data = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

# data is now a list of dicts
print(len(data))
print(data[0])


# %%
import json

input_path = "/home/mkeber/syn-bioner/data/init_synthetic_snomed_sent_type/init_synthetic_snomed_sent_type/generated_sentences_20260109.jsonl"
output_path = "term_list.txt"

terms = set()

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)

        # item["term"] is like: [['Septicemia during labor'], ['40398249']]
        for t in item.get("term", []):
            if isinstance(t, list) and t and not t[0].isdigit():
                terms.add(t[0])


# %%
len(terms)

# %%
# write to file
with open(output_path, "w", encoding="utf-8") as f:
    for term in sorted(terms):
        f.write(term + "\n")

print(f"Saved {len(terms)} unique terms to {output_path}")


# %%
with open('term_list.txt', 'r') as f:
    gent = [line.strip() for line in f.readlines()]
gent

# %%
import sys
import bioNER.utils.parsing_embedding as pe

# %%
import json
# '/home/mkeber/syn-bioner/data/ncbi/trf/ncbi_ner_train_10pct.json'
args={'wl_iterations': 1,
'dimensions': 16,
'workers': 24,
'learning_rate': 0.05,
'min_count': 1,
'epochs': 20}
with open('/home/mkeber/syn-bioner/data/ncbi/trf/syntax_features_sent_tree_head.json', 'r') as f:
    real_data = [json.loads(line) for line in f]
real_data = [ent for ent in real_data if ent.get('corpus') == 'NCBI_train']

start_time = time.time()
real_graphs, _ = pe.build_dependency_graphs(real_data)

real_emb, model = pe.get_graph_embedding(real_graphs, embedding_type="gl2vec",
                                            wl_iterations=args['wl_iterations'], 
                                            dimensions=args['dimensions'], 
                                            workers=args['workers'], 
                                            learning_rate= args['learning_rate'], 
                                            min_count = args['min_count'], 
                                            epochs = args['epochs'])
print(time.time()-start_time)

# %%
import torch
import random

def set_seed(seed: int = 42): ##za reproducility, izvor: https://medium.com/we-talk-data/how-to-set-random-seeds-in-pytorch-and-tensorflow-89c5f8e80ce4
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

# %%
with open('/home/mkeber/syn-bioner/data/ncbi/trf/ncbi_ner_train_10pct.json', 'r') as f:
    data_10 = [json.loads(line) for line in f]
set_seed(1)    
graphs_data_10, _ = pe.build_dependency_graphs(data_10)
data_10_emb, _ = pe.get_graph_embedding(graphs_data_10, model)

# %%
import bioNER.utils.kmeans_params
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

# %%
real_emb = normalize(real_emb)
km = KMeans(n_clusters=18, random_state=42)

labels = km.fit_predict(real_emb)

inertia = km.inertia_
silhouette = silhouette_score(real_emb, labels)

res = {
    "inertia": inertia,
    "silhouette": silhouette,
    "labels": labels,
    "centroids": km.cluster_centers_,
}
np.save(os.path.join('data/test', "cluster_labels.npy"), labels)
np.save(os.path.join('data/test', "cluster_centroids.npy"), km.cluster_centers_)

# %%
# res

# %%
real_labels = np.load(os.path.join('data/test', "cluster_labels.npy"))
centroids_real = np.load(os.path.join('data/test', "cluster_centroids.npy"))
clusters = [*range(0, max(real_labels)+1)]

# Save NCBI examples with cluster labels for later k-shot selection
# KARATE ENV
ncbi_clustered_path = os.path.join('data/test', "kshot_ncbi_clustered.jsonl")

# %%
with open('/home/mkeber/syn-bioner/data/ncbi/trf/ncbi_ner_train_10pct.json', 'r') as f:
    data_10 = [json.loads(line) for line in f]
set_seed(1)
graphs_data_10, _ = pe.build_dependency_graphs(data_10)
data_10_emb, _ = pe.get_graph_embedding(graphs_data_10, model)

# %%
from sklearn.metrics.pairwise import cosine_similarity
data_10_emb = normalize(data_10_emb)
similarities = cosine_similarity(data_10_emb, centroids_real)
kshot_labels = np.argmax(similarities, axis=1)
with open(ncbi_clustered_path, "w") as f:
    for sample, label in zip(data_10, kshot_labels):
        sample["cluster_id"] = int(label)
        f.write(json.dumps(sample) + "\n")
print(f"[INFO] Saved NCBI examples with cluster IDs → {ncbi_clustered_path}")

# %%
len(kshot_labels)

# %%
with open('/home/mkeber/syn-bioner/data/ncbi/trf/ncbi_ner_train_10pct.json', 'r') as f:
    data_10 = [json.loads(line) for line in f]
set_seed(1)
graphs_data_10, _ = pe.build_dependency_graphs(data_10)
data_10_emb2, _ = pe.get_graph_embedding(graphs_data_10, model)
data_10_emb2 = normalize(data_10_emb2)
similarities = cosine_similarity(data_10_emb2, centroids_real)
kshot_labels2 = np.argmax(similarities, axis=1)

# %%
len(kshot_labels2)

# %%
kshot_labels[:20]

# %%
kshot_labels2[:20]

# %%
data_10_emb2

# %%
import json
# '/home/mkeber/syn-bioner/data/ncbi/trf/ncbi_ner_train_10pct.json'
args={'wl_iterations': 1,
'dimensions': 16,
'workers': 24,
'learning_rate': 0.05,
'min_count': 1,
'epochs': 20}
with open('/home/mkeber/syn-bioner/data/ncbi/trf/ncbi_ner_train_10pct.json', 'r') as f:
    real_data = [json.loads(line) for line in f]
real_data=50*real_data
import time

# %%
len(real_data)

# %%
print(len(real_data))

# %%
l = [1, 2,2,3,4,]
l[1:1]

# %%
{}

# %%
l = [(['type 1 diabetes mellitus'], ['obesity']), (['DOID:9744'], ['DOID:9970'])]
for item, k in l:
    print(item, k)

# %%
for k, l in [(['Immersion, unspecified'], ['4020022'])]:
    print(k, l)
    

# %%
import json

with open('/home/mkeber/syn-bioner/bioNER/data/ncbi/trf/syntax_features/syntax_features_sent_tree_head.json', 'r') as file:
    data = json.load(file)


# %%
# data = data[:10]
# pprint(data[0])

# %%
import networkx as nx

def build_dependency_graphs(data, use_root=True):
    """
    Build NetworkX dependency graphs from dataset entries.

    Args:
        data (list[dict]): list of dependency-parsed sentence dicts.
        use_root (bool): if True, add an explicit root node (id='ROOT_<sent_id>').

    Returns:
        dict[int, nx.DiGraph]: mapping from sentence ID to its NetworkX graph.
    """
    graphs = {}

    for entry in data:
        sent_id = entry["id"]
        sentence = entry["sentence"]
        pos_tags = entry["pos"]
        dep_labels = entry["dep"]
        parents = entry["parents"]

        # Create directed dependency graph
        G = nx.DiGraph(id=sent_id, sentence=sentence)

        n = len(pos_tags)
        root_name = 0

        # Optionally add a synthetic root node
        if use_root:
            G.add_node(root_name, word="ROOT", feature="ROOT", is_root=True)

        # Add nodes (tokens)
        for i in range(1,n+1):
            G.add_node(i, feature=pos_tags[i-1], is_root=False)

        # Add edges (parent -> child)
        for child, parent in enumerate(parents):
            # print(f"child: {child}, parent: {parent}")
            if parent == 0:
                if use_root:
                    # link root node → child
                    # G.add_edge(pos_tags[child], root_name, dep=dep_labels[child])
                    G.add_edge(child+1, root_name, dep=dep_labels[child])
                    
            else:
                # parent index is 1-based, so subtract 1 for zero-based node index
                # parent_idx = parent - 1
                if 0 <= parent < n and use_root:
                    G.add_edge(child+1, parent, dep=dep_labels[child])
                elif 0 <= parent < n:
                    G.add_edge(child+1, parent-1, dep=dep_labels[child])

        graphs[sent_id] = G

    return graphs


# Example usage:
graphs = build_dependency_graphs(data, use_root=True)

# Inspect one example
example_id = 1
G = graphs[example_id]

# print(f"Sentence {example_id}: {G.graph['sentence']}")
# print("Nodes:")
# for node, attrs in G.nodes(data=True):
#     print(f"  {node}: {attrs}")

# print("\nEdges:")
# for u, v, attrs in G.edges(data=True):
#     print(f"  {u} -> {v}: {attrs}")
# print("\n"
#       "-----------------------------------\n"
#       )
# pprint(data[0])

# %%
from karateclub import FeatherGraph

# %%
from karateclub import Graph2Vec
model = Graph2Vec(wl_iterations=2, dimensions=16, attributed=True, workers=24, epochs=40, min_count=5, learning_rate=0.1)
model.fit(list(graphs.values()))
embeddings = model.get_embedding()

# %% [markdown]
# Probli smo: wl_iterations više nije bolje
# 
# learning_rate oko 0.1 default nije dobar
# 
# epochs oko 20 epoha
# 
# Pomaže ako uključimo atribute zapravo imena čvorova. 
# 

# %%
embeddings.shape
corpus = [d.get("corpus") for d in data]

# %%
import umap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=25,        # similar to t-SNE perplexity
    min_dist=0.1,          # controls cluster tightness
    metric="cosine",       # works well for text features
    # random_state=42,
    n_jobs=48,
)

X_umap = umap_model.fit_transform(embeddings)

# Save model for later reuse
# joblib.dump(umap_model, "syntax_umap.pkl")

df = pd.DataFrame(X_umap, columns=["UMAP1", "UMAP2"])
df["corpus"] = corpus

df_ncbi = df[df["corpus"] == "NCBI_train"]
df_gen = df[df["corpus"] == "generated_train"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# --- 1. Both datasets ---
ax = axes[0]
for corpus_name, group in df.groupby("corpus"):
    ax.scatter(group["UMAP1"], group["UMAP2"], alpha=0.6, label=corpus_name)
ax.legend()
ax.set_title("UMAP projection (Both corpora)")
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")

# --- 2. NCBI only ---
bx = axes[1]
bx.scatter(df_ncbi["UMAP1"], df_ncbi["UMAP2"], alpha=0.6, color="blue", label="NCBI_train")
bx.legend()
bx.set_title("UMAP projection (NCBI only)")
bx.set_xlabel("UMAP1")
bx.set_ylabel("UMAP2")

# --- 3. Generated only ---
cx = axes[2]
cx.scatter(df_gen["UMAP1"], df_gen["UMAP2"], alpha=0.6, color="orange", label="Generated")
cx.legend()
cx.set_title("UMAP projection (Generated only)")
cx.set_xlabel("UMAP1")
cx.set_ylabel("UMAP2")

plt.tight_layout()
plt.show()

# %%
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sns.set(style="whitegrid")

neighbors = [5, 10, 20, 50, 100]
min_dists = [0.0, 0.01, 0.05, 0.1, 0.5]

n_rows = len(neighbors)
n_cols = len(min_dists)

fig_all, axes_all = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
fig_ncbi, axes_ncbi = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
fig_gen, axes_gen = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

axes_all = axes_all.flatten()
axes_ncbi = axes_ncbi.flatten()
axes_gen = axes_gen.flatten()

# silhouette_results = []

# # Divide by corpus
# pos_texts_NCBI = [" ".join(d["pos"]) for d in data if d["corpus"] == "NCBI_train"]
# dep_texts_NCBI = [" ".join(d["dep"]) for d in data if d["corpus"] == "NCBI_train"]

# X_scaled_NCBI, _, _ = vectorize_and_scale(pos_texts_NCBI, dep_texts_NCBI)

# pos_texts_gen = [" ".join(d["pos"]) for d in data if d["corpus"] == "generated_train"]
# dep_texts_gen = [" ".join(d["dep"]) for d in data if d["corpus"] == "generated_train"]

# X_scaled_gen, _, _ = vectorize_and_scale(pos_texts_gen, dep_texts_gen)

# --- Iterate over grid ---
for i, n_neighbors in enumerate(neighbors):
    for j, min_dist in enumerate(min_dists):
        # Calculate the linear index for the flattened arrays
        linear_index = i * n_cols + j

        umap_model = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,        # similar to t-SNE perplexity
            min_dist=min_dist,          # controls cluster tightness
            metric="cosine",       # works well for text features
            # random_state=42,
            n_jobs=48,
        )

        umap_fit = umap_model.fit_transform(embeddings)
        # X_umap_NCBI = umap_fit.transform(embeddings)
        # X_umap_gen = umap_fit.transform(embeddings)

        df = pd.DataFrame(umap_fit, columns=["UMAP1", "UMAP2"])
        df["corpus"] = corpus

        df_ncbi = df[df["corpus"] == "NCBI_train"]
        df_gen = df[df["corpus"] == "generated_train"]

        # Select subplot axes
        ax = axes_all[linear_index]
        bx = axes_ncbi[linear_index]
        cx = axes_gen[linear_index]

        # --- Plot both datasets ---
        sns.scatterplot(data=df, x="UMAP1", y="UMAP2", hue="corpus", ax=ax, alpha=0.6, s=10)
        ax.set_title(f"Both | n_neighbors:{n_neighbors}, min_dist:{min_dist}", fontsize=8)
        ax.legend(fontsize=6)

        # --- Plot only NCBI ---
        bx.scatter(df_ncbi["UMAP1"], df_ncbi["UMAP2"], alpha=0.6, color="blue", label="NCBI_train", s=10)
        bx.set_title(f"NCBI | n_neighbors:{n_neighbors}, min_dist:{min_dist}", fontsize=8)
        bx.legend(fontsize=6)

        # --- Plot only Generated ---
        cx.scatter(df_gen["UMAP1"], df_gen["UMAP2"], alpha=0.6, color="orange", label="Generated_train", s=10)
        cx.set_title(f"Generated | n_neighbors:{n_neighbors}, min_dist:{min_dist}", fontsize=8)
        cx.legend(fontsize=6)

# Adjust spacing
for fig in [fig_all, fig_ncbi, fig_gen]:
    fig.tight_layout()

plt.show()


fig_all.savefig("umap_both_grid.png", dpi=300)
fig_ncbi.savefig("umap_ncbi_grid.png", dpi=300)
fig_gen.savefig("umap_generated_grid.png", dpi=300)

# %%
concept_df = pd.read_csv('/home/mkeber/syn-bioner/data/SNOMEDCT/CONCEPT.csv', 
                         delimiter='\t', on_bad_lines='skip', dtype=str)
filtered_df = concept_df[concept_df.domain_id == 'Condition'].copy()
concepts = filtered_df[filtered_df.concept_class_id.isin(['Disorder', 'Clinical Finding',
            'ICD10 code', 'Context-dependent', 'Event', 'Morph Abnormality'])
            ].concept_name.unique()

# %%
final_df = filtered_df[filtered_df.concept_class_id.isin(['Disorder', 'Clinical Finding',
            'ICD10 code', 'Context-dependent', 'Event', 'Morph Abnormality'])
            ][['concept_name', 'concept_id']].drop_duplicates(subset=['concept_name'])

# %%
for index, row in final_df.iterrows():
    print(row.iloc[0], row.iloc[1])
    break

# %%
filtered_df[filtered_df.concept_class_id.isin(['Disorder', 'Clinical Finding',
            'ICD10 code', 'Context-dependent', 'Event', 'Morph Abnormality'])
            ][['concept_name', 'concept_id']].drop_duplicates(subset=['concept_name']).to_csv(
                '/home/mkeber/syn-bioner/data/SNOMEDCT/concepts_filtered.csv', index=False
            )


# %%
with open('/home/mkeber/syn-bioner/data/SNOMEDCT/concepts.txt', 'w') as f:
    for concept in concepts:
        f.write(f"{concept}\n")

# %%
concept_df

# %%

concept_df.concept_class_id.unique()

# %%
concept_df[concept_df.domain_id == 'Condition'].concept_class_id.unique()

# %%
filtered_df = concept_df[concept_df.domain_id == 'Condition'].copy()
filtered_df[filtered_df.concept_class_id.isin(['Disorder', 'Clinical Finding',
            'ICD10 code', 'Context-dependent', 'Event', 'Morph Abnormality'])].concept_name.unique()

# concept_df[concept_df[concept_df.domain_id == 'Condition'].concept_class_id.isin(['Event'])]

# %%
filtered_df[filtered_df.concept_class_id.isin(['Disorder', 'Clinical Finding',
            'ICD10 code', 'Context-dependent', 'Event', 'Morph Abnormality'])].domain_id.unique()

# %%
# 'Morph Abnormality'
filtered_df[filtered_df.concept_class_id == 'Context-dependent']

# %%
filtered_df[filtered_df.concept_class_id.isin(['Disorder', 'Clinical Finding',
            'ICD10 code', 'Context-dependent', 'Event', 'Morph Abnormality'])].concept_name.unique()

# %%
concept_df.domain_id.unique()

# %%
vocabulary_df = pd.read_csv('/home/mkeber/syn-bioner/data/SNOMEDCT/VOCABULARY.csv', delimiter='\t', on_bad_lines='skip', dtype=str)

# %%
vocabulary_df

# %%

text = "Further investigation is required, to confirm the diagnosis of ACG1B! The patient diagnosed with APC. The patient has a history of familial adenomatous polyposis, which is a hereditary condition characterized by the development of numerous adenomatous polyps in the colon, increasing the risk of colorectal cancer."
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
# remove last sentence from doc
print(len(list(doc.sents)))
for sent in doc.sents:
    print(sent)
print("----")
# remove last sentence from doc
remaining = list(doc.sents)[:-1]
doc = nlp(" ".join([str(s) for s in remaining]))
print(len(list(doc.sents)))
for sent in doc.sents:
    print(sent)
doc = list(doc.sents)[0]
print("----")
doc = nlp(str(doc))
tokens = [token.text for token in doc]
print(len(list(doc.sents)))
for sent in doc.sents:
    print(sent)

# %%
eval("['O', 'O', 'O', 'O', 'O', 'O', 'B-DISEASE', 'O', 'O', 'B-DISEASE']")

# %%
tokens = [token.text for token in doc]
tokens_lower = [token.lower() for token in tokens]

# %%
tokens

# %%
import string
print([token.text.strip() for token in doc])
tokens_lower = [token.text.strip() for token in doc]
tokens_lower[-1] = tokens_lower[-1].rstrip(string.punctuation)
print(tokens_lower)

# %%
def generate_term_list(args, file_path: str):
    entities = []
    # Read the file (one dict per line)
    with open(file_path, "r") as f:
        data = [json.loads(line.strip().replace("'", '"')) for line in f if line.strip()]

    # Group by idx
    grouped = defaultdict(list)
    for item in data:
        grouped[item['idx']].append(item)

    # Process each idx
    for idx, items in grouped.items():
        tokens = []
        capture = False
        for item in items:
            gold = item['gold']
            if gold.startswith("B-"):
                if tokens:  # flush previous entity
                    entities.append({"idx": idx, "entity": " ".join(tokens)})
                    tokens = []
                tokens.append(item['token'])
                capture = True
            elif gold.startswith("I-") and capture:
                tokens.append(item['token'])
            else:
                if tokens:  # flush if ended
                    entities.append({"idx": idx, "entity": " ".join(tokens)})
                    tokens = []
                capture = False

        if tokens:  # flush last
            entities.append({"idx": idx, "entity": " ".join(tokens)})

    return entities 
file_path = "/home/mkeber/syn-bioner/data/NCBI-Disease/wrong_val.txt"
generate_term_list(None, file_path)

# %%



