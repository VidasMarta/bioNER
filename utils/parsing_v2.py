import argparse
import json
import re
import statistics
import string
import numpy as np
import spacy 
from tqdm import tqdm
import random
import os
import matplotlib.pyplot as plt
import subprocess
import sys
from pathlib import Path
# from nltk.tokenize.treebank import TreebankWordDetokenizer
from spacy.tokens import Doc
from typing import List, Dict, Tuple

SPACY_NLP = None

def spacy_load_model(model_name: str):
    global SPACY_NLP
    if SPACY_NLP is None:
        SPACY_NLP = get_spacy(model_name)
        return SPACY_NLP
    return SPACY_NLP   

def get_spacy(model_name: str):
    import spacy
    try: return spacy.load(model_name)
    except Exception as e:
        print(f"Error loading spaCy model: {e}")
        os.system(f"python3 -m spacy download {model_name}")
        print(f"Downloading {model_name} model")
        import spacy
        return spacy.load(model_name)


def parse_text_file(input_file):
    abstracts = {}
    annotations = {} 

    # Step 1: Parse the text file
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Abstract title/abstract text
            if "|t|" in line or "|a|" in line:
                pmid, section, text = line.split("|", 2)
                pmid = int(pmid)
                if pmid not in abstracts:
                    abstracts[pmid] = {"text": ""}
                abstracts[pmid]["text"] += text + " "

            # Entity annotation lines
            elif re.match(r"^\d+\t\d+\t\d+\t", line):
                parts = line.split("\t")
                pmid = int(parts[0])
                start, end = int(parts[1]), int(parts[2])
                mention = parts[3]
                code = parts[-1]
                if pmid not in annotations:
                    annotations[pmid] = []
                annotations[pmid].append({
                    "start": start,
                    "end": end,
                    "text": mention,
                    "code": code
                })
    return abstracts, annotations

def create_and_save_json(abstracts, annotations, output_file, model):
    nlp = spacy.load(model)
    json_data = []
    
    id = 0
    for pmid, abs_data in tqdm(abstracts.items()):
        text = abs_data["text"].strip()
        ents = annotations.get(pmid, [])

        # Sort entities by start offset
        ents.sort(key=lambda x: x["start"])

        doc = nlp(text)
        for sent in doc.sents:
            sent_start = sent.start_char
            sent_end = sent.end_char
            tokens = [t.text for t in sent]
            tags = [2] * len(tokens)  # default O
            entities_in_sent = []
            codes_in_sent = []

            # Step 3: assign BIO tags
            for ent in ents:
                if ent["end"] <= sent_start or ent["start"] >= sent_end:
                    continue  # entity outside sentence

                for i, token in enumerate(sent):
                    token_start = token.idx
                    token_end = token.idx + len(token.text)
                    if token_end <= ent["start"]:
                        continue
                    if token_start >= ent["end"]:
                        break
                    # Tagging scheme: 0=B, 1=I, 2=O
                    if ent["start"] <= token_start < ent["end"]:
                        if tags[i] == 2:
                            tags[i] = 0 if token_start == ent["start"] else 1
                    elif ent["start"] < token_end <= ent["end"]:
                        tags[i] = 1

                entities_in_sent.append(ent["text"])
                codes_in_sent.append(ent["code"])

            id += 1
            json_data.append({
                "id" : id,
                "sentence": sent.text.strip(),
                "tokens": tokens,
                "tags": tags,
                "entities": list(set(entities_in_sent)),
                "codes": list(set(codes_in_sent)),
                "abstract_id": pmid
            })

    # Step 4: Save to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(json_data)} sentence objects to {output_file}")

def filter_by_abstract_ids(input_file, output_file, sample_ratio, pct_train):
    # Load the full dataset
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    data = [item for item in data if item["corpus"] == 'NCBI_train']
    # Collect unique abstract IDs
    abstract_ids = sorted({item["abstract_id"] for item in data})
    print(f"Total abstracts: {len(abstract_ids)}")

    # Randomly sample args.pct of abstract IDs
    sample_size = max(1, int(len(abstract_ids) * sample_ratio))
    sampled_ids = set(random.sample(abstract_ids, sample_size))
    print(f"Selected {len(sampled_ids)} abstracts for the {pct_train*100}% sample.")

    # Filter all sentences that belong to sampled abstracts
    filtered_data = [item for item in data if item["abstract_id"] in sampled_ids]

    # Save to new JSON file
    filtered_abstracts = output_file + f"ncbi_ner_train_{pct_train*100:.0f}pct.json"
    with open(filtered_abstracts, "w", encoding="utf-8") as f:
        for item in filtered_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(filtered_data)} sentences to {filtered_abstracts}")

    return filtered_abstracts

def compute_stats(input_file, output_file):
    # Load data
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats_data = []

    # Define punctuation characters to count
    punct_chars = set(string.punctuation)

    for item in data:
        tokens = item["tokens"]
        entities = item.get("entities", [])
        
        # Basic metrics
        sentence_length = len(tokens)
        num_entities = len(entities)
        num_punct = sum(1 for tok in tokens if any(ch in punct_chars for ch in tok))
        
        # Compute entity lengths (in tokens)
        entity_lengths = []
        for ent in entities:
            ent_tokens = [t for t in tokens if t in ent.split()]  # approximate matching
            if ent_tokens:
                entity_lengths.append(len(ent_tokens))
        
        avg_entity_length = np.mean(entity_lengths) if entity_lengths else 0

        # Add metrics to each record
        item.update({
            "sentence_length": sentence_length,
            "num_entities": num_entities,
            "avg_entity_length": round(float(avg_entity_length), 2),
            "num_punctuations": num_punct
        })
        stats_data.append(item)

    # Save enriched dataset
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, ensure_ascii=False, indent=2)

    print(f"Saved statistics-enriched data to {output_file}")

    return stats_data

def describe(values):
        return {
            "mean": round(statistics.mean(values), 2),
            "min": min(values),
            "max": max(values),
            "stdev": round(statistics.stdev(values), 2) if len(values) > 1 else 0
        }

def plot_histogram(values, title, xlabel, filename, output_dir):
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins="fd", edgecolor="black", alpha=0.7)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    save_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved histogram: {save_path}")

def extract_entities(entry):
    tokens = entry["tokens"]
    tags = entry["tags"]

    entities = []
    current_entity = []

    for token, tag in zip(tokens, tags):
        if tag == 0: # B
            if current_entity:
                entities.append(" ".join(current_entity))
                current_entity = []
            current_entity.append(token)

        elif tag == 1:  # I
            if current_entity:
                current_entity.append(token)
            else:
                current_entity = [token]

        elif tag == 2:  # O
            if current_entity:
                entities.append(" ".join(current_entity))
                current_entity = []

    if current_entity:
        entities.append(" ".join(current_entity))

    return entities

def load_corpuses(ncbi_path, gen_path, nlp):
    corpus_list = []
    id = 0
    with open(ncbi_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
        for entry in json_data:
            id += 1
            corpus_list.append((id, entry["sentence"], entry["entities"], "NCBI_train", entry["abstract_id"]))
    
    with open(gen_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            doc = Doc(nlp.vocab, words=entry["tokens"])
            sentence = doc.text
            entities = extract_entities(entry)
            id += 1
            corpus_list.append((id, sentence, entities, "generated_train", None))
    
    return corpus_list


def extract_syntax_features(nlp, ids, sentences, entities, corpus_labels, abstract_ids,
                            output_path, rewrite):
    features = []
    if os.path.exists(output_path) and not rewrite:
        return 
    for (id, sent, entity, corpus_name, abstract_id) in zip(ids, sentences, entities, corpus_labels, abstract_ids):
        doc = nlp(sent)
        pos_tags = [token.pos_ for token in doc]
        dep_rels = [token.dep_ for token in doc]
        parents = [token.head.i for token in doc] #index of parent token

        features.append({
            "abstract_id": abstract_id,
            "id": id,
            "sentence": doc.text,
            "entities": entity,
            "corpus": corpus_name,
            "pos": pos_tags,
            "dep": dep_rels,
            "parents": parents,
            
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for feat in features:
            f.write(json.dumps(feat) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Parsing")
    parser.add_argument('--input_file', type=str, required=False, help='Path to where MeSH NCBI train json is saved', default="data/MeSH_NCBI/NCBItrainset_corpus.txt")
    parser.add_argument('--parsed_mesh_file', type=str, required=False, help='Path to where to save parsed mesh NCBI train json', default="data/MeSH_NCBI/sm/ncbi_ner_train.json")
    parser.add_argument('--filtered_parsed_mesh_file', type=str, required=False, help='Directory to where to save filtered parsed mesh NCBI train json', default="data/MeSH_NCBI/sm/")
    parser.add_argument('--stats_file', type=str, required=False, help='Path to where to save statistics of parsed mesh NCBI train json', default="data/MeSH_NCBI/sm/ncbi_ner_sentence_stats.json")
    parser.add_argument('--histograms', type=str, required=False, help='Path to where to save statistics of parsed mesh NCBI train json', default="data/MeSH_NCBI/sm/plots/")
    parser.add_argument('--pcts', type=float, nargs="+", required=False, help='Percentages of abstracts to extract from train (smaller ptcs are subsets from bigger)', default=0.10)  
    parser.add_argument('--spacy_model', type=str, required=False, help='Name of spacy model', default='en_core_web_sm')  
    parser.add_argument('--gen_train_path', type=str, required=False, help='Path to where generated train json is saved', default="data/ncbi/gen2_json/train.json")
    parser.add_argument('--output_path_features', type=str, required=False, help='Path where to save output features', default="data/MeSH_NCBI/sm/syntax_features_sent_tree_head.json")
    parser.add_argument('--rewrite', action='store_true', help='Whether to rewrite existing features file')
    parser.add_argument('--gen_pipeline', action='store_true', help='Run as a part of generation pipeline for loading datasets.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    for path in [args.parsed_mesh_file, args.filtered_parsed_mesh_file, args.stats_file, args.histograms, args.output_path_features]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # PARSE, SAVE AND CREATE STATS OF MESH NCBI 
    # abstracts, annotations = parse_text_file(args.input_file)
    # create_and_save_json(abstracts, annotations, args.parsed_mesh_file, args.model)
    # stats_data = compute_stats(args.parsed_mesh_file, args.stats_file)
    # sentence_lengths = [x["sentence_length"] for x in stats_data]
    # entity_counts = [x["num_entities"] for x in stats_data]
    # punct_counts = [x["num_punctuations"] for x in stats_data]
    # avg_ent_lens = [x["avg_entity_length"] for x in stats_data]
    # global_stats = {
    #     "sentence_length": describe(sentence_lengths),
    #     "num_entities": describe(entity_counts),
    #     "avg_entity_length": describe(avg_ent_lens),
    #     "num_punctuations": describe(punct_counts)
    # }
    # print("\nDataset Summary Statistics:")
    # for k, v in global_stats.items():
    #     print(f"{k}: {v}")
    # plot_histogram(sentence_lengths, "Sentence Length Distribution", "Number of Tokens", "hist_sentence_length.png", args.histograms)
    # plot_histogram(entity_counts, "Number of Entities per Sentence", "Number of Entities", "hist_num_entities.png", args.histograms)
    # plot_histogram(avg_ent_lens, "Average Entity Length per Sentence", "Entity Length (tokens)", "hist_avg_entity_length.png", args.histograms)
    # plot_histogram(punct_counts, "Punctuation Count per Sentence", "Number of Punctuations", "hist_num_punctuations.png", args.histograms)
    # print(f"\nAll histograms saved in folder: {args.histograms}")


    # EXTRACT SYNATX FEATURES FOR BOTH DATASETS
    if args.gen_pipeline:
        nlp = spacy_load_model(args.spacy)
        
        corpus_list = load_corpuses(args.parsed_mesh_file, args.gen_train_path, nlp)
        ids, sentences, entities, corpus_labels, abstract_id = zip(*corpus_list)
        print(f"Loaded {len(sentences)} sentences: "
            f"{corpus_labels.count('NCBI_train')} from NCBI_train and "
            f"{corpus_labels.count('generated_train')} from Generated.")
        
        extract_syntax_features(
            nlp,
            ids, 
            sentences,
            entities,
            corpus_labels,
            abstract_id,
            args.output_path_features,
            args.rewrite
        )

    # EXTRACT % ABSTRACTS 
    # subset_pcts = sorted(args.pcts, reverse=True) #make sure pcts go from bigger to smaller
    # available_abstracts = args.output_path_features
    # previous_pct = 1

    # for pct in subset_pcts:
    #    samples_pct = pct / previous_pct # so that it contains given % from train dataset and not subset it is being extracted from
    #    filtered_abstracts = filter_by_abstract_ids(available_abstracts, args.filtered_parsed_mesh_file, samples_pct, pct)
    #    available_abstracts = filtered_abstracts
    #    previous_pct = pct



"""    
python3 bioNER/utils/parsing_v2.py \
    --input_file data/ncbi/NCBItrainset_corpus/NCBItrainset_corpus.txt \
    --parsed_mesh_file data/ncbi/trf/ncbi_ner_train.json \
    --filtered_parsed_mesh_file data/ncbi/trf/ \
    --stats_file data/ncbi/trf/ncbi_ner_sentence_stats.json \
    --histograms data/ncbi/trf/plots/ \
    --pct 0.50 0.20 0.10 \
    --model en_core_web_trf \
    --gen_train_path data/ncbi/gen2_json/train.json \
    --output_path_features data/ncbi/trf/syntax_features_sent_tree_head.json
"""