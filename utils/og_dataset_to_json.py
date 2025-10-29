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

            json_data.append({
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

def filter_by_abstract_ids(input_file, output_file, sample_ratio):
    # Load the full dataset
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Collect unique abstract IDs
    abstract_ids = sorted({item["abstract_id"] for item in data})
    print(f"Total abstracts: {len(abstract_ids)}")

    # Randomly sample args.pct of abstract IDs
    sample_size = max(1, int(len(abstract_ids) * sample_ratio))
    sampled_ids = set(random.sample(abstract_ids, sample_size))
    print(f"Selected {len(sampled_ids)} abstracts for the 10% sample.")

    # Filter all sentences that belong to sampled abstracts
    filtered_data = [item for item in data if item["abstract_id"] in sampled_ids]

    # Save to new JSON file
    with open(output_file + f"ncbi_ner_train_{sample_ratio*100:.0f}pct.json", "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(filtered_data)} sentences to {output_file}")


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


def parse_args():
    parser = argparse.ArgumentParser(description="Parsing")
    parser.add_argument('--input_file', type=str, required=False, help='Path to where MeSH NCBI train json is saved', default="/home/martavidas/Documents/FER/Diplomski/Diplomski/data/MeSH_NCBI/NCBItrainset_corpus.txt")
    parser.add_argument('--parsed_mesh_file', type=str, required=False, help='Path to where to save parsed mesh NCBI train json', default="/home/martavidas/Documents/FER/Diplomski/Diplomski/data/MeSH_NCBI/ncbi_ner_train.json")
    parser.add_argument('--filtered_parsed_mesh_file', type=str, required=False, help='Path to where to save filtered parsed mesh NCBI train json', default="/home/martavidas/Documents/FER/Diplomski/Diplomski/data/MeSH_NCBI/")
    parser.add_argument('--stats_file', type=str, required=False, help='Path to where to save statistics of parsed mesh NCBI train json', default="/home/martavidas/Documents/FER/Diplomski/Diplomski/data/MeSH_NCBI/ncbi_ner_sentence_stats.json")
    parser.add_argument('--histograms', type=str, required=False, help='Path to where to save statistics of parsed mesh NCBI train json', default="/home/martavidas/Documents/FER/Diplomski/Diplomski/data/MeSH_NCBI/plots/")
    parser.add_argument('--pct', type=float, required=False, help='Percentage of abstracts to extract', default=0.10)  
    parser.add_argument('--model', type=str, required=False, help='Name of spacy model', default='en_core_web_sm')    
    return parser.parse_args()

def extract_args():
    args = parse_args()
    input_file = args.input_file
    parsed_mesh_file = args.parsed_mesh_file
    model = args.model
    filtered_parsed_mesh_file = args.filtered_parsed_mesh_file
    pct = args.pct
    stats_file = args.stats_file
    histograms = args.histograms
    for path in [args.parsed_mesh_file, args.filtered_parsed_mesh_file, args.stats_file, args.histograms]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return input_file, parsed_mesh_file, model, filtered_parsed_mesh_file, pct, stats_file, histograms


if __name__ == "__main__":
    input_file, parsed_mesh_file, model, filtered_parsed_mesh_file, pct, stats_file, histograms= extract_args()

    ''''abstracts, annotations = parse_text_file(input_file)
    create_and_save_json(abstracts, annotations, parsed_mesh_file, model)
    filter_by_abstract_ids(parsed_mesh_file, filtered_parsed_mesh_file, pct)'''
    stats_data = compute_stats(parsed_mesh_file, stats_file)

    # --- Aggregate statistics ---
    def describe(values):
        return {
            "mean": round(statistics.mean(values), 2),
            "min": min(values),
            "max": max(values),
            "stdev": round(statistics.stdev(values), 2) if len(values) > 1 else 0
        }

    sentence_lengths = [x["sentence_length"] for x in stats_data]
    entity_counts = [x["num_entities"] for x in stats_data]
    punct_counts = [x["num_punctuations"] for x in stats_data]
    avg_ent_lens = [x["avg_entity_length"] for x in stats_data]

    global_stats = {
        "sentence_length": describe(sentence_lengths),
        "num_entities": describe(entity_counts),
        "avg_entity_length": describe(avg_ent_lens),
        "num_punctuations": describe(punct_counts)
    }

    print("\nDataset Summary Statistics:")
    for k, v in global_stats.items():
        print(f"{k}: {v}")

    plot_histogram(sentence_lengths, "Sentence Length Distribution", "Number of Tokens", "hist_sentence_length.png", histograms)
    plot_histogram(entity_counts, "Number of Entities per Sentence", "Number of Entities", "hist_num_entities.png", histograms)
    plot_histogram(avg_ent_lens, "Average Entity Length per Sentence", "Entity Length (tokens)", "hist_avg_entity_length.png", histograms)
    plot_histogram(punct_counts, "Punctuation Count per Sentence", "Number of Punctuations", "hist_num_punctuations.png", histograms)

    print(f"\nAll histograms saved in folder: {histograms}")

