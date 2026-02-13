import argparse
import json
import re
import statistics
import spacy
from tqdm import tqdm
import os
import subprocess
import sys
from og_dataset_to_json import *

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
            
            elif "\tCID\t" in line:
                continue

            # Entity annotation lines
            elif re.match(r"^\d+\t\d+\t\d+\t", line):
                parts = line.split("\t")
                pmid = int(parts[0])
                start, end = int(parts[1]), int(parts[2])
                mention = parts[3]
                ent_type = parts[4]      # Chemical or Disease
                code = parts[-1]
                if pmid not in annotations:
                    annotations[pmid] = []
                annotations[pmid].append({
                    "start": start,
                    "end": end,
                    "text": mention,
                    "type": ent_type,
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
                if ent["type"] != "Disease":
                    continue  # ignore chemicals entirely
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


def parse_args():
    parser = argparse.ArgumentParser(description="Parsing")
    parser.add_argument('--input_file', type=str, required=False, help='Path to where MeSH bc5cdr train json is saved', default="bioNER/data/bc5cdr/bc5cdr_train.txt")
    parser.add_argument('--parsed_mesh_file', type=str, required=False, help='Path to where to save parsed mesh bc5cdr train json', default="bioNER/data/bc5cdr/trf/bc5cdr_ner_train.json")
    parser.add_argument('--filtered_parsed_mesh_file', type=str, required=False, help='Path to where to save filtered parsed mesh bc5cdr train json', default="bioNER/data/bc5cdr/trf/")
    parser.add_argument('--stats_file', type=str, required=False, help='Path to where to save statistics of parsed mesh bc5cdr train json', default="bioNER/data/bc5cdr/trf/bc5cdr_ner_sentence_stats.json")
    parser.add_argument('--histograms', type=str, required=False, help='Path to where to save statistics of parsed mesh bc5cdr train json', default="bioNER/data/bc5cdr/trf/plots/")
    parser.add_argument('--pct', type=float, required=False, help='Percentage of abstracts to extract', default=0.10)  
    parser.add_argument('--model', type=str, required=False, help='Name of spacy model', default='en_core_web_trf')    
    return parser.parse_args()

"""
python3 bioNER/data_wranglig/bc5cdr_to_json.py \
    --input_file /home/mkeber/syn-bioner/bioNER/data/bc5cdr/bc5cdr_train.txt \
    --parsed_mesh_file bioNER/data/bc5cdr/trf/bc5cdr_ner_train.json \
    --filtered_parsed_mesh_file bioNER/data/bc5cdr/trf/ \
    --stats_file bioNER/data/bc5cdr/trf/bc5cdr_ner_sentence_stats.json \
    --histograms bioNER/data/bc5cdr/trf/plots/ \
    --pct 0.10 \
    --model en_core_web_trf
"""

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
    if not spacy.util.is_package(model):
        subprocess.run([sys.executable, "-m", "spacy", "download", model])
    abstracts, annotations = parse_text_file(input_file)
    create_and_save_json(abstracts, annotations, parsed_mesh_file, model)
    filter_by_abstract_ids(parsed_mesh_file, filtered_parsed_mesh_file, pct, "bc5cdr")
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

