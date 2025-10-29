import argparse
import json
import re
import spacy
from tqdm import tqdm
import random

# Load spaCy English model
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



def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--input_file', type=str, required=False, help='Path to where MeSH NCBI train json is saved', default="/home/martavidas/Documents/FER/Diplomski/Diplomski/data/MeSH_NCBI/NCBItrainset_corpus.txt")
    parser.add_argument('--parsed_mesh_file', type=str, required=False, help='Path to where to save parsed mesh NCBI train json', default="/home/martavidas/Documents/FER/Diplomski/Diplomski/data/MeSH_NCBI/ncbi_ner_train.json")
    parser.add_argument('--filtered_parsed_mesh_file', type=str, required=False, help='Path to where to save filtered parsed mesh NCBI train json', default="/home/martavidas/Documents/FER/Diplomski/Diplomski/data/MeSH_NCBI/")
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
    return input_file, parsed_mesh_file, model, filtered_parsed_mesh_file, pct


if __name__ == "__main__":
    input_file, parsed_mesh_file, model, filtered_parsed_mesh_file, pct = extract_args()

    abstracts, annotations = parse_text_file(input_file)
    create_and_save_json(abstracts, annotations, parsed_mesh_file, model)
    filter_by_abstract_ids(parsed_mesh_file, filtered_parsed_mesh_file, pct)

