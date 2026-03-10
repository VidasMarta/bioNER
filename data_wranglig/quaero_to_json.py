import json
import os
import random
import subprocess
import sys
import xml.etree.ElementTree as ET
import spacy
from collections import defaultdict
import argparse
from dataset_to_json import argparse_args, save_to_json_line, load_yaml_with_env

def parse_bioc_file(xml_path):
    """
    Parses BioC XML file and returns:
    texts = {doc_id: text}
    annotations = {doc_id: [ {start, end, type} ]}
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()

    texts = {}
    annotations = {}

    for document in root.iter("document"):
        doc_id = document.find("id").text
        full_text = ""
        ann_list = []

        for passage in document.iter("passage"):
            text_elem = passage.find("text")
            if text_elem is None:
                continue

            passage_text = text_elem.text
            #passage_offset = int(passage.find("offset").text)

            full_text += passage_text

            for ann in passage.iter("annotation"):
                entity_type = None
                for infon in ann.iter("infon"):
                    if infon.attrib.get("key") == "type":
                        entity_type = infon.text

                location = ann.find("location")
                start = int(location.attrib["offset"])
                length = int(location.attrib["length"])
                end = start + length

                ann_list.append({
                    "start": start,
                    "end": end,
                    "type": entity_type
                })

        texts[doc_id] = full_text
        annotations[doc_id] = ann_list

    return texts, annotations


def convert_to_bio(texts, annotations, nlp):
    data = []

    for doc_id, text in texts.items():
        doc = nlp(text)
        ents = annotations.get(doc_id, [])

        for sent in doc.sents:
            tokens = [tok.text for tok in sent]
            tags = [2] * len(tokens)
            entities = []

            for ent in ents:
                if ent["type"] != "DISO":
                    continue  # ignore non-DISO entities

                if ent["end"] <= sent.start_char or ent["start"] >= sent.end_char:
                    continue
                
                entity_tokens = []
                for i, tok in enumerate(sent):
                    tok_start = tok.idx
                    tok_end = tok.idx + len(tok.text)

                    if tok_end <= ent["start"]:
                        continue
                    if tok_start >= ent["end"]:
                        break

                    if tok_start == ent["start"]:
                        tags[i] = 0 #B_DISO
                        entity_tokens.append(tok.text)
                    else:
                        tags[i] = 1 # I_DISO
                        entity_tokens.append(tok.text)
                if len(entity_tokens) > 0:
                    entities.append(" ".join(entity_tokens))
            data.append({
                "tokens": tokens,
                "tags": tags, 
                "entities": entities,
                "document_id": doc_id
            })

    return data


def filter_by_text_file(corpus_name, input_file, output_file, sample_ratio, pct_train):
    # Load the full dataset
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # Collect unique document ids
    document_ids = sorted({item["document_id"] for item in data})
    print(f"Total document ids: {len(document_ids)}")

    # Randomly sample args.pct of text files
    sample_size = max(1, int(len(document_ids) * sample_ratio))
    sampled_ids = set(random.sample(document_ids, sample_size))
    print(f"Selected {len(sampled_ids)} document ids for the {float(pct_train)*100:.0f}% sample.")

    # Filter all sentences that belong to sampled text files
    filtered_data = [item for item in data if item["document_id"] in sampled_ids]

    # Save to new JSON file
    filtered = output_file + f"{corpus_name}_ner_train_{float(pct_train)*100:.0f}pct.json"
    save_to_json_line(filtered, filtered_data)
    print(f"Saved {len(filtered_data)} sentences to {output_file}")

    return filtered

def save_json(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} sentences to {output_path}")


if __name__ == "__main__":
    init_args = argparse_args()
    yaml_args = load_yaml_with_env(init_args.config_file,)
    args = argparse.Namespace(**yaml_args)

    if not spacy.util.is_package(args.model):
        subprocess.run([sys.executable, "-m", "spacy", "download", args.model])
    nlp = spacy.load(args.model, disable=["ner", "parser", "tagger"])
    nlp.add_pipe("sentencizer")

    os.makedirs(args.parsed_mesh_folder_emea, exist_ok=True)

    #train
    texts, annotations = parse_bioc_file(args.training_emea)
    data = convert_to_bio(texts, annotations, nlp)
    save_to_json_line(os.path.join(args.parsed_mesh_folder_emea, "emea_train.jsonl"), data)

    #devel
    texts, annotations = parse_bioc_file(args.devel_emea)
    data = convert_to_bio(texts, annotations, nlp)
    save_to_json_line(os.path.join(args.parsed_mesh_folder_emea, "emea_dev.jsonl"), data)

    #test
    texts, annotations = parse_bioc_file(args.test_emea)
    data = convert_to_bio(texts, annotations, nlp)
    save_to_json_line(os.path.join(args.parsed_mesh_folder_emea, "emea_test.jsonl"), data)

    # Filter by text file
    subset_pcts = sorted(args.pcts, reverse=True)
    available_text_files = args.parsed_mesh_folder_emea + "emea_train.jsonl"
    previous_pct = 1
    for pct in subset_pcts:
        samples_pct = float(pct) / float(previous_pct) # so that it contains given % from train dataset and not subset it is being extracted from
        filtered_text_files = filter_by_text_file("emea", available_text_files, args.parsed_mesh_folder_emea, samples_pct, pct)
        available_text_files = filtered_text_files
        previous_pct = pct


    #MEDLINE
    os.makedirs(args.parsed_mesh_folder_medline, exist_ok=True)
    texts, annotations = parse_bioc_file(args.training_medline)
    data = convert_to_bio(texts, annotations, nlp)
    save_to_json_line(os.path.join(args.parsed_mesh_folder_medline, "medline_train.jsonl"), data)

    #devel
    texts, annotations = parse_bioc_file(args.devel_medline)
    data = convert_to_bio(texts, annotations, nlp)
    save_to_json_line(os.path.join(args.parsed_mesh_folder_medline, "medline_dev.jsonl"), data)

    #test
    texts, annotations = parse_bioc_file(args.test_medline)
    data = convert_to_bio(texts, annotations, nlp)
    save_to_json_line(os.path.join(args.parsed_mesh_folder_medline, "medline_test.jsonl"), data)

    # Filter by text file
    subset_pcts = sorted(args.pcts, reverse=True)
    available_text_files = args.parsed_mesh_folder_medline + "medline_train.jsonl"
    previous_pct = 1
    for pct in subset_pcts:
        samples_pct = float(pct) / float(previous_pct) # so that it contains given % from train dataset and not subset it is being extracted from
        filtered_text_files = filter_by_text_file("medline", available_text_files, args.parsed_mesh_folder_medline, samples_pct, pct)
        available_text_files = filtered_text_files
        previous_pct = pct
    