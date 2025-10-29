import json
import re
import spacy
from tqdm import tqdm

# Load spaCy English model
nlp = spacy.load("en_core_web_trf")

input_file = "/home/martavidas/Documents/FER/Diplomski/Diplomski/data/MeSH_NCBI/NCBItrainset_corpus.txt"
output_file = "/home/martavidas/Documents/FER/Diplomski/Diplomski/data/MeSH_NCBI/ncbi_ner_train.json"

# Containers
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

# Step 2: Create sentence-level JSON objects
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
