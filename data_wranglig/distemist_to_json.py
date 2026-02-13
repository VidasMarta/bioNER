import glob
import os
import json
import spacy
import random

def load_distemist(data_dir):
    # Collect text files
    texts = {}
    for p in glob.glob(os.path.join(data_dir, "text_files/*.txt")):
        fname = os.path.basename(p)
        with open(p, "r", encoding="utf-8") as f:
            texts[fname] = f.read()

    # Collect annotations
    ann = {}
    for p in glob.glob(os.path.join(data_dir, "subtrack1_entities/*.tsv")):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                filename, mark, label, off0, off1, span = line.strip().split("\t")
                start = int(off0)
                end = int(off1)
                ann.setdefault(filename, []).append({
                    "start": start,
                    "end": end,
                    "text": span
                })
    return texts, ann

def parse_to_bio(texts, ann, nlp):
    data_out = []
    for fname, text in texts.items():
        doc = nlp(text)
        ents = ann.get(fname, [])
        ents.sort(key=lambda x: x["start"])

        for sent in doc.sents:
            sent_text = sent.text
            tokens = [tok.text for tok in sent]
            tags = [2] * len(tokens)

            for e in ents:
                if e["end"] <= sent.start_char or e["start"] >= sent.end_char:
                    continue

                for i, tok in enumerate(sent):
                    tok_start = tok.idx
                    tok_end = tok.idx + len(tok.text)

                    if tok_end <= e["start"]:
                        continue
                    if tok_start >= e["end"]:
                        break

                    if tok_start == e["start"]:
                        tags[i] = 0 #B-Disease
                    else:
                        tags[i] = 1 #I-Disease

            data_out.append({
                "tokens": tokens,
                "tags": tags,
                "text_file": fname #Ovo možda umjesto abstract_ida, pa za 10, 20, 50 % uzimati iz fileova
            })
    return data_out

if __name__=='__main__':
    nlp = spacy.load("es_core_news_trf", disable=["ner", "tagger", "parser", "lemmatizer"])
    nlp.add_pipe("sentencizer")
    train_texts, train_ann = load_distemist("/home/martavidas/Documents/FER/Diplomski/Diplomski/data/distemist/training")
    train_data = parse_to_bio(train_texts, train_ann, nlp)

    # Write JSON
    with open("distemist_train.json", "w", encoding="utf-8") as f:
        for row in train_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Train JSON saved")

    # If you want a dev split (e.g., 10%), shuffle and split
    random.shuffle(train_data)
    split_idx = int(len(train_data) * 0.9)
    train_split = train_data[:split_idx]
    dev_split = train_data[split_idx:]

    with open("distemist_train_split.json","w",encoding="utf-8") as f:
        for row in train_split:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open("distemist_dev_split.json","w",encoding="utf-8") as f:
        for row in dev_split:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("Train/Dev split saved")

    # Similarly parse test set:
    test_texts, test_ann = load_distemist("/home/martavidas/Documents/FER/Diplomski/Diplomski/data/distemist/test_annotated")
    test_data = parse_to_bio(test_texts, test_ann, nlp)
    with open("distemist_test.json","w",encoding="utf-8") as f:
        for row in test_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("Test JSON saved")
