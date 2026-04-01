import pandas as pd
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter generated sentences using SNOMED CT concepts."
    )

    parser.add_argument(
        "--concepts_eng_file",
        type=str,
        default="/home/mkeber/syn-bioner/data/KB/SNOMEDCT/concepts_disease_filtered_en.csv",
        help="Path to SNOMED CT English concepts file"
    )

    parser.add_argument(
        "--file_to_filter",
        type=str,
        default="/home/mkeber/syn-bioner/data/synthetic/ncbi/kshot_syn_generation_10pct/corrected_generated_sentences_20260302.jsonl",
        help="Path to input JSONL file to filter"
    )

    parser.add_argument(
        "--file_to_save",
        type=str,
        default="/home/mkeber/syn-bioner/data/synthetic/ncbi/kshot_syn_generation_10pct/filtered_corrected_generated_sentences_20260302.jsonl",
        help="Path to save filtered output JSONL"
    )

    return parser.parse_args()

""" python utils/filter_terms.py --concepts_eng_file /home/mkeber/syn-bioner/data/KB/SNOMEDCT/concepts_disease_filtered_sp.csv\
    --file_to_filter /home/mkeber/syn-bioner/data/synthetic/distemist/kshot_syn_generation_10pct_filter/corrected_generated_sentences_20260326.jsonl\
    --file_to_save /home/mkeber/syn-bioner/data/synthetic/distemist/kshot_syn_generation_10pct_filter/filtered_corrected_generated_sentences_20260326.jsonl """
if __name__ == "__main__":
    args = parse_args()

    concepts_eng_file = args.concepts_eng_file
    file_to_filter = args.file_to_filter
    file_to_save = args.file_to_save

    print("Concepts file:", concepts_eng_file)
    print("Input file:", file_to_filter)
    print("Output file:", file_to_save)

    # Your existing processing logic here
    concepts = pd.read_csv(concepts_eng_file)
    disease_term_ids = [int(i) for i in concepts["concept_id"].values]

    with open(file_to_filter, "r") as f:
        data = [json.loads(line) for line in f.readlines()]

    terms_used = [sent["terms"] for sent in data]

    idx_to_remove = []
    generated_ter_ids = []
    for i, terms in enumerate(terms_used):
        for term, term_id in terms:
            if "DOID" in term_id[0].upper():
                continue
            if int(term_id[0]) not in disease_term_ids:
                idx_to_remove.append(i)
            

    print(f"Num of sentences before filtering = {len(data)}")
    print(f"Num of sentences to remove = {len(idx_to_remove)}")

    written = 0
    with open(file_to_save, "w") as f:
        for i, sentence in enumerate(data):
            if i in idx_to_remove:
                continue
            else:
                f.write(json.dumps(sentence) + "\n")
                written += 1

    print(f"Num of sentences after filtering = {written}")
