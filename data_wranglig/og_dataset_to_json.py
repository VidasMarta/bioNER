import json
import string
import numpy as np
import random
import os
import matplotlib.pyplot as plt

def filter_by_abstract_ids(input_file, output_file, sample_ratio, dataset_name):
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
    with open(output_file + f"{dataset_name}_ner_train_{sample_ratio*100:.0f}pct.json", "w", encoding="utf-8") as f:
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








