import re
import string
import obonet
from typing import List, Dict, Any, Optional
import argparse

def remove_code_fences(text: str) -> str:
    return re.sub(r'```[\w]*\n?', '', text).strip()

def clean_text(text: str) -> str:
    text = re.sub(r'```+', '', text)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.!?,])([^\s])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<\|.*?\|>', '', text)
    return text.strip()

def check_last_token(tokens_lower: List[str]) -> List[str]:
    tokens_lower[-1] = tokens_lower[-1].rstrip(string.punctuation)
    return tokens_lower

def get_diseases(args: argparse.Namespace):
    graph = obonet.read_obo(args.obo_file_path)
    do_terms = [(data["name"], node) for node, data in graph.nodes(data=True) if "name" in data]
    if args.verbose:
        print(f"Number of nodes (terms): {graph.number_of_nodes()}")
        print(f"Number of edges (relations): {graph.number_of_edges()}")
        print('First 20 terms: ', do_terms[:20])  # Show first 20 terms
    return do_terms