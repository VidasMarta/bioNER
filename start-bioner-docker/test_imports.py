#!/usr/bin/env python3
import importlib

packages = [
    "karateclub",
    "transformers",
    "networkx",
    "sklearn",          # scikit-learn
    "matplotlib",
    "spacy",
    "jupyterlab",
    "nltk",
    "optuna",
    "umap",             # umap-learn
    "seaborn",
    "psycopg2",
    "requests",
    "urllib3",
    "tqdm",
    "pandas",
    "regex",
    "numpy",
    "joblib",
    "seqeval",
    "yaml",             # PyYAML
    "obonet"
]

failed = []

print("=== Testing Python imports ===")

for pkg in packages:
    try:
        importlib.import_module(pkg)
        print(f"[OK]       {pkg}")
    except Exception as e:
        print(f"[FAILED]   {pkg} -> {e}")
        failed.append(pkg)

print("\n=== Summary ===")

if failed:
    print("IMPORTS FAILED:")
    for f in failed:
        print(f" - {f}")
    exit(1)
else:
    print("ALL OK — ALL IMPORTS SUCCESSFUL.")
    exit(0)
