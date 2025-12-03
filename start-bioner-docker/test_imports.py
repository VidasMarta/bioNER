#!/usr/bin/env python3
"""
Robust import tester for Docker + Conda

Tests:
1. Python imports in BASE environment
2. spaCy model loading in BASE environment
3. Imports inside conda environment "gen"

Any failure exits with code 1.
"""

import importlib
import subprocess
import sys
import shlex

# ----------------------------------------------------------
# Utilities
# ----------------------------------------------------------

def run(cmd):
    """Run a shell command. Return (out, err, code)."""
    p = subprocess.Popen(cmd, shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         text=True)
    out, err = p.communicate()
    return out.strip(), err.strip(), p.returncode

def ok(msg):
    print(f"\033[92m[OK]\033[0m     {msg}")

def fail(msg):
    print(f"\033[91m[FAILED]\033[0m {msg}")

failed = []


# ==========================================================
# 1. BASE ENV IMPORTS
# ==========================================================

print("\n=== Testing imports in BASE environment ===")

base_packages = [
    "spacy",
    "numpy",
    "sklearn",
    "tqdm",
    "transformers",
    "TorchCRF",
    "seqeval",
    "matplotlib",
    "joblib",
    "optuna",
    "datasets",
    "umap",
    "seaborn",
    "pandas",
    "regex",
    "yaml",
]

for pkg in base_packages:
    try:
        importlib.import_module(pkg)
        ok(pkg)
    except Exception as e:
        fail(f"{pkg} -> {e}")
        failed.append(f"base:{pkg}")


# ==========================================================
# 2. BASE ENV SPACY MODEL LOAD TESTS
# ==========================================================

print("\n=== Testing spaCy model loading in BASE environment ===")

spacy_models = [
    "en_core_web_sm",
    "en_core_web_md",
    "en_core_web_lg",
    "en_core_web_trf",
]

try:
    import spacy
except Exception as e:
    fail(f"spacy not importable: {e}")
    failed.append("base:spacy_models")
    spacy = None

if spacy:
    for model in spacy_models:
        try:
            spacy.load(model)
            ok(f"spacy.load('{model}')")
        except Exception as e:
            fail(f"spacy.load('{model}') -> {e}")
            failed.append(f"base:{model}")


# ==========================================================
# 3. CHECK CONDA ENV EXISTS
# ==========================================================

print("\n=== Checking conda environment 'gen' ===")

out, err, code = run("conda env list")

if code != 0:
    fail(f"'conda env list' failed: {err}")
    failed.append("conda_env_list")
elif "gen" not in out:
    fail("Conda environment 'gen' NOT found")
    failed.append("conda_env_gen_missing")
else:
    ok("Conda environment 'gen' exists")


# ==========================================================
# 4. IMPORTS INSIDE CONDA ENV "gen"
# ==========================================================

print("\n=== Testing imports inside conda env 'gen' ===")

gen_packages = [
    "karateclub",
    "networkx",
    "numpy",
    "tqdm",
    "yaml",
    "sklearn",
    "regex",
    "umap",
    "obonet",
    "requests",
    "matplotlib"
]

# Python code executed inside conda env
inside_code = "import importlib\nerrors=[]\n"

for pkg in gen_packages:
    inside_code += f"""
try:
    importlib.import_module("{pkg}")
    print("[OK] {pkg}")
except Exception as e:
    print("[FAILED] {pkg} ->", e)
    errors.append("{pkg}")
"""

inside_code += "\nimport sys; sys.exit(1 if errors else 0)\n"

quoted = shlex.quote(inside_code)

cmd = f"conda run -n gen python -c {quoted}"

out, err, code = run(cmd)
print(out)

if code != 0:
    fail("One or more imports FAILED in conda env 'gen'")
    if err:
        print("stderr:", err)
    failed.append("conda_imports_gen")
else:
    ok("All imports succeeded inside conda env 'gen'")


# ==========================================================
# SUMMARY
# ==========================================================

print("\n=== SUMMARY ===")

if failed:
    print("\033[91mImport Test FAILED\033[0m")
    for f in failed:
        print(" -", f)
    sys.exit(1)

print("\033[92mAll tests passed successfully!\033[0m")
sys.exit(0)
