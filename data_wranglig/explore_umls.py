#!/usr/bin/env python3
"""
Quick UMLS Explorer - Inspect and test UMLS extraction.

Use this to:
1. Check available languages and sources
2. Find semantic types
3. Search for specific concepts
4. Validate extraction before running full extraction
"""

import pandas as pd
from pathlib import Path
import argparse


class UMLSExplorer:
    """Explore UMLS structure and validate data."""
    
    def __init__(self, umls_path: str):
        self.umls_path = Path(umls_path)
    
    def explore_languages(self, sample_size: int = 100000) -> pd.DataFrame:
        """Find all available languages in MRCONSO.RRF."""
        print("Exploring available languages...\n")
        
        df = pd.read_csv(
            self.umls_path / 'MRCONSO.RRF',
            sep='|',
            header=None,
            usecols=[1],  # LAT
            names=['LAT'],
            dtype=str,
            nrows=sample_size,
            on_bad_lines='skip'
        )
        
        lang_counts = df['LAT'].value_counts()
        print("Languages found:")
        print("-" * 50)
        for lang, count in lang_counts.items():
            print(f"  {lang:10s}: {count:8d} terms")
        
        return lang_counts
    
    def explore_sources(self, sample_size: int = 100000) -> pd.DataFrame:
        """Find all available sources in MRCONSO.RRF."""
        print("\nExploring available sources...\n")
        
        df = pd.read_csv(
            self.umls_path / 'MRCONSO.RRF',
            sep='|',
            header=None,
            usecols=[11],  # SAB
            names=['SAB'],
            dtype=str,
            nrows=sample_size,
            on_bad_lines='skip'
        )
        
        source_counts = df['SAB'].value_counts()
        print("Sources found:")
        print("-" * 50)
        for source, count in source_counts.items():
            print(f"  {source:15s}: {count:8d} terms")
        
        return source_counts
    
    def explore_semantic_types(self) -> pd.DataFrame:
        """Find all semantic types in MRSTY.RRF."""
        print("\nExploring semantic types...\n")
        
        try:
            df = pd.read_csv(
                self.umls_path / 'MRSTY.RRF',
                sep='|',
                header=None,
                usecols=[4],  # STY
                names=['STY'],
                dtype=str,
                on_bad_lines='skip'
            )
            
            sty_counts = df['STY'].value_counts()
            print("Semantic types found:")
            print("-" * 50)
            
            disorder_keywords = ['disease', 'disorder', 'syndrome', 'condition', 'neoplasm']
            
            for sty, count in sty_counts.items():
                is_disorder = any(kw in str(sty).lower() for kw in disorder_keywords)
                marker = " ← DISORDER" if is_disorder else ""
                print(f"  {sty:30s}: {count:8d} concepts{marker}")
            
            return sty_counts
            
        except FileNotFoundError:
            print("⚠ MRSTY.RRF not found. Cannot explore semantic types.")
            return None
    
    def search_concept(self, search_term: str, language: str = 'FRE', limit: int = 20):
        """Search for a specific concept by name."""
        print(f"\nSearching for: '{search_term}' (Language: {language}, Limit: {limit} results)\n")
        
        df = pd.read_csv(
            self.umls_path / 'MRCONSO.RRF',
            sep='|',
            header=None,
            usecols=[0, 1, 11, 13, 14],
            names=['CUI', 'LAT', 'SAB', 'CODE', 'STR'],
            dtype=str,
            on_bad_lines='skip'
        )
        
        # Filter by language
        df = df[df['LAT'].str.upper() == language.upper()]
        
        # Filter by search term (case-insensitive)
        search_lower = search_term.lower()
        mask = df['STR'].str.lower().str.contains(search_lower, na=False)
        results = df[mask].head(limit)
        
        if len(results) == 0:
            print(f"❌ No results found for '{search_term}'")
            return
        
        print(f"Found {len(results)} results:\n")
        print("-" * 100)
        for idx, row in results.iterrows():
            print(f"CUI: {row['CUI']:12s} | CODE: {row['CODE']:15s} | SOURCE: {row['SAB']:15s}")
            print(f"  → {row['STR']}")
            print()
    
    def get_concept_info(self, cui: str) -> dict:
        """Get all information about a specific concept."""
        print(f"\nGetting info for CUI: {cui}\n")
        
        # Get from MRCONSO
        df_conso = pd.read_csv(
            self.umls_path / 'MRCONSO.RRF',
            sep='|',
            header=None,
            usecols=[0, 1, 11, 12, 13, 14],
            names=['CUI', 'LAT', 'SAB', 'TTY', 'CODE', 'STR'],
            dtype=str,
            on_bad_lines='skip'
        )
        
        concept_terms = df_conso[df_conso['CUI'] == cui]
        
        if len(concept_terms) == 0:
            print(f"❌ CUI {cui} not found")
            return None
        
        print(f"Concept: {cui}")
        print(f"Found in {concept_terms['LAT'].nunique()} languages, {concept_terms['SAB'].nunique()} sources")
        print("\nTerms by language/source:")
        print("-" * 100)
        
        for idx, row in concept_terms.iterrows():
            print(f"  {row['LAT']:5s} | {row['SAB']:15s} | CODE: {row['CODE']:15s} | {row['STR']}")
        
        # Try to get semantic types
        try:
            df_sty = pd.read_csv(
                self.umls_path / 'MRSTY.RRF',
                sep='|',
                header=None,
                usecols=[0, 4],
                names=['CUI', 'STY'],
                dtype=str,
                on_bad_lines='skip'
            )
            
            sty = df_sty[df_sty['CUI'] == cui]['STY'].unique()
            if len(sty) > 0:
                print(f"\nSemantic Types: {', '.join(sty)}")
        except:
            pass
        
        return concept_terms
    
    def validate_extraction_params(self, language: str, sample_size: int = 100000):
        """Validate parameters before running full extraction."""
        print(f"\nValidating parameters for language: {language}\n")
        print("-" * 60)
        
        df = pd.read_csv(
            self.umls_path / 'MRCONSO.RRF',
            sep='|',
            header=None,
            usecols=[0, 1, 11],
            names=['CUI', 'LAT', 'SAB'],
            dtype=str,
            nrows=sample_size,
            on_bad_lines='skip'
        )
        
        # Check language
        lang_count = len(df[df['LAT'].str.upper() == language.upper()])
        print(f"Concepts in {language}: {lang_count} (sampled)")
        
        if lang_count == 0:
            print(f"⚠ WARNING: No concepts found for language {language}")
            print(f"  Available languages: {df['LAT'].unique()}")
            return False
        
        # Check sources
        sources = df[df['LAT'].str.upper() == language.upper()]['SAB'].value_counts()
        print(f"\nTop sources for {language}:")
        for source, count in sources.head(10).items():
            print(f"  {source:15s}: {count:6d}")
        
        print("\n✓ Parameters look valid")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Explore UMLS data structure and test extraction parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Explore all languages and sources
  python explore_umls.py --explore

  # Search for a concept
  python explore_umls.py --search "infarctus" --language FRE

  # Get detailed info about a CUI
  python explore_umls.py --cui C0020538

  # Validate parameters before extraction
  python explore_umls.py --validate FRE
        '''
    )
    
    parser.add_argument(
        '--umls-path',
        default='data/KB/umls/2025AB/META',
        help='Path to UMLS META directory'
    )
    
    # Exploration options
    parser.add_argument(
        '--explore',
        action='store_true',
        help='Explore all languages, sources, and semantic types'
    )
    parser.add_argument(
        '--languages',
        action='store_true',
        help='List all available languages'
    )
    parser.add_argument(
        '--sources',
        action='store_true',
        help='List all available sources'
    )
    parser.add_argument(
        '--semantic-types',
        action='store_true',
        help='List all semantic types'
    )
    
    # Search/lookup options
    parser.add_argument(
        '--search',
        metavar='TERM',
        help='Search for a concept by term name'
    )
    parser.add_argument(
        '--cui',
        metavar='CUI',
        help='Get detailed info about a specific CUI'
    )
    parser.add_argument(
        '--language',
        default='FRE',
        help='Language code for search/validation (default: FRE)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=20,
        help='Limit number of search results (default: 20)'
    )
    
    # Validation option
    parser.add_argument(
        '--validate',
        metavar='LANGUAGE',
        help='Validate extraction parameters for a language'
    )
    
    args = parser.parse_args()
    
    try:
        explorer = UMLSExplorer(args.umls_path)
        
        if args.explore:
            explorer.explore_languages()
            explorer.explore_sources()
            explorer.explore_semantic_types()
        
        if args.languages:
            explorer.explore_languages()
        
        if args.sources:
            explorer.explore_sources()
        
        if args.semantic_types:
            explorer.explore_semantic_types()
        
        if args.search:
            explorer.search_concept(args.search, args.language, args.limit)
        
        if args.cui:
            explorer.get_concept_info(args.cui)
        
        if args.validate:
            explorer.validate_extraction_params(args.validate)
        
        # Default: show help if no options
        if not any([args.explore, args.languages, args.sources, args.semantic_types,
                   args.search, args.cui, args.validate]):
            parser.print_help()
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
