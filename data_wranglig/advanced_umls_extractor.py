#!/usr/bin/env python3
"""
Advanced UMLS disease extractor with semantic type filtering.

Uses MRSTY.RRF to properly match concepts with semantic types.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Set
import argparse


class AdvancedUMLSExtractor:
    """Extract disease terms from UMLS with semantic type filtering."""
    
    def __init__(self, umls_path: str):
        """
        Initialize UMLS extractor.
        
        Args:
            umls_path: Path to UMLS META directory
        """
        self.umls_path = Path(umls_path)
        self._validate_files()
    
    def _validate_files(self):
        """Validate that required UMLS files exist."""
        required_files = ['MRCONSO.RRF', 'MRSTY.RRF']
        for filename in required_files:
            filepath = self.umls_path / filename
            if not filepath.exists():
                raise FileNotFoundError(f"{filename} not found at {filepath}")
    
    def _load_semantic_types(self) -> Set[str]:
        """
        Load disorder-related semantic types from MRSTY.RRF.
        
        Returns:
            Set of semantic type codes (e.g., 'DISO', 'DSYN')
        """
        print("Loading semantic types...")
        
        # MRSTY.RRF columns: CUI | TUI | STN | STY | ATUI | CVF
        df_sty = pd.read_csv(
            self.umls_path / 'MRSTY.RRF',
            sep='|',
            header=None,
            usecols=[0, 4],  # CUI, STY
            names=['CUI', 'STY'],
            dtype=str,
            low_memory=False,
            on_bad_lines='skip'
        )

        # AFTER line 52 (after the pd.read_csv for MRSTY.RRF):
        print(f"Total rows in MRSTY.RRF: {len(df_sty)}")
        print(f"Unique CUIs in MRSTY: {df_sty['CUI'].nunique()}")
        print(f"Unique STY values: {df_sty['STY'].nunique()}")
        print(f"Sample STY values: {df_sty['STY'].unique()[:20]}")  # First 20 semantic types
        print(f"STY value counts:\n{df_sty['STY'].value_counts().head(20)}")
        
        # Filter for disorder-related semantic types
        disorder_keywords = ['disease', 'disorder', 'syndrome', 'condition', 'neoplasm', 'lesion']
        disorder_sty = set()
        
        for sty in df_sty['STY'].unique():
            if pd.notna(sty):
                if any(keyword in str(sty).lower() for keyword in disorder_keywords):
                    disorder_sty.add(sty)
        
        print(f"Found semantic types: {sorted(disorder_sty)}")
        return disorder_sty, df_sty
    
    def extract_diseases(
        self,
        language: str = 'FRE',
        output_file: str = 'umls_diseases.csv',
        min_term_length: int = 3
    ) -> pd.DataFrame:
        """
        Extract disease terms from UMLS.
        
        Args:
            language: Language code (FRE, ENG, SPA, etc.)
            output_file: Output CSV file path
            min_term_length: Minimum term length to include
        
        Returns:
            DataFrame with extracted diseases
        """
        print(f"\n{'='*70}")
        print(f"Extracting {language} disease terms from UMLS")
        print(f"{'='*70}\n")
        
        # Step 1: Load semantic types
        disorder_sty, df_sty = self._load_semantic_types()
        
        if not disorder_sty:
            print("⚠ No disorder semantic types found. Using fallback filtering.")
            disorder_sty = {'DISO', 'DSYN', 'NEOP', 'PATF'}
        
        # Step 2: Read MRCONSO.RRF
        print(f"Reading MRCONSO.RRF (this may take a moment)...")
        
        df_conso = pd.read_csv(
            self.umls_path / 'MRCONSO.RRF',
            sep='|',
            header=None,
            usecols=[0, 1, 11, 12, 13, 14],
            names=['CUI', 'LAT', 'SAB', 'TTY', 'CODE', 'STR'],
            dtype=str,
            low_memory=False,
            on_bad_lines='skip'
        )
        
        print(f"Total concepts loaded: {len(df_conso)}")
        
        # Step 3: Filter by language
        df_lang = df_conso[df_conso['LAT'].str.upper() == language.upper()].copy()
        print(f"Concepts in {language}: {len(df_lang)}")
        
        # Step 4: Match with semantic types
        # Create CUI to semantic type mapping
        cui_sty_map = df_sty[df_sty['STY'].isin(disorder_sty)].drop_duplicates(subset=['CUI'])
        print(f"Concepts with disorder semantic types: {len(cui_sty_map)}")
        
        # Filter concepts that have disorder semantic types
        df_diseases = df_lang[df_lang['CUI'].isin(cui_sty_map['CUI'])].copy()
        print(f"Disease concepts in {language}: {len(df_diseases)}")
        print(f"\nDiagnostic Info:")
        print(f"Total unique CUIs in MRCONSO: {df_conso['CUI'].nunique()}")
        print(f"Total unique CUIs in MRSTY: {df_sty['CUI'].nunique()}")
        print(f"CUIs that match between both files: {df_conso['CUI'].isin(df_sty['CUI']).sum()}")
        print(f"\nSample MRCONSO CUIs: {df_conso['CUI'].head(5).tolist()}")
        print(f"Sample MRSTY CUIs: {df_sty['CUI'].head(5).tolist()}")
        # Step 5: Clean and prepare output
        # Filter by minimum term length
        df_diseases = df_diseases[df_diseases['STR'].str.len() >= min_term_length].copy()
        
        # Remove duplicates (keep preferred terms - first occurrence per CUI)
        df_diseases = df_diseases.drop_duplicates(subset=['CUI'], keep='first')
        
        # Prepare final output
        output_df = df_diseases[['STR', 'CUI', 'SAB']].copy()
        output_df.columns = ['concept_name', 'concept_id', 'concept_type']
        
        # Sort alphabetically
        output_df = output_df.sort_values('concept_name').reset_index(drop=True)
        
        # Step 6: Save output
        output_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n✓ Results saved to: {output_file}")
        print(f"✓ Total disease concepts extracted: {len(output_df)}")
        
        return output_df
    
    def extract_snomed_diseases(
        self,
        language: str = 'FRE',
        output_file: str = 'umls_snomed_diseases.csv'
    ) -> pd.DataFrame:
        """
        Extract ONLY SNOMED CT disease terms (like your 4274025 example).
        
        Args:
            language: Language code
            output_file: Output CSV file path
        
        Returns:
            DataFrame with SNOMED disease concepts
        """
        print(f"\n{'='*70}")
        print(f"Extracting SNOMED CT {language} disease terms")
        print(f"{'='*70}\n")
        
        # Load semantic types
        disorder_sty, df_sty = self._load_semantic_types()
        
        if not disorder_sty:
            disorder_sty = {'DISO', 'DSYN', 'NEOP', 'PATF'}
        
        # Read MRCONSO.RRF
        print(f"Reading MRCONSO.RRF...")
        
        df_conso = pd.read_csv(
            self.umls_path / 'MRCONSO.RRF',
            sep='|',
            header=None,
            usecols=[0, 1, 11, 12, 13, 14],
            names=['CUI', 'LAT', 'SAB', 'TTY', 'CODE', 'STR'],
            dtype=str,
            low_memory=False,
            on_bad_lines='skip'
        )
        
        # Filter: SNOMED + Language + Disorders
        df_snomed = df_conso[
            (df_conso['SAB'].str.contains('SNOMED', case=False, na=False)) &
            (df_conso['LAT'].str.upper() == language.upper())
        ].copy()
        
        print(f"SNOMED concepts in {language}: {len(df_snomed)}")
        
        # Match with disorder semantic types
        cui_sty_map = df_sty[df_sty['STY'].isin(disorder_sty)].drop_duplicates(subset=['CUI'])
        df_snomed = df_snomed[df_snomed['CUI'].isin(cui_sty_map['CUI'])].copy()
        
        # Remove duplicates per CUI
        df_snomed = df_snomed.drop_duplicates(subset=['CUI'], keep='first')
        
        # Prepare output
        output_df = df_snomed[['STR', 'CODE', 'SAB']].copy()
        output_df.columns = ['concept_name', 'concept_id', 'concept_type']
        output_df = output_df.sort_values('concept_name').reset_index(drop=True)
        
        # Save
        output_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n✓ SNOMED diseases saved to: {output_file}")
        print(f"✓ Total SNOMED disease concepts: {len(output_df)}")
        
        return output_df


def main():
    parser = argparse.ArgumentParser(
        description='Extract disease terms from UMLS metathesaurus',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Extract French diseases from all sources
  python advanced_umls_extractor.py --language FRE

  # Extract SNOMED CT diseases only
  python advanced_umls_extractor.py --snomed-only --language FRE

  # Extract English diseases with custom output
  python advanced_umls_extractor.py --language ENG --output diseases_en.csv
        '''
    )
    
    parser.add_argument(
        '--umls-path',
        default='data/KB/umls/2025AB/META',
        help='Path to UMLS META directory'
    )
    parser.add_argument(
        '--language',
        default='FRE',
        help='Language code (FRE, ENG, SPA, GER, POR, ITA, etc.)'
    )
    parser.add_argument(
        '--output',
        default='umls_diseases.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--snomed-only',
        action='store_true',
        help='Extract SNOMED CT diseases only'
    )
    
    args = parser.parse_args()
    
    try:
        extractor = AdvancedUMLSExtractor(args.umls_path)
        
        if args.snomed_only:
            output_file = args.output.replace('.csv', '_snomed.csv')
            df = extractor.extract_snomed_diseases(
                language=args.language,
                output_file=output_file
            )
        else:
            df = extractor.extract_diseases(
                language=args.language,
                output_file=args.output
            )
        
        # Display sample
        print(f"\n{'='*70}")
        print(f"Sample of extracted diseases (first 15 rows):")
        print(f"{'='*70}")
        print(df.head(15).to_string(index=False))
        
        if len(df) > 15:
            print(f"\n... and {len(df) - 15} more rows")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
