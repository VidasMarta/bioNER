#!/usr/bin/env python3
"""
Extract disease/disorder terms from UMLS metathesaurus.
Filters by language and semantic type (e.g., Disorders).

UMLS file format (pipe-delimited, |):
- MRCONSO.RRF: Concept names with language codes
- MRDOC.RRF: Semantic type documentation
"""

import pandas as pd
import os
from pathlib import Path
from typing import List, Set, Dict
import argparse


class UMLSExtractor:
    """Extract disease terms from UMLS metathesaurus."""
    
    # UMLS semantic types for disorders/diseases
    SEMANTIC_TYPES = {
        'DISO': 'Disorder (Disease, syndrome, or medical condition)',
        'DSYN': 'Disease or Syndrome',
        'NEOP': 'Neoplastic Process',
        'PATF': 'Pathologic Function',
    }
    
    # Language codes
    LANGUAGE_CODES = {
        'FRE': 'French',
        'ENG': 'English',
        'SPA': 'Spanish',
        'GER': 'German',
        'POR': 'Portuguese',
        'ITA': 'Italian',
    }
    
    def __init__(self, umls_path: str):
        """
        Initialize UMLS extractor.
        
        Args:
            umls_path: Path to UMLS data directory (e.g., data/KB/umls/2025AB/META)
        """
        self.umls_path = Path(umls_path)
        self.mrconso_file = self.umls_path / 'MRCONSO.RRF'
        self.mrdoc_file = self.umls_path / 'MRDOC.RRF'
        
        if not self.mrconso_file.exists():
            raise FileNotFoundError(f"MRCONSO.RRF not found at {self.mrconso_file}")
    
    def get_semantic_types_for_disorder(self) -> Set[str]:
        """
        Get all semantic type codes associated with disorders/diseases.
        
        Returns:
            Set of semantic type codes
        """
        semantic_types = set()
        
        try:
            # Read MRDOC.RRF to find disease-related semantic types
            # Columns: DOCKEY | VALUE | TYPE | EXPL
            df_doc = pd.read_csv(
                self.mrdoc_file,
                sep='|',
                header=None,
                usecols=[0, 1, 2, 3],
                names=['DOCKEY', 'VALUE', 'TYPE', 'EXPL'],
                dtype=str,
                low_memory=False
            )
            
            # Filter for semantic type rows containing disorder-related keywords
            disorder_keywords = ['disease', 'disorder', 'syndrome', 'condition', 'neoplasm']
            
            for idx, row in df_doc.iterrows():
                if pd.notna(row['TYPE']) and row['TYPE'] == 'semantic_type':
                    expl_lower = str(row['EXPL']).lower() if pd.notna(row['EXPL']) else ''
                    if any(keyword in expl_lower for keyword in disorder_keywords):
                        semantic_types.add(str(row['VALUE']))
            
            print(f"Found semantic types for disorders: {semantic_types}")
            
        except Exception as e:
            print(f"Warning: Could not read MRDOC.RRF: {e}")
            # Fallback to common disorder semantic types
            semantic_types = {'DISO', 'DSYN', 'NEOP', 'PATF'}
        
        return semantic_types
    
    def extract_diseases(
        self,
        language: str = 'FRE',
        semantic_types: List[str] = None,
        output_file: str = 'umls_diseases.csv'
    ) -> pd.DataFrame:
        """
        Extract disease terms from UMLS.
        
        Args:
            language: Language code (e.g., 'FRE' for French, 'ENG' for English)
            semantic_types: List of semantic type codes to filter by.
                          If None, extracts all disorder-related types.
            output_file: Output CSV file path
        
        Returns:
            DataFrame with extracted diseases
        """
        print(f"Extracting {self.LANGUAGE_CODES.get(language, language)} disease terms...")
        
        # If no semantic types specified, auto-detect disorder types
        if semantic_types is None:
            semantic_types = list(self.get_semantic_types_for_disorder())
        
        print(f"Using semantic types: {semantic_types}")
        
        # Read MRCONSO.RRF
        # Columns (key ones):
        # 0: CUI (Concept Unique Identifier)
        # 1: LAT (Language)
        # 2: TS (Term Status)
        # 3: LUI (Lexical Unique Identifier)
        # 4: STT (String Type)
        # 5: SUI (String Unique Identifier)
        # 6: ISPREF (Is Preferred)
        # 7: AUI (Atom Unique Identifier)
        # 8: SAUI (Source Atom Identifier)
        # 9: SCUI (Source Concept Identifier)
        # 10: SDUI (Source Definition Identifier)
        # 11: SAB (Source Abbreviation)
        # 12: TTY (Term Type)
        # 13: CODE (Concept Code)
        # 14: STR (String/Term)
        # 15: SRL (Source Restriction Level)
        # 16: SUPPRESS (Suppression Flag)
        # 17: CVF (Content View Flag)
        
        print("Reading MRCONSO.RRF (this may take a moment for large files)...")
        
        df = pd.read_csv(
            self.mrconso_file,
            sep='|',
            header=None,
            usecols=[0, 1, 11, 12, 13, 14],  # CUI, LAT, SAB, TTY, CODE, STR
            names=['CUI', 'LAT', 'SAB', 'TTY', 'CODE', 'STR'],
            dtype=str,
            low_memory=False,
            on_bad_lines='skip'
        )
        
        print(f"Total rows in MRCONSO.RRF: {len(df)}")
        
        # Filter by language
        df_lang = df[df['LAT'].str.upper() == language.upper()].copy()
        print(f"Rows for language {language}: {len(df_lang)}")
        
        # Now we need to match with semantic types
        # This requires reading concept information from other files
        # For now, we'll filter using source abbreviations commonly associated with disorders
        
        # Filter by source to get only SNOMED CT and other disease-focused sources
        # Common sources: SNOMED CT, ICD10, ICD9, MEDLINEPLUS, etc.
        disorder_sources = ['SNOMED', 'ICD10', 'ICD9CM', 'ICD10CM', 'MEDLINEPLUS']
        df_filtered = df_lang[df_lang['SAB'].str.contains('|'.join(disorder_sources), na=False)]
        
        print(f"Rows after source filtering: {len(df_filtered)}")
        
        # Remove duplicates (keep first occurrence)
        df_filtered = df_filtered.drop_duplicates(subset=['CUI'], keep='first')
        
        # Prepare output columns
        output_df = df_filtered[['STR', 'CUI', 'SAB']].copy()
        output_df.columns = ['concept_name', 'concept_id', 'concept_type']
        
        # Sort by concept name
        output_df = output_df.sort_values('concept_name').reset_index(drop=True)
        
        print(f"Final extracted diseases: {len(output_df)}")
        
        # Save to CSV
        output_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
        return output_df


def main():
    parser = argparse.ArgumentParser(
        description='Extract disease terms from UMLS metathesaurus'
    )
    parser.add_argument(
        '--umls-path',
        default='data/KB/umls/2025AB/META',
        help='Path to UMLS META directory (default: data/KB/umls/2025AB/META)'
    )
    parser.add_argument(
        '--language',
        default='FRE',
        choices=['FRE', 'ENG', 'SPA', 'GER', 'POR', 'ITA'],
        help='Language code (default: FRE for French)'
    )
    parser.add_argument(
        '--output',
        default='umls_diseases.csv',
        help='Output CSV file path (default: umls_diseases.csv)'
    )
    
    args = parser.parse_args()
    
    try:
        extractor = UMLSExtractor(args.umls_path)
        df = extractor.extract_diseases(
            language=args.language,
            output_file=args.output
        )
        
        print(f"\n{'='*60}")
        print(f"Sample of extracted diseases (first 10 rows):")
        print(f"{'='*60}")
        print(df.head(10).to_string(index=False))
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
