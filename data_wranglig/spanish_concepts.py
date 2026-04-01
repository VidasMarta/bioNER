import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Filter SNOMEDCT concepts by language and ancestor.")
    
    # Input Arguments
    parser.add_argument('--lang_id', type=int, default=4182511, 
                        help='Language concept ID (e.g., 4182511 for Spanish)')
    parser.add_argument('--ancestors', type=int, nargs='+', default=[441840, 4274025],
                        help='Ancestor IDs. Note: 441840 is Clinical Finding, 4274025 is Disease.')
    parser.add_argument('--output', type=str, default='concepts_filtered.csv',
                        help='Path and name for the output CSV file')
    
    # Path Arguments (keeping your defaults)
    parser.add_argument('--rel_path', type=str, default='/home/mkeber/syn-bioner/data/KB/SNOMEDCT/CONCEPT_RELATIONSHIP.csv')
    parser.add_argument('--class_path', type=str, default='/home/mkeber/syn-bioner/data/KB/SNOMEDCT/CONCEPT_CLASS.csv')
    parser.add_argument('--anc_path', type=str, default='/home/mkeber/syn-bioner/data/KB/SNOMEDCT/CONCEPT_ANCESTOR.csv')
    parser.add_argument('--syn_path', type=str, default='/home/mkeber/syn-bioner/data/KB/SNOMEDCT/CONCEPT_SYNONYM.csv')

    args = parser.parse_args()

    # Load Data
    df_class = pd.read_csv(args.class_path, delimiter='\t', on_bad_lines='skip')
    df_ancestor = pd.read_csv(args.anc_path, delimiter='\t', on_bad_lines='skip')
    df_concepts = pd.read_csv(args.syn_path, delimiter='\t', on_bad_lines='skip')

    # Filter by ancestor
    descendants = df_ancestor[df_ancestor['ancestor_concept_id'].isin(args.ancestors)]['descendant_concept_id'].values

    # Filter by language and descendants
    sp_df = df_concepts[df_concepts['language_concept_id'] == args.lang_id]
    output = sp_df[sp_df['concept_id'].isin(descendants)]

    # Save Output
    output[['concept_synonym_name', 'concept_id']].to_csv(args.output, index=False)
    print(f"File saved successfully to: {args.output}")

if __name__ == "__main__":
    main()

