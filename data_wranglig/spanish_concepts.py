import pandas as pd
import numpy as np
df = pd.read_csv('/home/mkeber/syn-bioner/data/KB/SNOMEDCT/CONCEPT_RELATIONSHIP.csv',
                 delimiter='\t', on_bad_lines='skip')
df_class = pd.read_csv('/home/mkeber/syn-bioner/data/KB/SNOMEDCT/CONCEPT_CLASS.csv', 
                          delimiter='\t', on_bad_lines='skip')
diseases = []
for class_id in df_class['concept_class_id']. unique():
    if "finding" in class_id.lower() or "disease" in class_id.lower(): diseases.append(class_id)


classes = df_class[df_class['concept_class_id'].isin(diseases)]['concept_class_concept_id']
df_ancestor = pd.read_csv('/home/mkeber/syn-bioner/data/KB/SNOMEDCT/CONCEPT_ANCESTOR.csv', 
                          delimiter='\t', on_bad_lines='skip')

desecendants = df_ancestor[df_ancestor['ancestor_concept_id'].isin([441840, 4274025])]['descendant_concept_id'].values
df_concepts = pd.read_csv('/home/mkeber/syn-bioner/data/KB/SNOMEDCT/CONCEPT_SYNONYM.csv', 
                          delimiter='\t', on_bad_lines='skip')
sp_df = df_concepts[df_concepts['language_concept_id'] == 4182511]
output = sp_df[sp_df['concept_id'].isin(desecendants)]

output[['concept_synonym_name', 'concept_id']].to_csv(
    '/home/mkeber/syn-bioner/data/KB/SNOMEDCT/concepts_filtered_sp.csv', index=False)
