# %%
import os
import typing
import pandas as pd
import numpy as np
from pprint import pprint

df_ancestor = pd.read_csv('/home/mkeber/syn-bioner/data/KB/SNOMEDCT/CONCEPT_ANCESTOR.csv', delimiter='\t', on_bad_lines='skip')
df_concepts = pd.read_csv('/home/mkeber/syn-bioner/data/KB/SNOMEDCT/CONCEPT.csv', delimiter='\t', on_bad_lines='skip')

df_concepts.head()

descendants = df_ancestor[df_ancestor['ancestor_concept_id'].isin([4274025])]['descendant_concept_id'].values

output = df_concepts[df_concepts['concept_id'].isin(descendants)]

print(output.concept_class_id.unique(),
output.domain_id.unique())

output= output[output.concept_class_id!='Event']

output=output[output.concept_class_id!='Clinical Finding']

output=output[output.concept_class_id!='Procedure']

output.head()

output[['concept_name', 'concept_id']].to_csv('/home/mkeber/syn-bioner/data/KB/SNOMEDCT/concepts_disease_filtered_en.csv', index=False)
print(f"File saved successfully to: '/home/mkeber/syn-bioner/data/KB/SNOMEDCT/concepts_disease_filtered_en.csv' ")