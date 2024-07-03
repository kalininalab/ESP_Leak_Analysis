import pandas as pd
import numpy as np
import random
from os.path import join
import os
import re
import sys
import time
import collections
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from Bio import SeqIO
sys.path.append('./additional_code')
from additional_code.data_preprocessing import *
from additional_code.negative_data_generator import *
warnings.filterwarnings("ignore")
CURRENT_DIR = os.getcwd()
print(CURRENT_DIR)

"""
Attention: to completely run this script the following files should be located in data_ESP folder

UNIPROT_df.pkl			
chebiID_to_inchi.tsv
df_UID_MID.pkl
df_test_with_ESM1b_ts_GNN.pkl
df_train_with_ESM1b_ts_GNN.pkl

Please download all of them from  ESP model repository:
(https://github.com/AlexanderKroll/ESP/tree/main/data/splits)
"""

######
# Extract all experimental data point from train and test set
######

train_set = pd.read_pickle(join(CURRENT_DIR, ".." ,"data", "data_ESP", "df_train_with_ESM1b_ts_GNN.pkl"))
test_set = pd.read_pickle(join(CURRENT_DIR, ".." ,"data", "data_ESP", "df_test_with_ESM1b_ts_GNN.pkl"))
train_set=train_set[train_set["Binding"]==1]
test_set=test_set[test_set["Binding"]==1]

# Drop not used columns
columns_to_drop = ['ECFP_2048', 'ECFP_512','ESM1b', 'GNN rep','ESM1b_ts_mean']
train_set.drop(columns=columns_to_drop, inplace=True)
test_set.drop(columns=columns_to_drop, inplace=True)

# Drop for nan value
train_set.dropna(subset=['ECFP','ESM1b_ts'], inplace=True)
test_set.dropna(subset=['ECFP','ESM1b_ts'], inplace=True)

result, total_samples, test_ratio = two_split_report(train_set, test_set)
print(result.to_string())
print(f"Total number of samples: {total_samples}")
print(f"Ratio of test set to total dataset: {test_ratio}")
# Drop for empty string
train_set = train_set[~(train_set['GNN rep (pretrained)'].str.strip() == '')]
test_set = test_set[~(test_set['GNN rep (pretrained)'].str.strip() == '')]

result, total_samples, test_ratio = two_split_report(train_set, test_set)
print(result.to_string())
print(f"Total number of samples: {total_samples}")
print(f"Ratio of test set to total dataset: {test_ratio}")

train_set.reset_index(drop=True, inplace=True)
test_set.reset_index(drop=True, inplace=True)

data_ESP = pd.concat([train_set, test_set], ignore_index=True)
data_ESP['substrate ID'] = data_ESP['substrate ID'].str.replace('CHEBI:', '')
UNIPROT_df = pd.read_pickle(join(CURRENT_DIR, ".." ,"data", "data_ESP", "UNIPROT_df.pkl"))


# Use the data in https://github.com/AlexanderKroll/ProSmith to map the molecule IDs
# to their corresponding SMILES string

train_pro=pd.read_csv(join(CURRENT_DIR, ".." ,"data", "data_ProSmith","ESP_train_df.csv"))
test_pro=pd.read_csv(join(CURRENT_DIR, ".." ,"data", "data_ProSmith","ESP_test_df.csv"))
val_pro=pd.read_csv(join(CURRENT_DIR, ".." ,"data", "data_ProSmith","ESP_val_df.csv"))
data_pro = pd.concat([train_pro, test_pro,val_pro], ignore_index=True)
data_pro.reset_index(drop=True, inplace=True)

# Map Uniprot Ids to  their corresponding protein sequence
uniprot_id_to_seq = dict(zip(UNIPROT_df['Uniprot ID'], UNIPROT_df['Sequence']))
mol_id_to_smiles = dict(zip(data_pro['molecule ID'], data_pro['SMILES']))
data_ESP['Sequence'] = data_ESP['Uniprot ID'].map(uniprot_id_to_seq)
data_ESP['SMILES'] = data_ESP['molecule ID'].map(mol_id_to_smiles)
data_ESP.reset_index(drop=True, inplace=True)


###########
# Prepare data for split
###########

# Add Ids column according to DataSAIL documentation for 1D split method:
data_ESP['ids'] = ['ID' + str(index) for index in data_ESP.index]
data_ESP.to_pickle(join(CURRENT_DIR, ".." ,"data", "data_ESP", "dataESP.pkl"))

# Delete ATP
data_ESP=pd.read_pickle(join(CURRENT_DIR, ".." ,"data", "data_ESP", "dataESP.pkl"))
plot_top_keys_values(
    data_ESP,
    key_column="molecule ID",
    value_column=None,
    xlabel='molecule ID',
    ylabel='Count',
    title='Number of data point per molecule ID',
    color="red",
    figsize=(14, 12),
    top_count=100
)
ATP_ids={'CHEBI:30616','C00002'}
# Remove ATP
data_ESP_NOATP = data_ESP[~data_ESP['molecule ID'].isin(ATP_ids)]
print(f"number of ATP Ids{len(data_ESP['molecule ID'].isin(ATP_ids))}")
plot_top_keys_values(
    data_ESP_NOATP,
    key_column="molecule ID",
    value_column=None,
    xlabel='molecule ID',
    ylabel='Count',
    title='Number of data point per molecule ID',
    color="red",
    figsize=(14, 12),
    top_count=100
)
data_ESP_NOATP.reset_index(drop=True, inplace=True)
data_ESP_NOATP.to_pickle(join(CURRENT_DIR, ".." ,"data", "data_ESP", "dataESP_NoATP.pkl"))

# Random delete of 3408 data points
data_ESP=pd.read_pickle(join(CURRENT_DIR, ".." ,"data", "data_ESP", "dataESP.pkl"))
# 18313-14905=3408
num_rows_to_delete = 3408
rows_to_delete = data_ESP.sample(n=num_rows_to_delete).index
data_ESP_D3408=data_ESP.drop(rows_to_delete)
data_ESP_D3408.reset_index(drop=True, inplace=True)
data_ESP_D3408.to_pickle(join(CURRENT_DIR, ".." ,"data", "data_ESP", "dataESP_D3408.pkl"))

plot_top_keys_values(
    data_ESP_D3408,
    key_column="molecule ID",
    value_column=None,
    xlabel='molecule ID',
    ylabel='Count',
    title='Number of data point per molecule ID',
    color="red",
    figsize=(14, 12),
    top_count=100
)
