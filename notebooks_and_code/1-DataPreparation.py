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
sys.path.append("./additional_code")
from additional_code.helper_functions import *
from additional_code.negative_data_generator import *
warnings.filterwarnings("ignore")
CURRENT_DIR = os.getcwd()
print(CURRENT_DIR)

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
# Drop for empty string
train_set = train_set[~(train_set['GNN rep (pretrained)'].str.strip() == '')]
test_set = test_set[~(test_set['GNN rep (pretrained)'].str.strip() == '')]
train_set.reset_index(drop=True, inplace=True)
test_set.reset_index(drop=True, inplace=True)
dataESP = pd.concat([train_set, test_set], ignore_index=True)
dataESP['substrate ID'] = dataESP['substrate ID'].str.replace('CHEBI:', '')
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
dataESP['Sequence'] = dataESP['Uniprot ID'].map(uniprot_id_to_seq)
dataESP['SMILES'] = dataESP['molecule ID'].map(mol_id_to_smiles)
dataESP.rename(columns={'GNN rep (pretrained)': 'PreGNN'},inplace=True)
dataESP.reset_index(drop=True, inplace=True)

###########
# Prepare data for split
###########
# Add Ids column according to DataSAIL documentation for split
dataESP['ids'] = ['ID' + str(index) for index in dataESP.index]
dataESP.to_pickle(join(CURRENT_DIR, ".." ,"data", "data_ESP", "dataESP.pkl"))
print(data_report(dataESP))

