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

train = pd.read_pickle(join(CURRENT_DIR, "..", "data", "3splits", "test_ESPC2_3S.pkl"))
print(train.shape)
splits= pd.read_pickle(join(CURRENT_DIR, ".." ,"data","3splits", "C2_3splits.pkl"))
splits_dict = dict(zip(splits['ids'], splits['splits']))


data = pd.read_pickle(join(CURRENT_DIR, ".." ,"data","data_ESP", "dataESP.pkl"))
data['splits'] = data.apply(
    lambda row: splits_dict.get((row['molecule ID'], row['Uniprot ID']), None),
    axis=1
)

data_filtered = data[(data['splits'] == "train") | (data['splits'] == "test") | (data['splits'] == "val")]
data_filtered.reset_index(drop=True, inplace=True)
train = data_filtered[data_filtered["splits"] == "train"]
train.reset_index(drop=True, inplace=True)
test = data_filtered[data_filtered["splits"] == "test"]
test.reset_index(drop=True, inplace=True)
val = data_filtered[data_filtered["splits"] == "val"]
val.reset_index(drop=True, inplace=True)

# Create negative data points ###########################
print(f"Start to create negative data points for the train set...")
train = drop_samples_without_mol_file(df=train)
df_metabolites_train, similarity_matrix_train = get_metabolites_and_similarities(df=train)
print(f"Number of metabolites in train set: {len(df_metabolites_train)}")
train["Binding"] = 1
train.reset_index(inplace=True, drop=True)
train = create_negative_samples(df=train, df_metabolites=df_metabolites_train,
                                similarity_matrix=similarity_matrix_train)
train = map_negative_samples2embedding(train)
print(f"Creating negative data points for the train set DONE")

print(f"Start to create negative data points for the test set...")
test = drop_samples_without_mol_file(df=test)
df_metabolites_test, similarity_matrix_test = get_metabolites_and_similarities(df=test)
print(f"Number of metabolites in test set: {len(df_metabolites_test)}")
test["Binding"] = 1
test.reset_index(inplace=True, drop=True)
test = create_negative_samples(df=test, df_metabolites=df_metabolites_test,
                               similarity_matrix=similarity_matrix_test)
test = map_negative_samples2embedding(test)
print(f"Creating negative data points for the test set DONE")

print(f"Start to create negative data points for the val set...")
val = drop_samples_without_mol_file(df=val)
df_metabolites_val, similarity_matrix_val = get_metabolites_and_similarities(df=val)
print(f"Number of metabolites in val set: {len(df_metabolites_val)}")
val["Binding"] = 1
val.reset_index(inplace=True, drop=True)
val = create_negative_samples(df=val, df_metabolites=df_metabolites_val,
                              similarity_matrix=similarity_matrix_val)
val = map_negative_samples2embedding(val)
print(f"Creating negative data points for the val set DONE")

# Reports ###########################
result, total_samples, test_ratio, val_ratio = three_split_report(train, test, val)
print(
    f"Data report after adding negative data and check for NaN or null cells in the data\n{result.to_string()}")
print(f"Total number of samples: {total_samples}")
print(f"Ratio of test set to dataset: {test_ratio}")
print(f"Ratio of val set to dataset: {val_ratio}")

dict_train = collections.Counter(train["Binding"])
dict_test = collections.Counter(test["Binding"])
print(f"The ratio of negative to positive data in train: {round(dict_train[0] / dict_train[1], 2)}")
print(f"The ratio of negative to positive data in test: {round(dict_test[0] / dict_test[1], 2)}")
dict_val = collections.Counter(val["Binding"])
print(f"The ratio of negative to positive data in val: {round(dict_val[0] / dict_val[1], 2)}")


train.drop("splits", axis=1, inplace=True)
test.drop("splits", axis=1, inplace=True)
val.drop("splits", axis=1, inplace=True)
train.to_pickle(join(CURRENT_DIR, "..", "data", "3splits", "train_C2_3S.pkl"))
print(data_report(train))
test.to_pickle(join(CURRENT_DIR, "..", "data", "3splits", "test_C2_3S.pkl"))
print(data_report(test))
val.to_pickle(join(CURRENT_DIR, "..", "data", "3splits", "val_C2_3S.pkl"))
print(data_report(val))
data_ESPC2_3S=data_filtered.copy()
data_ESPC2_3S.drop("splits", axis=1, inplace=True)
data_ESPC2_3S.reset_index(inplace=True, drop=True)
data_ESPC2_3S.to_pickle(join(CURRENT_DIR, "..", "data", "data_ESP", "data_ESPC2_3S.pkl"))


print(len(train)/(len(test)+len(val)+ len(train)))
print(len(test)/(len(test)+len(val)+ len(train)))
print(len(val)/(len(test)+len(val)+ len(train)))


print(
    f"Ratio of  of non-substrate to substrates in train set: {round(len(train.loc[train['Binding'] == 0]) / len(train.loc[train['Binding'] == 1]), 2)}")
print(
    f"Ratio of  of non-substrate to non-substrates in test set: {round(len(test.loc[test['Binding'] == 0]) / len(test.loc[test['Binding'] == 1]), 2)}")
print(
    f"Ratio of  of non-substrate to non-substrates in val set: {round(len(val.loc[val['Binding'] == 0]) / len(val.loc[val['Binding'] == 1]), 2)}")