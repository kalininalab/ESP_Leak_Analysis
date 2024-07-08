import pandas as pd
import inspect
import re
import sys
import time
import collections
import requests
import matplotlib.pyplot as plt
from Bio import PDB
import warnings
import logging
import torch
from os.path import join
warnings.filterwarnings("ignore")
import inspect
import shutil
from libchebipy import ChebiEntity
import os
from datasail.sail import datasail
import numpy as np


def split_on_empty_lines(s):
    # greedily match 2 or more new-lines
    blank_line_regex = r"(?:\r?\n){2,}"
    return re.split(blank_line_regex, s.strip())


def remove_whitespace_end(s):
    # Remove occurrences of \n, \t, or space, or a combination of them from the end of the string
    return re.sub(r'[\n\t\s]+$', '', s)


def sub_protein_pair(dataframe, ID_Col, Protein_col, substrate_col):
    brenda = pd.read_pickle(dataframe)
    # Create an empty list to store tuples
    data = []
    # Iterate over each row
    for ind, row in brenda.iterrows():
        # Extract protein IDs and substrates
        EC_IDs = row[ID_Col]
        protein_ids = row[Protein_col]
        substrates = row[substrate_col]
        # Filter substrates that match the first element of protein IDs and do not contain 'more'
        matching_substrate = [(EC_IDs, protein_id[1], substrate[1]) for protein_id in protein_ids for substrate in
                              substrates if
                              protein_id[0] == substrate[0] and 'more' not in substrate[1]]
        # Extend data list with matching substrates
        data.extend(matching_substrate)
    # Create DataFrame from the list of tuples
    sub_Protein = pd.DataFrame(data, columns=['EC_ID', 'Uni_SwissProt', 'Substrate'])
    return sub_Protein


def inh_protein_pair(dataframe, ID_Col, Protein_col, inhibitor_col):
    brenda = pd.read_pickle(dataframe)
    # Create an empty list to store tuples
    data = []
    # Iterate over each row
    for ind, row in brenda.iterrows():
        # Extract protein IDs and substrates
        EC_IDs = row[ID_Col]
        protein_ids = row[Protein_col]
        inhibitors = row[inhibitor_col]
        # Filter substrates that match the first element of protein IDs and do not contain 'more'
        matching_substrate = [(EC_IDs, protein_id[1], inhibitor[1]) for protein_id in protein_ids for inhibitor in
                              inhibitors if
                              protein_id[0] == inhibitor[0] and 'more' not in inhibitor[1]]
        # Extend data list with matching substrates
        data.extend(matching_substrate)
    # Create DataFrame from the list of tuples
    sub_Protein = pd.DataFrame(data, columns=['EC_ID', 'Uni_SwissProt', 'Inhibitors'])
    return sub_Protein


def chebi2smiles(chebi_ids):
    chebi_to_smiles = {}
    for chebi_id in chebi_ids:
        try:
            entity = ChebiEntity(chebi_id)
            smiles = entity.get_smiles()
            chebi_to_smiles[chebi_id] = smiles
        except Exception as e:
            print(f"Error retrieving SMILES for {chebi_id}: {e}")
    return chebi_to_smiles


def create_empty_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


###############################################################################


# Function to read UniProt IDs from a file or DataFrame
def read_uniprot_ids(file_path=None, df=None):
    if file_path:
        with open(file_path, 'r') as file:
            return [line.strip() for line in file]
    elif df is not None:
        return df['Uni_SwissProt'].dropna().unique().tolist()
    else:
        raise ValueError("Either file_path or df must be provided")


# Function to write UniProt IDs to a file
def write_uniprot_ids(file_path, uniprot_ids):
    with open(file_path, 'w') as file:
        for uniprot_id in uniprot_ids:
            file.write(f"{uniprot_id}\n")


# Function to get PDB entries for a given UniProt ID
def get_pdb_entries(uniprot_id, retries=3, backoff_factor=1.0):
    url = f'https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{uniprot_id}'
    for attempt in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                return data.get(uniprot_id, [])
            else:
                return []
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(backoff_factor * (2 ** attempt))
    return []


# Function to select the best PDB entry
def select_best_experimental_pdb(pdb_entries):
    experimental_entries = [entry for entry in pdb_entries if entry['experimental_method'] != 'Computational Model']
    if not experimental_entries:
        return None
    sorted_entries = sorted(
        experimental_entries,
        key=lambda x: (x.get('resolution') if x.get('resolution') is not None else float('inf'),
                       x['experimental_method'] != 'X-ray'))
    return sorted_entries[0]['pdb_id'] if sorted_entries else None


# Function to download a PDB file
def download_pdb(pdb_id, output_dir, uniprot_id):
    pdb_url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    for attempt in range(3):
        try:
            response = requests.get(pdb_url)
            if response.status_code == 200:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                pdb_file_path = os.path.join(output_dir, f'{uniprot_id}.pdb')
                with open(pdb_file_path, 'wb') as file:
                    file.write(response.content)
                return pdb_file_path
        except requests.exceptions.RequestException as e:
            print(f"Download attempt {attempt + 1} failed: {e}")
            time.sleep(1 * (2 ** attempt))
    return None


# Class to remove ligands from PDB files
class ProteinSelect(PDB.Select):
    def accept_residue(self, residue):
        return PDB.is_aa(residue)


# Function to remove ligands from a PDB file
def remove_ligands(pdb_file_path):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('', pdb_file_path)
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(pdb_file_path, select=ProteinSelect())
################################################################################


def map_embedded_pro_to_uniprot(dataframe,path):
    embeddings_dict = {uniprot_id: None for uniprot_id in dataframe["Uni_SwissProt"].unique()}
    for i in range(16):
        file_path = join(path, f'Protein_embeddings_V{i}.pt')
        if os.path.exists(file_path):
            rep_dict = torch.load(file_path)
            print(f"Loaded embeddings from {file_path}")
            for uniprot_id in dataframe["Uni_SwissProt"].unique():
                if uniprot_id in rep_dict:
                    if embeddings_dict[uniprot_id] is None:
                        embeddings_dict[uniprot_id] = rep_dict[uniprot_id].tolist()
                    else:
                        embeddings_dict[uniprot_id].extend(rep_dict[uniprot_id].tolist())
                else:
                    print(f"Embedding for {uniprot_id} not found in {file_path}")
        else:
            print(f"File {file_path} does not exist")

    return embeddings_dict


def map_embedded_smiles_to_mol_id(dataframe,path):
    embeddings_dict = {mol_id: None for mol_id in dataframe["molecule_ID"].unique()}
    for i in range(16):
        file_path = join(path, f'SMILES_repr_{i}.pkl')
        if os.path.exists(file_path):
            rep_dict = pd.read_pickle(file_path)
            print(f"Loaded embeddings from {file_path}")
            for mol_id in dataframe["molecule_ID"].unique():
                if mol_id in rep_dict:
                    if embeddings_dict[mol_id] is None:
                        embeddings_dict[mol_id] = rep_dict[mol_id][0]
                    else:
                        embeddings_dict[mol_id].extend(rep_dict[mol_id][0])
                else:
                    print(f"Embedding for {mol_id} not found in {file_path}")
        else:
            print(f"File {file_path} does not exist")
    return embeddings_dict

##################################################################################


def data_report(df, display_limit=None):
    nan_check = df.isnull().sum()
    nan_check.name = 'NaN'
    empty_check = pd.Series(0, index=df.columns, name='Empty')
    for col in df.select_dtypes(include=['object']).columns:
        empty_check[col] = df[col].apply(lambda x: isinstance(x, str) and x == '').sum()
    empty_list = df.apply(
        lambda col: col.apply(lambda x: isinstance(x, list) and len(x) == 0 if x is not None else False)).sum()
    empty_list = pd.Series(empty_list, name='empty_list')
    def count_unique(x):
        if isinstance(x.iloc[0], (list, np.ndarray)):
            return len(x)
        else:
            return len(pd.Series(x).dropna().unique()) if x is not None else 0
    unique_count = df.apply(count_unique)
    unique_count = pd.Series(unique_count, name='Unique')
    result = pd.concat([nan_check, empty_check, empty_list, unique_count], axis=1)
    if display_limit:
        result = result.iloc[:, :display_limit]
    caller_frame = inspect.currentframe().f_back
    df_name = [var_name for var_name, var in caller_frame.f_locals.items() if var is df][0]
    print(f"Dimension for {df_name}: {str(df.shape)}")
    return result


def two_split_report(train_set, test_set):
    # Calculate NaN counts and empty string counts for train and test sets
    nan_check_train_set = train_set.isnull().sum()
    empty_check_train_set = train_set.applymap(lambda x: isinstance(x, str) and x == '').sum()

    nan_check_test_set = test_set.isnull().sum()
    empty_check_test_set = test_set.applymap(lambda x: isinstance(x, str) and x == '').sum()

    # Concatenate results side by side
    result = pd.concat(
        [nan_check_train_set, empty_check_train_set, nan_check_test_set, empty_check_test_set], axis=1)
    result.columns = ['NaNTrain', 'NullTrain', 'NaNTest', 'NullTest']
    test_to_data = round(len(test_set) / (len(test_set) + len(train_set)), 2)
    number_data=len(train_set) + len(test_set)

    return result, number_data, test_to_data


def three_split_report(train_set, test_set, val_set):
    # Calculate NaN counts and empty string counts for train and test sets
    nan_check_train_set = train_set.isnull().sum()
    empty_check_train_set = train_set.applymap(lambda x: isinstance(x, str) and x == '').sum()

    nan_check_test_set = test_set.isnull().sum()
    empty_check_test_set = test_set.applymap(lambda x: isinstance(x, str) and x == '').sum()

    nan_check_val_set = val_set.isnull().sum()
    empty_check_val_set = val_set.applymap(lambda x: isinstance(x, str) and x == '').sum()

    # Concatenate results side by side
    result = pd.concat(
        [nan_check_train_set, empty_check_train_set, nan_check_test_set, empty_check_test_set, nan_check_val_set, empty_check_val_set], axis=1)
    result.columns = ['nanTrain', 'NullTrain', 'NaNTest', 'NullTest', 'nanVal','NullVal']
    test_to_data = round(len(test_set) / (len(test_set) + len(train_set) + len(val_set)), 2)
    val_to_data = round(len(val_set) / (len(test_set) + len(train_set) + len(val_set)), 2)
    number_data=len(train_set) + len(test_set) + len(val_set)

    return result, number_data , test_to_data, val_to_data


def plot_top_keys_values(df, key_column, xlabel, ylabel, title, color='blue', figsize=(12, 10), top_count=30):
    counter = collections.Counter(df[key_column])
    top = counter.most_common(top_count)
    keys, values = zip(*top)
    plt.figure(figsize=figsize)
    plt.bar(keys, values, color=color, alpha=0.8, label='Data Points')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=90, fontsize='xx-small')
    plt.subplots_adjust(bottom=0.25)
    plt.legend()
    plt.show()


###########################################################################################################

def datasail_wrapper(split_method, DataFrame, split_number):
    names = ["train", "test"]
    if len(split_number) == 3:
        names.append("val")

    if split_method in ["C1e", "I1e"]:
        e_splits, f_splits, inter_sp = datasail(
            techniques=[split_method],
            splits=split_number,
            names=names,
            solver="SCIP",
            e_type="M",
            e_data=dict(DataFrame[["ids", "SMILES"]].values.tolist()),
            e_sim="ecfp",
            epsilon=0
        )
    elif split_method in ["C1f", "I1f"]:
        e_splits, f_splits, inter_sp = datasail(
            techniques=[split_method],
            splits=split_number,
            names=names,
            solver="SCIP",
            f_type="P",
            f_data=dict(DataFrame[["ids", "Sequence"]].values.tolist()),
            f_sim="cdhit",
            epsilon=0
        )
    elif split_method in ["C2"]:
        e_splits, f_splits, inter_sp = datasail(
            techniques=[split_method],
            splits=split_number,
            names=names,
            solver="SCIP",
            inter=[(x[0], x[0]) for x in DataFrame[["ids"]].values.tolist()],
            e_type="M",
            e_data=dict(DataFrame[["ids", "SMILES"]].values.tolist()),
            f_type="P",
            f_data=dict(DataFrame[["ids", "Sequence"]].values.tolist())
        )
    else:
        raise ValueError("Invalid split method provided. Use one of ['C2','C1e', 'C1f', 'I1e', 'I1f']")
    return e_splits, f_splits, inter_sp


def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

