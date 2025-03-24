import pandas as pd
import inspect
import re
import sys
import time
import matplotlib as mpl
import collections
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import matplotlib.pyplot as plt
from Bio import PDB
import warnings
import logging
import torch
from os.path import join
warnings.filterwarnings("ignore")
import inspect
import shutil
from collections import Counter
from libchebipy import ChebiEntity
import os
from datasail.sail import datasail
import numpy as np
plt.style.use('CCB_plot_style_0v4.mplstyle');
c_styles      = mpl.rcParams['axes.prop_cycle'].by_key()['color']
high_contrast = ['#004488', '#DDAA33', '#BB5566', '#000000']



def split_on_empty_lines(s):
    blank_line_regex = r"(?:\r?\n){2,}"
    return re.split(blank_line_regex, s.strip())


def remove_whitespace_end(s):
    return re.sub(r'[\n\t\s]+$', '', s)


def sub_protein_pair(dataframe, ID_Col, Protein_col, substrate_col):
    brenda = pd.read_pickle(dataframe)
    data = []
    for ind, row in brenda.iterrows():
        EC_IDs = row[ID_Col]
        protein_ids = row[Protein_col]
        substrates = row[substrate_col]
        matching_substrate = [(EC_IDs, protein_id[1], substrate[1]) for protein_id in protein_ids for substrate in
                              substrates if
                              protein_id[0] == substrate[0] and 'more' not in substrate[1]]
        data.extend(matching_substrate)
    sub_Protein = pd.DataFrame(data, columns=['EC_ID', 'Uni_SwissProt', 'Substrate'])
    return sub_Protein


def inh_protein_pair(dataframe, ID_Col, Protein_col, inhibitor_col):
    brenda = pd.read_pickle(dataframe)
    data = []
    for ind, row in brenda.iterrows():
        EC_IDs = row[ID_Col]
        protein_ids = row[Protein_col]
        inhibitors = row[inhibitor_col]
        matching_substrate = [(EC_IDs, protein_id[1], inhibitor[1]) for protein_id in protein_ids for inhibitor in
                              inhibitors if
                              protein_id[0] == inhibitor[0] and 'more' not in inhibitor[1]]
        data.extend(matching_substrate)
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


def read_uniprot_ids(file_path=None, df=None):
    if file_path:
        with open(file_path, 'r') as file:
            return [line.strip() for line in file]
    elif df is not None:
        return df['Uni_SwissProt'].dropna().unique().tolist()
    else:
        raise ValueError("Either file_path or df must be provided")

def write_uniprot_ids(file_path, uniprot_ids):
    with open(file_path, 'w') as file:
        for uniprot_id in uniprot_ids:
            file.write(f"{uniprot_id}\n")

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

def select_best_experimental_pdb(pdb_entries):
    experimental_entries = [entry for entry in pdb_entries if entry['experimental_method'] != 'Computational Model']
    if not experimental_entries:
        return None
    sorted_entries = sorted(
        experimental_entries,
        key=lambda x: (x.get('resolution') if x.get('resolution') is not None else float('inf'),
                       x['experimental_method'] != 'X-ray'))
    return sorted_entries[0]['pdb_id'] if sorted_entries else None

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

class ProteinSelect(PDB.Select):
    def accept_residue(self, residue):
        return PDB.is_aa(residue)


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


def map_negative_samples2embedding(df):
        df['ESM1b_ts'] = df.groupby('Uniprot ID')['ESM1b_ts'].transform(
            lambda x: x.fillna(method='ffill').fillna(method='bfill'))
        df['Sequence'] = df.groupby('Uniprot ID')['Sequence'].transform(
            lambda x: x.fillna(method='ffill').fillna(method='bfill'))
        df['PreGNN'] = df.groupby('molecule ID')['PreGNN'].transform(
            lambda x: x.fillna(method='ffill').fillna(method='bfill'))
        df['ECFP'] = df.groupby('molecule ID')['ECFP'].transform(
            lambda x: x.fillna(method='ffill').fillna(method='bfill'))
        df['substrate ID'] = df.groupby('molecule ID')['substrate ID'].transform(
            lambda x: x.fillna(method='ffill').fillna(method='bfill'))
        df['SMILES'] = df.groupby('molecule ID')['SMILES'].transform(
            lambda x: x.fillna(method='ffill').fillna(method='bfill'))
        return df
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
    nan_check_train_set = train_set.isnull().sum()
    empty_check_train_set = train_set.applymap(lambda x: isinstance(x, str) and x == '').sum()
    nan_check_test_set = test_set.isnull().sum()
    empty_check_test_set = test_set.applymap(lambda x: isinstance(x, str) and x == '').sum()
    result = pd.concat(
        [nan_check_train_set, empty_check_train_set, nan_check_test_set, empty_check_test_set], axis=1)
    result.columns = ['NaNTrain', 'NullTrain', 'NaNTest', 'NullTest']
    test_to_data = round(len(test_set) / (len(test_set) + len(train_set)), 2)
    number_data=len(train_set) + len(test_set)
    return result, number_data, test_to_data


def three_split_report(train_set, test_set, val_set):
    nan_check_train_set = train_set.isnull().sum()
    empty_check_train_set = train_set.applymap(lambda x: isinstance(x, str) and x == '').sum()
    nan_check_test_set = test_set.isnull().sum()
    empty_check_test_set = test_set.applymap(lambda x: isinstance(x, str) and x == '').sum()
    nan_check_val_set = val_set.isnull().sum()
    empty_check_val_set = val_set.applymap(lambda x: isinstance(x, str) and x == '').sum()
    result = pd.concat(
        [nan_check_train_set, empty_check_train_set, nan_check_test_set, empty_check_test_set, nan_check_val_set, empty_check_val_set], axis=1)
    result.columns = ['nanTrain', 'NullTrain', 'NaNTest', 'NullTest', 'nanVal','NullVal']
    test_to_data = round(len(test_set) / (len(test_set) + len(train_set) + len(val_set)), 2)
    val_to_data = round(len(val_set) / (len(test_set) + len(train_set) + len(val_set)), 2)
    number_data=len(train_set) + len(test_set) + len(val_set)
    return result, number_data , test_to_data, val_to_data

def plot_top_keys_values(df, key_column, xlabel, ylabel, title, color='blue', figsize=(12, 10), top_count=30):
    caller_frame = inspect.currentframe().f_back
    df_name = [var_name for var_name, var in caller_frame.f_locals.items() if var is df][0]
    counter = collections.Counter(df[key_column])
    top = counter.most_common(top_count)
    keys, values = zip(*top)
    plt.figure(figsize=figsize)
    plt.bar(keys, values, color=color, alpha=0.8, label='Data Points')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title.format(df_name=df_name))
    plt.xticks(rotation=90, fontsize=12)  # Set fontsize to 12 or any other value you prefer
    plt.subplots_adjust(bottom=0.25)
    plt.legend()
    plt.show()


def plot_top_keys_values_multiple_df(dfs, df_names, key_column, xlabel, ylabel, title, figsize=(15, 10), top_count=30,
                                     title_fontsize=10):
    if not isinstance(dfs, list) or not isinstance(df_names, list):
        raise ValueError("dfs and df_names must be lists.")

    if len(dfs) != len(df_names):
        raise ValueError("The length of dfs and df_names must be the same.")
    combined_counter = collections.Counter()
    for df in dfs:
        combined_counter.update(df[key_column])
    top_combined = combined_counter.most_common(top_count)
    keys, _ = zip(*top_combined)
    keys = list(map(str, keys))
    if len(dfs) >= 2:
        common_keys = set(map(str, dfs[0][key_column])) & set(map(str, dfs[1][key_column]))
    else:
        common_keys = set()
    np.random.seed(0)
    colors_map = {key: np.random.rand(3, ) for key in keys}
    common_color = [1, 0, 0]
    fig, axes = plt.subplots(1, len(dfs), figsize=figsize)
    if len(dfs) == 1:
        axes = [axes]
    for ax, df, df_name in zip(axes, dfs, df_names):
        counter = collections.Counter(df[key_column])
        values = [counter.get(key, 0) for key in keys]
        bar_colors = [common_color if key in common_keys else colors_map[key] for key in keys]
        ax.barh(keys, values, color=bar_colors, alpha=0.6, edgecolor='black')
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_title(title.format(df_name=df_name, top_count=top_count), fontsize=title_fontsize)
        ax.tick_params(axis='y', labelsize=6)
    plt.tight_layout()
    plt.show()

def parse_log(file_path):
    with open(file_path, 'r') as file:
        log_lines = file.readlines()

    iteration_data = []
    unique_losses = set()

    for line in log_lines:
        match_iteration = re.search(r'Iteration (\d+)', line)
        match_loss = re.search(r'Best loss so far: (\d+\.\d+)', line)

        if match_iteration:
            current_iteration = int(match_iteration.group(1))
        elif match_loss:
            current_loss = float(match_loss.group(1))
            if current_loss not in unique_losses:
                iteration_data.append({'iteration': current_iteration, 'loss': current_loss})
                unique_losses.add(current_loss)

    return pd.DataFrame(iteration_data)


def plotting_loss(column, experiment=None, log_directory=None, color_map=None, split_number=None, title=None):
    log_files = [f for f in os.listdir(log_directory) if f.endswith('.log')]
    label_map = {
        "C2": "S2",
        "C1f": "S1$_{p}$",
        "C1e": "S1$_{l}$",
        "I1e": "I1$_{l}$",
        "I1f": "I1$_{p}$",
        "ESPC2": r'ESP$_{S2}$'
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    for log_file in log_files:
        match = re.search(rf'HOP_ESM1bts_and_{column}_(\w+)_{split_number}S.log', log_file)
        split_name = None

        if experiment == "1D":
            if match:
                split_name = match.group(1)
                if "NoEng" in split_name or "D5258" in split_name or "C2" in split_name:
                    exp="Eng"
                    continue
                else:
                    split_name = match.group(1)
            else:
                continue
        elif experiment == "2D":
            if match:
                split_name = match.group(1)
                if "C2" in split_name or "ESPC2" in split_name:
                    split_name = match.group(1)
                else:
                    continue
            else:
                continue
        elif experiment == "NoEng":
            if match:
                split_name = match.group(1)
                if "NoEng" in split_name or "D5258" in split_name:
                    split_name = match.group(1)
                else:
                    continue
            else:
                continue

        # Apply label mapping for display name and color
        display_name = label_map.get(split_name, split_name)
        color = color_map.get(display_name, 'gray')

        # Set line style based on the column name
        if column == "PreGNN":
            line_style = '--'  # Dashed line for PreGNN
        elif column == "ECFP":
            line_style = '-'   # Solid line for ECFP
        else:
            line_style = '-'   # Default to solid line

        log_data = parse_log(os.path.join(log_directory, log_file))
        window_size = min(20, max(1, len(log_data['iteration']) // 10))
        smooth_loss = np.convolve(log_data['loss'], np.ones(window_size) / window_size, mode='same')
        smooth_iteration = log_data['iteration']
        smooth_iteration = pd.Series(smooth_iteration)
        smooth_loss = pd.Series(smooth_loss)
        last_iteration = smooth_iteration.iloc[-1]
        remaining_iterations = np.arange(last_iteration + 1, 2001)
        extension_loss = np.full_like(remaining_iterations, smooth_loss.iloc[-1])
        smooth_iteration = pd.concat([smooth_iteration, pd.Series(remaining_iterations)], ignore_index=True)
        smooth_loss = pd.concat([smooth_loss, pd.Series(extension_loss)], ignore_index=True)
        ax.plot(smooth_iteration, smooth_loss, color=color, label=display_name, alpha=0.8, linestyle=line_style)
        min_loss_index = log_data['loss'].idxmin()
        min_loss_iteration = log_data['iteration'].iloc[min_loss_index]
        min_loss = int(log_data['loss'].iloc[min_loss_index])
        ax.scatter(min_loss_iteration, min_loss, color=color, s=50, marker='o')
        ax.annotate(f'{round(min_loss)}', (min_loss_iteration, min_loss), textcoords="offset points", xytext=(5, 5),
            ha='center', fontsize=10)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)

    # Use the title argument or a default if not provided
    ax.set_title(title, fontsize=14, fontname='Arial', pad=30)

    ax.set_xlim(0, 2000)
    ax.legend(loc='upper right', fontsize=12, bbox_to_anchor=(1.2, 1))
    #plt.savefig(f'/Users/vahidatabaigi/Desktop/Thesis/thesis-template/Figures/{column}-{experiment}-{split_number}.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'/Users/vahidatabaigi/Desktop/{column}-{experiment}-{split_number}.png', dpi=600, bbox_inches='tight')
    plt.tight_layout()
    plt.show()





###########################################################################################################

def datasail_wrapper(split_method, DataFrame, split_size):
    names = ["train", "test"]
    if len(split_size) == 3:
        names.append("val")

    if split_method in ["C1e", "I1e"]:
        e_splits, f_splits, inter_sp = datasail(
            techniques=[split_method],
            splits=split_size,
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
            splits=split_size,
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
            splits=split_size,
            names=names,
            solver="SCIP",
            inter=[(x[0], x[0]) for x in DataFrame[["ids"]].values.tolist()],
            e_type="M",
            e_sim="ecfp",
            e_data=dict(DataFrame[["ids", "SMILES"]].values.tolist()),
            f_type="P",
            f_sim="cdhit",
            f_data=dict(DataFrame[["ids", "Sequence"]].values.tolist()),
            epsilon=0.0,
        )
    else:
        raise ValueError("Invalid split method provided. Use one of ['C2','C1e', 'C1f', 'I1e', 'I1f']")
    return e_splits, f_splits, inter_sp


def datasail_wrapper_v2(split_method, DataFrame, split_size, stratification=False, epsilon=0, delta=0):
    names = ["train", "test"]
    if len(split_size) == 3:
        names.append("val")

    if split_method in ["C1e", "I1e"]:
        e_splits, f_splits, inter_sp = datasail(
            techniques=[split_method],
            splits=split_size,
            names=names,
            solver="GUROBI",
            e_type="M",
            e_data=dict(DataFrame[["ids", "SMILES"]].values.tolist()),
            e_strat=dict(DataFrame[["ids", "Binding"]].values.tolist()) if stratification else None,
            e_sim="ecfp",
            epsilon=epsilon,
            delta=delta,
            max_sec = 100000
        )
    elif split_method in ["C1f", "I1f"]:
        e_splits, f_splits, inter_sp = datasail(
            techniques=[split_method],
            splits=split_size,
            names=names,
            solver="GUROBI",
            f_type="P",
            f_data=dict(DataFrame[["ids", "Sequence"]].values.tolist()),
            f_strat=dict(DataFrame[["ids", "Binding"]].values.tolist()) if stratification else None,
            f_sim="cdhit",
            epsilon=epsilon,
            delta=delta,
            max_sec=100000
        )
    elif split_method in ["C2","I2"]:
        e_splits, f_splits, inter_sp = datasail(
            techniques=[split_method],
            splits=split_size,
            names=names,
            solver="GUROBI",
            inter=[(x[0], x[0]) for x in DataFrame[["ids"]].values.tolist()],
            e_type="M",
            e_sim="ecfp",
            e_data=dict(DataFrame[["ids", "SMILES"]].values.tolist()),
            e_strat=dict(DataFrame[["ids", "Binding"]].values.tolist()) if stratification else None,
            f_type="P",
            f_sim="cdhit",
            f_data=dict(DataFrame[["ids", "Sequence"]].values.tolist()),
            f_strat=dict(DataFrame[["ids", "Binding"]].values.tolist()) if stratification else None,
            epsilon=epsilon,
            delta=delta,
            max_sec=100000
        )
    else:
        raise ValueError("Invalid split method provided. Use one of ['C2','C1e', 'C1f', 'I1e', 'I1f','I2']")
    return e_splits, f_splits, inter_sp

##########################################################################
def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
#######################################################################


def get_protein_sequences_with_retry(uniprot_ids, retries=3, backoff_factor=0.5):
    base_url = 'https://www.uniprot.org/uniprot/'
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)

    sequences = {}
    found_count = 0
    not_found_count = 0

    for uniprot_id in uniprot_ids:
        response = session.get(f'{base_url}{uniprot_id}.fasta')
        if response.status_code == 200:
            lines = response.text.split('\n')
            sequence = ''.join(lines[1:])
            sequences[uniprot_id] = sequence
            found_count += 1
            print(f"Sequence found for UniProt ID {uniprot_id}. Total found: {found_count}")
        else:
            sequences[uniprot_id] = None
            not_found_count += 1
            print(f"No sequence found for UniProt ID {uniprot_id}. Total not found: {not_found_count}")

    return sequences




