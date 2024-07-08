import pandas as pd
import numpy as np
import random
import os
import re
import sys
import time
import collections
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from Bio import SeqIO
import warnings
import argparse
import os.path
from colorama import init, Fore, Style
sys.path.append("./additional_code")
from additional_code.helper_functions import *
from additional_code.negative_data_generator import *
warnings.filterwarnings("ignore")


def main(args):
    CURRENT_DIR = os.getcwd()
    split_method = args.split_method
    split_size = args.split_size
    Data_suffix = args.Data_suffix

    if len(split_size) not in [2, 3]:
        raise ValueError("The split-number argument must be a list of either two or three integers.")

    log_file = os.path.join(CURRENT_DIR, "..", "data", "Reports", f'Report_{len(split_size)}Splits_{split_method}{Data_suffix}.log')
    if os.path.exists(log_file):
        os.remove(log_file)
    setup_logging(log_file)
    logging.info(f"Current Directory: {CURRENT_DIR}")

    data_file = os.path.join(CURRENT_DIR, "..", "data", "data_ESP", f"dataESP{Data_suffix}.pkl")
    train_output_file = os.path.join(CURRENT_DIR, "..", "data", f"{len(split_size)}splits", f"train_{split_method}{Data_suffix}_{len(split_size)}S.pkl")
    test_output_file = os.path.join(CURRENT_DIR, "..", "data", f"{len(split_size)}splits", f"test_{split_method}{Data_suffix}_{len(split_size)}S.pkl")
    val_output_file = os.path.join(CURRENT_DIR, "..", "data", f"{len(split_size)}splits",f"val_{split_method}{Data_suffix}_{len(split_size)}S.pkl")

    if split_method in ["C1e", "I1e"]:
        logging.info("*** Start running the dataSAIL***'\n'For more information about dataSAIL please check the dataSAIL documentation: https://datasail.readthedocs.io/en/latest/index.html")
        data = pd.read_pickle(data_file)
        e_splits, _, _ = datasail_wrapper(split_method, data, split_size)
        for key in e_splits.keys():
            data['split_mol'] = data['ids'].map(e_splits[key][0])
        data_filtered = data[(data['split_mol'] == "train") | (data['split_mol'] == "test")|(data['split_mol'] == "val")]
        data_filtered.reset_index(drop=True, inplace=True)
        train = data_filtered[data_filtered["split_mol"] == "train"]
        train.reset_index(drop=True, inplace=True)
        test = data_filtered[data_filtered["split_mol"] == "test"]
        test.reset_index(drop=True, inplace=True)
        if len(split_size) ==3:
            val = data_filtered[data_filtered["split_mol"] == "val"]
            val.reset_index(drop=True, inplace=True)
        logging.info("dataSAIL splits the data successfully")

    elif split_method in ["C1f", "I1f"]:
        logging.info("Start running dataSAIL")
        data_ESP = pd.read_pickle(data_file)
        _, f_splits, _ = datasail_wrapper(split_method, data_ESP, split_size)
        for key in f_splits.keys():
            data_ESP['split_seq'] = data_ESP['ids'].map(f_splits[key][0])
        data_ESP_filtered = data_ESP[(data_ESP['split_seq'] == "train") | (data_ESP['split_seq'] == "test")|(data_ESP['split_seq'] == "val")]
        data_ESP_filtered.reset_index(drop=True, inplace=True)
        train = data_ESP_filtered[data_ESP_filtered["split_seq"] == "train"]
        train.reset_index(drop=True, inplace=True)
        test = data_ESP_filtered[data_ESP_filtered["split_seq"] == "test"]
        test.reset_index(drop=True, inplace=True)
        if len(split_size) ==3:
            val = data_ESP_filtered[data_ESP_filtered["split_seq"] == "val"]
            val.reset_index(drop=True, inplace=True)
        logging.info("dataSAIL split the data successfully")

    elif split_method in ["C2"]:
        logging.info("Start running dataSAIL")
        data= pd.read_pickle(data_file)
        _, _, inter_sp = datasail_wrapper(split_method, data, split_size)
        for key in inter_sp.keys():
            inter_dict = {k[0]: v for k, v in inter_sp[key][0].items()}
            data['split_inter'] = data['ids'].map(inter_dict)
        data_filtered = data[(data['split_inter'] == "train") | (data['split_inter'] == "test")|(data['split_inter'] == "val")]
        data_filtered.reset_index(drop=True, inplace=True)
        train = data_filtered[data_filtered["split_inter"] == "train"]
        train.reset_index(drop=True, inplace=True)
        test = data_filtered[data_filtered["split_inter"] == "test"]
        test.reset_index(drop=True, inplace=True)
        if len(split_size) ==3:
            val = data_filtered[data_filtered["split_inter"] == "val"]
            val.reset_index(drop=True, inplace=True)
        logging.info("dataSAIL split the data successfully")

    else:
        raise ValueError("Invalid split method provided. Use one of ['C2','C1e', 'C1f', 'I1e', 'I1f']")
    if len(split_size) == 2:
        result, total_samples, test_ratio = two_split_report(train, test)
        logging.info(f"Data report after splitting data by {split_method} split method and check for nan or null cells '\n'{result.to_string()}")
        logging.info(f"Total number of samples: {total_samples}")
        logging.info(f"Ratio of test set to dataset: {test_ratio}")
    elif len(split_size) == 3:
        result, total_samples, test_ratio, val_ratio = three_split_report(train, test, val)
        logging.info(f"Data report after splitting data by {split_method} split method and check for nan or null cells '\n'{result.to_string()}")
        logging.info(f"Total number of samples: {total_samples}")
        logging.info(f"Ratio of test set to dataset: {test_ratio}")
        logging.info(f"Ratio of val set to dataset: {val_ratio}")

    logging.info("Start to create negative data points for the training set...")
    train = drop_samples_without_mol_file(df=train)
    df_metabolites_train, similarity_matrix_train = get_metabolites_and_similarities(df=train)
    logging.info(f"Number of metabolites in training set: {len(df_metabolites_train)}")
    train["Binding"] = 1
    train.reset_index(inplace=True, drop=True)
    train = create_negative_samples(df=train, df_metabolites=df_metabolites_train, similarity_matrix=similarity_matrix_train)
    train['ESM1b_ts'] = train.groupby('Uniprot ID')['ESM1b_ts'].transform(
        lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    train['Sequence'] = train.groupby('Uniprot ID')['Sequence'].transform(
        lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    train['PreGNN'] = train.groupby('molecule ID')['PreGNN'].transform(
        lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    train['ECFP'] = train.groupby('molecule ID')['ECFP'].transform(
        lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    train['substrate ID'] = train.groupby('molecule ID')['substrate ID'].transform(
        lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    train['SMILES'] = train.groupby('molecule ID')['SMILES'].transform(
        lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    logging.info("Creating negative data points for the train set DONE")

    logging.info("Start to create negative data points for the test set...")
    test = drop_samples_without_mol_file(df=test)
    df_metabolites_test, similarity_matrix_test = get_metabolites_and_similarities(df=test)
    logging.info(f"Number of metabolites in test set: {len(df_metabolites_test)}")
    test["Binding"] = 1
    test.reset_index(inplace=True, drop=True)
    test = create_negative_samples(df=test, df_metabolites=df_metabolites_test,similarity_matrix=similarity_matrix_test)
    test['ESM1b_ts'] = test.groupby('Uniprot ID')['ESM1b_ts'].transform(
        lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    test['Sequence'] = test.groupby('Uniprot ID')['Sequence'].transform(
        lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    test['PreGNN'] = test.groupby('molecule ID')['PreGNN'].transform(
        lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    test['ECFP'] = test.groupby('molecule ID')['ECFP'].transform(
        lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    test['substrate ID'] = test.groupby('molecule ID')['substrate ID'].transform(
        lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    test['SMILES'] = test.groupby('molecule ID')['SMILES'].transform(
        lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    logging.info("Creating negative data points for the test set DONE")
    if len(split_size) == 3:
        logging.info("Start to create negative data points for the val set...")
        val = drop_samples_without_mol_file(df=val)
        df_metabolites_val, similarity_matrix_val = get_metabolites_and_similarities(df=val)
        logging.info(f"Number of metabolites in val set: {len(df_metabolites_val)}")
        val["Binding"] = 1
        val.reset_index(inplace=True, drop=True)
        val = create_negative_samples(df=val, df_metabolites=df_metabolites_val, similarity_matrix=similarity_matrix_val)
        val['ESM1b_ts'] = val.groupby('Uniprot ID')['ESM1b_ts'].transform(
            lambda x: x.fillna(method='ffill').fillna(method='bfill'))
        val['Sequence'] = val.groupby('Uniprot ID')['Sequence'].transform(
            lambda x: x.fillna(method='ffill').fillna(method='bfill'))
        val['PreGNN'] = val.groupby('molecule ID')['PreGNN'].transform(
            lambda x: x.fillna(method='ffill').fillna(method='bfill'))
        val['ECFP'] = val.groupby('molecule ID')['ECFP'].transform(
            lambda x: x.fillna(method='ffill').fillna(method='bfill'))
        val['substrate ID'] = val.groupby('molecule ID')['substrate ID'].transform(
            lambda x: x.fillna(method='ffill').fillna(method='bfill'))
        val['SMILES'] = val.groupby('molecule ID')['SMILES'].transform(
            lambda x: x.fillna(method='ffill').fillna(method='bfill'))
        logging.info("Creating negative data points for the val set DONE")

    if len(split_size) == 2:
        result, total_samples, test_ratio = two_split_report(train, test)
        logging.info(f"Data report after adding negative data and check for nan or null data cells '\n'{result.to_string()}")
        logging.info(f"Total number of samples: {total_samples}")
        logging.info(f"Ratio of test set to total dataset: {test_ratio}")
    elif len(split_size) == 3:
        result, total_samples, test_ratio, val_ratio = three_split_report(train, test, val)
        logging.info(
            f"Data report after adding negative data and check for nan or null data cells '\n'{result.to_string()}")
        logging.info(f"Total number of samples: {total_samples}")
        logging.info(f"Ratio of test set to dataset: {test_ratio}")
        logging.info(f"Ratio of val set to dataset: {val_ratio}")

    dict_train = collections.Counter(train["Binding"])
    dict_test = collections.Counter(test["Binding"])
    logging.info(f"the ratio of negative to positive data in train: {round(dict_train[0] / dict_train[1], 2)}")
    logging.info(f"the ratio of negative to positive data in test: {round(dict_test[0] / dict_test[1], 2)}")
    if len(split_size) == 3:
        dict_val = collections.Counter(val["Binding"])
        logging.info(f"the ratio of negative to positive data in val: {round(dict_val[0] / dict_val[1], 2)}")

    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    train.to_pickle(train_output_file)
    test.to_pickle(test_output_file)
    if len(split_size) == 3:
        val.to_pickle(val_output_file)

    init()
    logging.info(
        Fore.GREEN + f"***** PROCESS COMPLETED: For an overview, "
                     f"please review the Report_{len(split_size)}Splits_{split_method}{Data_suffix}.log file in Reports folder. *****" + Style.RESET_ALL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the data preprocessing and negative sample generation script.")
    parser.add_argument('--split-method', type=str, required=True,
                        help="The split method should be one of [C2,C1e, C1f, I1e, I1f]")
    parser.add_argument('--split-size', type=int, nargs='+', required=True,
                        help="List of integers for splitting, e.g., 8 2 or 7 2 1")
    parser.add_argument('--Data-suffix',default="", type=str, required=True,
                        help="The Dataframe suffix name should be one of [ _NoATP ,  _D3408 , ''] ")
    args = parser.parse_args()
    main(args)
