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

"""
The related code to generate negative data points was sourced from the ESP repository. 
"""


def main(args):
    CURRENT_DIR = os.getcwd()
    split_method = args.split_method
    split_size = args.split_size
    input_path = args.input_path
    df_name = input_path.split("/")[-1]
    Data_suffix = ""
    if "_" in df_name:
        Data_suffix = "_" + df_name.split(".")[0].split("_")[-1]

    if len(split_size) not in [2, 3]:
        raise ValueError("The split-size argument must be a list of either two or three integers.")

    log_file = os.path.join(CURRENT_DIR, "..", "data", "Reports", "split_report",
                            f"Report_{split_method}{Data_suffix}_{len(split_size)}S.log")
    if os.path.exists(log_file):
        os.remove(log_file)
    setup_logging(log_file)
    logging.info(f"Current Directory: {CURRENT_DIR}")

    train_output_file = os.path.join(CURRENT_DIR, "..", "data", f"{len(split_size)}splits",
                                     f"train_{split_method}{Data_suffix}_{len(split_size)}S.pkl")
    test_output_file = os.path.join(CURRENT_DIR, "..", "data", f"{len(split_size)}splits",
                                    f"test_{split_method}{Data_suffix}_{len(split_size)}S.pkl")
    val_output_file = os.path.join(CURRENT_DIR, "..", "data", f"{len(split_size)}splits",
                                   f"val_{split_method}{Data_suffix}_{len(split_size)}S.pkl")

    data = pd.read_pickle(input_path)
    logging.info(
        "*** Start running the DataSAIL***\nFor more information about DataSAIL please check it's webpage: "
        "https://datasail.readthedocs.io/en/latest/index.html")
    e_splits, f_splits, inter_sp = datasail_wrapper(split_method, data, split_size)
    if split_method in ["C1e", "I1e"]:
        for key in e_splits.keys():
            data['split'] = data['ids'].map(e_splits[key][0])
    elif split_method in ["C1f", "I1f"]:
        for key in f_splits.keys():
            data['split'] = data['ids'].map(f_splits[key][0])
    elif split_method in ["C2"]:
        for key in inter_sp.keys():
            inter_dict = {k[0]: v for k, v in inter_sp[key][0].items()}
            data['split'] = data['ids'].map(inter_dict)
    data_filtered = data[(data['split'] == "train") | (data['split'] == "test") | (data['split'] == "val")]
    data_filtered.reset_index(drop=True, inplace=True)
    train = data_filtered[data_filtered["split"] == "train"]
    train.reset_index(drop=True, inplace=True)
    test = data_filtered[data_filtered["split"] == "test"]
    test.reset_index(drop=True, inplace=True)
    val = None
    if len(split_size) == 3:
        val = data_filtered[data_filtered["split"] == "val"]
        val.reset_index(drop=True, inplace=True)
        logging.info("DataSAIL split the data successfully")

    if len(split_size) == 2:
        result, total_samples, test_ratio = two_split_report(train, test)
        logging.info(
            f"Data report after splitting data by {split_method} split method and check for NaN or null cells in the "
            f"data\n{result.to_string()}")
        logging.info(f"Total number of samples: {total_samples}")
        logging.info(f"Ratio of test set to dataset: {test_ratio}")
    elif len(split_size) == 3:
        result, total_samples, test_ratio, val_ratio = three_split_report(train, test, val)
        logging.info(
            f"Data report after splitting data by {split_method} split method and check for NaN or null cells in the "
            f"data\n{result.to_string()}")
        logging.info(f"Total number of samples: {total_samples}")
        logging.info(f"Ratio of test set to dataset: {test_ratio}")
        logging.info(f"Ratio of val set to dataset: {val_ratio}")

    logging.info(f"Start to create negative data points for the train set...")
    train = drop_samples_without_mol_file(df=train)
    df_metabolites_train, similarity_matrix_train = get_metabolites_and_similarities(df=train)
    logging.info(f"Number of metabolites in train set: {len(df_metabolites_train)}")
    train["Binding"] = 1
    train.reset_index(inplace=True, drop=True)
    train = create_negative_samples(df=train, df_metabolites=df_metabolites_train,
                                    similarity_matrix=similarity_matrix_train)
    train = map_negative_samples2embedding(train)
    logging.info(f"Creating negative data points for the train set DONE")

    logging.info(f"Start to create negative data points for the test set...")
    test = drop_samples_without_mol_file(df=test)
    df_metabolites_test, similarity_matrix_test = get_metabolites_and_similarities(df=test)
    logging.info(f"Number of metabolites in test set: {len(df_metabolites_test)}")
    test["Binding"] = 1
    test.reset_index(inplace=True, drop=True)
    test = create_negative_samples(df=test, df_metabolites=df_metabolites_test,
                                   similarity_matrix=similarity_matrix_test)
    test = map_negative_samples2embedding(test)
    logging.info(f"Creating negative data points for the test set DONE")

    if len(split_size) == 3:
        logging.info(f"Start to create negative data points for the val set...")
        val = drop_samples_without_mol_file(df=val)
        df_metabolites_val, similarity_matrix_val = get_metabolites_and_similarities(df=val)
        logging.info(f"Number of metabolites in val set: {len(df_metabolites_val)}")
        val["Binding"] = 1
        val.reset_index(inplace=True, drop=True)
        val = create_negative_samples(df=val, df_metabolites=df_metabolites_val,
                                      similarity_matrix=similarity_matrix_val)
        val = map_negative_samples2embedding(val)
        logging.info(f"Creating negative data points for the val set DONE")

    if len(split_size) == 2:
        result, total_samples, test_ratio = two_split_report(train, test)
        logging.info(
            f"Data report after adding negative data and check for NaN or null cells in the data\n{result.to_string()}")
        logging.info(f"Total number of samples: {total_samples}")
        logging.info(f"Ratio of test set to total dataset: {test_ratio}")
    elif len(split_size) == 3:
        result, total_samples, test_ratio, val_ratio = three_split_report(train, test, val)
        logging.info(
            f"Data report after adding negative data and check for NaN or null cells in the data\n{result.to_string()}")
        logging.info(f"Total number of samples: {total_samples}")
        logging.info(f"Ratio of test set to dataset: {test_ratio}")
        logging.info(f"Ratio of val set to dataset: {val_ratio}")

    dict_train = collections.Counter(train["Binding"])
    dict_test = collections.Counter(test["Binding"])
    logging.info(f"The ratio of negative to positive data in train: {round(dict_train[0] / dict_train[1], 2)}")
    logging.info(f"The ratio of negative to positive data in test: {round(dict_test[0] / dict_test[1], 2)}")
    if len(split_size) == 3:
        dict_val = collections.Counter(val["Binding"])
        logging.info(f"The ratio of negative to positive data in val: {round(dict_val[0] / dict_val[1], 2)}")

    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    train.to_pickle(train_output_file)
    test.to_pickle(test_output_file)
    if len(split_size) == 3:
        val.to_pickle(val_output_file)

    init()
    logging.info(
        Fore.GREEN + f"***** PROCESS COMPLETED *****\nFor an overview,"
                     f"please review the Report_{split_method}{Data_suffix}_{len(split_size)}S.log file in Reports "
                     f"folder. " + Style.RESET_ALL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Please check the DataSAL webpage: https://datasail.readthedocs.io/en/latest/index.html")
    parser.add_argument('--split-method', type=str, required=True,
                        help="The split method should be one of [C2,C1e, C1f, I1e, I1f]")
    parser.add_argument('--split-size', type=int, nargs='+', required=True,
                        help="List of integers for splitting, e.g., 8 2 or 7 2 1")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input data (pickle file).")
    args = parser.parse_args()
    main(args)
