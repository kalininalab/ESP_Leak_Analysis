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
import warnings
import torch
from colorama import init, Fore, Style

sys.path.append("./additional_code")
from additional_code.helper_functions import *
from additional_code.split_by_esp_method import *
from additional_code.negative_data_generator import *

warnings.filterwarnings("ignore")
CURRENT_DIR = os.getcwd()
print(CURRENT_DIR)

"""
The related code to clustering and 2-split method (train:test) was sourced from the ESP repository. 
We have modified it to accommodate a 3-split method (train:test:val).
"""


def main(args):
    CURRENT_DIR = os.getcwd()
    input_path = args.input_path
    df_name = input_path.split("/")[-1]
    Data_suffix = ""
    if "_" in df_name:
        Data_suffix = "_" + df_name.split(".")[0].split("_")[-1]
    split_size = args.split_size
    if len(split_size) not in [2, 3]:
        raise ValueError("The split-size argument must be a list of either two or three integers.")

    log_file = os.path.join(CURRENT_DIR, "..", "data", "Reports", "split_report",
                            f"Report_ESP{"C2" if "C2" in df_name else ""}{Data_suffix}_{len(split_size)}S.log")
    if os.path.exists(log_file):
        os.remove(log_file)
    setup_logging(log_file)
    logging.info(f"Current Directory: {CURRENT_DIR}")

    train_output_file = os.path.join(CURRENT_DIR, "..", "data", f"{len(split_size)}splits",
                                     f"train_ESP{"C2" if "C2" in df_name else ""}{Data_suffix}_{len(split_size)}S.pkl")
    test_output_file = os.path.join(CURRENT_DIR, "..", "data", f"{len(split_size)}splits",
                                    f"test_ESP{"C2" if "C2" in df_name else ""}{Data_suffix}_{len(split_size)}S.pkl")
    val_output_file = os.path.join(CURRENT_DIR, "..", "data", f"{len(split_size)}splits",
                                   f"val_ESP{"C2" if "C2" in df_name else ""}{Data_suffix}_{len(split_size)}S.pkl")

    data = pd.read_pickle(input_path)
    data.drop(columns=["ids"], inplace=True)
    data.reset_index(drop=True, inplace=True)

    logging.info(f"Start Clustering the data with CD-HIT")
    ofile = open(
        join(CURRENT_DIR, "..", "data", "clusters",
             f"all_sequences_ESP{"C2" if "C2" in df_name else ""}{Data_suffix}_{len(split_size)}S.fasta"), "w")

    for ind in data.index:
        seq = data["Sequence"][ind]
        if not pd.isnull(seq):
            seq_end = seq.find("#")
            seq = seq[:seq_end]
            ofile.write(">" + str(data["Uniprot ID"][ind]) + "\n" + seq + "\n")
    ofile.close()

    cluster_folder = join(CURRENT_DIR, "..", "data", "clusters")
    start_folder = cluster_folder
    cluster_all_levels(start_folder,
                       cluster_folder,
                       filename=f"all_sequences_ESP{"C2" if "C2" in df_name else ""}{Data_suffix}_{len(split_size)}S")

    cluster_folder = join(CURRENT_DIR, "..", "data", "clusters")

    # cluster the fasta files
    start_folder = cluster_folder
    cluster_all_levels_80(start_folder,
                          cluster_folder,
                          filename=f"all_sequences_ESP{"C2" if "C2" in df_name else ""}{Data_suffix}_{len(split_size)}S")

    # collect cluster members
    df_80 = find_cluster_members_80(folder=cluster_folder,
                                    filename=f"all_sequences_ESP{"C2" if "C2" in df_name else ""}{Data_suffix}_{len(split_size)}S")
    logging.info(f"Clustering report for 80% similarity\n{df_80.describe()}")

    cluster_all_levels_60(start_folder,
                          cluster_folder,
                          filename=f"all_sequences_ESP{"C2" if "C2" in df_name else ""}{Data_suffix}_{len(split_size)}S")

    # collect cluster members
    df_60 = find_cluster_members_60(folder=cluster_folder,
                                    filename=f"all_sequences_ESP{"C2" if "C2" in df_name else ""}{Data_suffix}_{len(split_size)}S")
    logging.info(f"Clustering report for 60% similarity\n{df_60.describe()}")

    # cluster the fasta files
    cluster_all_levels(start_folder,
                       cluster_folder,
                       filename=f"all_sequences_ESP{"C2" if "C2" in df_name else ""}{Data_suffix}_{len(split_size)}S")

    # collect cluster members
    df_40 = find_cluster_members(folder=cluster_folder,
                                 filename=f"all_sequences_ESP{"C2" if "C2" in df_name else ""}{Data_suffix}_{len(split_size)}S")

    logging.info(f"Clustering report for 40% similarity\n{df_40.describe()}")

    data["cluster"] = np.nan
    for ind in df_80.index:
        member = df_80["member"][ind]
        cluster = df_80["cluster"][ind]
        data.loc[data["Uniprot ID"] == member, "cluster"] = cluster

    logging.info(f"Clustering the data with CD-HIT DONE")

    clusters = list(set(data["cluster"]))
    random.seed(1)
    random.shuffle(clusters)
    logging.info(f"Start Splitting the data based one {split_size} split size ")
    train_clusters = None
    test_clusters = None
    val_clusters = None
    if len(split_size) == 2:
        n = int(len(clusters) * (split_size[0] / 10))
        train_clusters = clusters[:n]
        test_clusters = clusters[n:]
    elif len(split_size) == 3:
        n_train = int(len(clusters) * (split_size[0] / 10))
        n_test = int(len(clusters) * (split_size[1] / 10))
        train_clusters = clusters[:n_train]
        test_clusters = clusters[n_train:n_train + n_test]
        val_clusters = clusters[n_train + n_test:]

    training_UIDs = data["Uniprot ID"].loc[data["cluster"].isin(train_clusters)]
    test_UIDs = data["Uniprot ID"].loc[data["cluster"].isin(test_clusters)]
    if len(split_size) == 3:
        validation_UIDs = data["Uniprot ID"].loc[data["cluster"].isin(val_clusters)]

    df_80["split"] = np.nan
    df_60["split"] = np.nan
    df_40["split"] = np.nan
    df_80["split"].loc[df_80["cluster"].isin(train_clusters)] = "train"
    df_80["split"].loc[df_80["cluster"].isin(test_clusters)] = "test"
    if len(split_size) == 3:
        df_80["split"].loc[df_80["cluster"].isin(val_clusters)] = "validation"
    train_members = list(df_80["member"].loc[df_80["split"] == "train"])
    test_members = list(df_80["member"].loc[df_80["split"] == "test"])
    if len(split_size) == 3:
        validation_members = list(df_80["member"].loc[df_80["split"] == "validation"])

    df_60["split"].loc[df_60["member"].isin(train_members)] = "train"
    df_60["split"].loc[df_60["member"].isin(test_members)] = "test"
    df_40["split"].loc[df_40["member"].isin(train_members)] = "train"
    df_40["split"].loc[df_40["member"].isin(test_members)] = "test"
    if len(split_size) == 3:
        df_60["split"].loc[df_60["member"].isin(validation_members)] = "validation"
        df_40["split"].loc[df_40["member"].isin(validation_members)] = "validation"

    train = data.loc[data["Uniprot ID"].isin(training_UIDs)]
    test = data.loc[data["Uniprot ID"].isin(test_UIDs)]
    if len(split_size) == 3:
        val = data.loc[data["Uniprot ID"].isin(validation_UIDs)]

    df_80["identity"] = np.nan
    df_80["identity"].loc[df_80["split"].isin(["test"])] = "60-80%"

    test_indices = list(df_80.loc[~pd.isnull(df_80["identity"])].index)

    for ind in test_indices:

        member = df_80["member"][ind]
        cluster = list(df_40["cluster"].loc[df_40["member"] == member])[0]
        cluster_splits = list(df_40["split"].loc[df_40["cluster"] == cluster])
        if not "train" in cluster_splits:
            df_80["identity"][ind] = "<40%"
        else:
            cluster = list(df_60["cluster"].loc[df_60["member"] == member])[0]
            cluster_splits = list(df_60["split"].loc[df_60["cluster"] == cluster])
            if not "train" in cluster_splits:
                df_80["identity"][ind] = "40-60%"

        if ind % 1000 == 0:
            print(ind)

    ind = 0
    data["identity"] = np.nan
    for ind in data.index:
        try:
            data["identity"][ind] = list(df_80["identity"].loc[df_80["member"] == str(ind)])[0]
        except:
            None

    data.to_pickle(join(CURRENT_DIR, "..", "data", f"{len(split_size)}splits",
                        f"SeqIdentities_ESP{"C2" if "C2" in df_name else ""}{Data_suffix}_{len(split_size)}S.pkl"))

    if len(split_size) == 2:
        result, total_samples, test_ratio = two_split_report(train, test)
        logging.info(
            f"Data Report after splitting data by ESP split method and check for NaN or null cells in the data\n{result.to_string()}")
        logging.info(f"Total number of samples: {total_samples}")
        logging.info(f"Ratio of test set to total dataset: {test_ratio}")
    elif len(split_size) == 3:
        result, total_samples, test_ratio, val_ratio = three_split_report(train, test, val)
        logging.info(
            f"Data Report after splitting data by ESP split method and check for NaN or null cells in the data\n{result.to_string()}")
        logging.info(f"Ratio of test set to dataset: {test_ratio}")
        logging.info(f"Ratio of val set to dataset: {val_ratio}")
    logging.info(f"Drop nan data based on cluster column, if exist")
    train.dropna(subset=['cluster'], inplace=True)
    train.reset_index(drop=True, inplace=True)
    test.dropna(subset=['cluster'], inplace=True)
    test.reset_index(drop=True, inplace=True)
    if len(split_size) == 3:
        val.dropna(subset=['cluster'], inplace=True)
        val.reset_index(drop=True, inplace=True)

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
        logging.info(f"Number of metabolites in training set: {len(df_metabolites_val)}")
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
        Fore.GREEN + f"***** PROCESS COMPLETED: For an overview, "
                     f"please review the Report_ESP{"C2" if "C2" in df_name else ""}{Data_suffix}_{len(split_size)}S.log file in "
                     f"Reports folder. *****" + Style.RESET_ALL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"This script generates a control case for each split method of DataSAIL by combining the related "
                    f"split results from DataSAIL and re-splitting them using the ESP method")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input data (pickle file).")
    parser.add_argument('--split-size', type=int, nargs='+', required=True,
                        help="List of integers for splitting, e.g., 8 2 or 7 2 1")
    args = parser.parse_args()
    main(args)
