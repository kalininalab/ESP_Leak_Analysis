import numpy as np
from os.path import join
import os
import time
import pandas as pd
import warnings
import sys
import argparse
from rdkit import RDLogger
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import subprocess
CURRENT_DIR = os.getcwd()
print(CURRENT_DIR)
warnings.filterwarnings("ignore")
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def calculate_fingerprints(smiles_list):
    """
    Generate fingerprints for a list of SMILES strings.
    """
    fps = [
        AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=2048)
        for smiles in smiles_list
    ]
    return fps


def rdkit_sim(fps):
    """
    Compute the Tanimoto similarity matrix for a list of RDKit fingerprints.
    """
    n = len(fps)
    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i, i] = 1
        matrix[i, :i] = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        matrix[:i, i] = matrix[i, :i]
    return matrix


def save_fasta(protein_list, filename="proteins.fasta"):
    """
    Save protein sequences to a FASTA file for DIAMOND input.
    """
    with open(filename, "w") as fasta_file:
        for i, sequence in enumerate(protein_list):
            fasta_file.write(f">seq_{i}\n{sequence}\n")


def run_diamond(protein_fasta, output_file, threads=128):
    """
    Run DIAMOND to get pairwise protein similarities with multiple threads.
    """
    subprocess.run(["diamond", "makedb", "--in", protein_fasta, "-d", "diamond_db"], check=True)
    subprocess.run([
        "diamond", "blastp",
        "-d", "diamond_db",
        "-q", protein_fasta,
        "-o", output_file,
        "--outfmt", "6",
        "--threads", str(threads)
    ], check=True)


def parse_diamond_output(output_file, num_proteins):
    """Parse DIAMOND output and construct a similarity matrix."""
    similarity_matrix = np.zeros((num_proteins, num_proteins))

    with open(output_file, "r") as file:
        for line in file:
            columns = line.strip().split()
            if len(columns) < 3:
                continue
            qseqid, sseqid, pident = columns[:3]
            i = int(qseqid.split('_')[1])
            j = int(sseqid.split('_')[1])
            similarity_matrix[i, j] = float(pident)
            similarity_matrix[j, i] = float(pident)

    return similarity_matrix


def similarity_matrix_proteins(protein_list):
    """Generate a protein similarity matrix using DIAMOND."""
    protein_fasta = "proteins.fasta"
    output_file = "diamond_output.tsv"

    save_fasta(protein_list, protein_fasta)
    run_diamond(protein_fasta, output_file)

    similarity_matrix = parse_diamond_output(output_file, len(protein_list))
    return similarity_matrix


def similarity_based_leakage_calculator(train_df, test_df, val_df=None):
    smiles_all = list(train_df["SMILES"]) + list(test_df["SMILES"]) + (list(val_df["SMILES"]))
    protein_all = list(train_df["Sequence"]) + list(test_df["Sequence"]) + (list(val_df["Sequence"]))

    print("Step 1: Generating fingerprints")
    start_time = time.time()
    smiles_fps = calculate_fingerprints(smiles_all)
    end_time = time.time()
    print(f"Time taken for step 1: {end_time - start_time:.2f} seconds")

    print("Step 2: Computing SMILES similarity matrix")
    start_time = time.time()
    smiles_similarity_matrix = rdkit_sim(smiles_fps)
    end_time = time.time()
    print(f"Time taken for step 2: {end_time - start_time:.2f} seconds")

    print("Step 3: Computing Diamond similarity matrix")
    protein_similarity_matrix = similarity_matrix_proteins(protein_all)
    print("Step 4: Calculate  leakage ")

    n_train, n_test, n_val = len(train_df), len(test_df), len(val_df)
    train_test_mask = np.zeros(smiles_similarity_matrix.shape, dtype=bool)
    train_test_mask[:n_train, n_train:n_train + n_test] = True
    train_val_mask = np.zeros(smiles_similarity_matrix.shape, dtype=bool)
    train_val_mask[:n_train, n_train + n_test:] = True
    test_val_mask = np.zeros(smiles_similarity_matrix.shape, dtype=bool)
    test_val_mask[n_train:n_train + n_test, n_train + n_test:] = True

    upper_indices = np.triu_indices(smiles_similarity_matrix.shape[0], k=0)
    total_smiles_similarity = np.sum(smiles_similarity_matrix[upper_indices])
    total_protein_similarity = np.sum(protein_similarity_matrix[upper_indices])

    train_test_smiles_leakage = np.sum(smiles_similarity_matrix[train_test_mask]) / total_smiles_similarity
    train_test_protein_leakage = np.sum(protein_similarity_matrix[train_test_mask]) / total_protein_similarity
    test_val_protein_leakage = np.sum(protein_similarity_matrix[test_val_mask]) / total_protein_similarity
    train_val_protein_leakage = np.sum(protein_similarity_matrix[train_val_mask]) / total_protein_similarity
    test_val_smiles_leakage = np.sum(smiles_similarity_matrix[test_val_mask]) / total_smiles_similarity
    train_val_smiles_leakage = np.sum(smiles_similarity_matrix[train_val_mask]) / total_smiles_similarity

    return {'train_test_smiles_leakage': round(train_test_smiles_leakage, 3),
            'train_test_protein_leakage': round(train_test_protein_leakage, 3),
            'train_val_smiles_leakage': round(train_val_smiles_leakage, 3),
            'train_val_protein_leakage': round(train_val_protein_leakage, 3),
            'test_val_smiles_leakage': round(test_val_smiles_leakage, 3),
            'test_val_protein_leakage': round(test_val_protein_leakage, 3)
            }


def main(args):
    results = []
    if args.split_scenario == 2:
        cv_folder = join("..", "data", f"{args.split_scenario}splits", f"CV_{args.molecule_column}")
        if not os.path.exists(cv_folder):
            os.makedirs(cv_folder)
        splits = ["C1e", "I1e", "C2", "C1f", "I1f", "ESP", "ESPC2"]
        for s1 in splits:
            cv_train = None
            cv_val = None
            if s1 in ["C1e", "I1e", "C2"]:
                cv_train = pd.read_pickle(join(CURRENT_DIR, "..", "data", f"{args.split_scenario}splits", f"CV_train_indices_{s1}_{args.molecule_column}.pkl"))
                cv_val = pd.read_pickle(join(CURRENT_DIR, "..", "data", f"{args.split_scenario}splits", f"CV_test_indices_{s1}_{args.molecule_column}.pkl"))
            elif s1 in ["C1f", "I1f", "ESP", "ESPC2"]:
                cv_train = pd.read_pickle(
                    join(CURRENT_DIR, "..", "data", f"{args.split_scenario}splits", f"CV_train_indices_{s1}_ESM1b_ts.pkl"))
                cv_val = pd.read_pickle(join(CURRENT_DIR, "..", "data", f"{args.split_scenario}splits", f"CV_test_indices_{s1}_ESM1b_ts.pkl"))
            train_ESP = pd.read_pickle(join("..", "data", f"{args.split_scenario}splits", f"train_{s1}_{args.split_scenario}S.pkl"))

            for i in range(len(cv_train)):
                list_train = cv_train[i]
                list_val = cv_val[i]
                train_cv_data = train_ESP[train_ESP.index.map(lambda x: x in list_train)]
                val_cv_data = train_ESP[train_ESP.index.map(lambda x: x in list_val)]
                train_cv_data.to_pickle(join(cv_folder, f"train_{s1}_fold{i}_{args.split_scenario}S.pkl"))
                val_cv_data.to_pickle(join(cv_folder, f"val_{s1}_fold{i}_{args.split_scenario}S.pkl"))
        for s2 in splits:
            print(f"Start leakage calculation for {s2}")
            for i in range(5):
                train_ESP = pd.read_pickle(join(cv_folder, f"train_{s2}_fold{i}_{args.split_scenario}S.pkl"))
                val_ESP = pd.read_pickle(join(cv_folder, f"val_{s2}_fold{i}_{args.split_scenario}S.pkl"))
                test_ESP = pd.read_pickle(join("..", "data", f"{args.split_scenario}splits", f"test_{s2}_{args.split_scenario}S.pkl"))
                leakage = similarity_based_leakage_calculator(train_ESP, test_ESP, val_ESP)
                leakage['Split method'] = s2
                leakage['Fold_number'] = i
                results.append(leakage)
    elif args.split_scenario == 3:
        splits = ["C1e", "I1e", "C1f", "I1f", "ESP","C2", "ESPC2"]
        for s in splits:
            print(f"Start leakage calculation for {s}")
            train_ESP = pd.read_pickle(join("..", "data", f"{args.split_scenario}splits", f"train_{s}_{args.split_scenario}S.pkl"))
            val_ESP = pd.read_pickle(join("..", "data", f"{args.split_scenario}splits", f"val_{s}_{args.split_scenario}S.pkl"))
            test_ESP = pd.read_pickle(join("..", "data", f"{args.split_scenario}splits", f"test_{s}_{args.split_scenario}S.pkl"))
            leakage = similarity_based_leakage_calculator(train_ESP, test_ESP, val_ESP)
            leakage['Split method'] = s
            results.append(leakage)
    df_results = pd.DataFrame(results)
    if args.split_scenario == 2:
        df_results.to_csv(
            join("..", "data", f"{args.split_scenario}splits", f"Similarity_leakage_ESM1bts_{args.molecule_column}_{args.split_scenario}S.csv"),
            index=False)
    elif args.split_scenario == 3:
        df_results.to_csv(
            join("..", "data", f"{args.split_scenario}splits", f"Similarity_leakage_{args.molecule_column}_{args.split_scenario}S.csv"),
            index=False)
    print("Processed  finished successfully ")
    print(f"Results are saved in folder {args.split_scenario}splits")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate data leakage between subsets")
    parser.add_argument('--split-scenario', type=int, required=True, help="THe --split-scenario is must be 2 or 3")
    parser.add_argument('--molecule-column', type=str, required=False, help="ECFP or PreGNN in case of two-split scenario")

    args = parser.parse_args()
    main(args)
