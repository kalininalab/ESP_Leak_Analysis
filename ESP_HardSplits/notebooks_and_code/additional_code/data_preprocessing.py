import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
import matplotlib.pyplot as plt
from rdkit.Chem import AllChem
import random
import time
import gzip
import os
import collections
from os.path import join

CURRENT_DIR = os.getcwd()
mol_folder = join(CURRENT_DIR, ".." ,"additional_data_ESP", "mol-files")
df_UID_MID = pd.read_pickle(join(CURRENT_DIR, ".." ,"data", "data_ESP", "df_UID_MID.pkl"))
df_chebi_to_inchi = pd.read_csv(join(CURRENT_DIR, ".." ,"data", "data_ESP", "chebiID_to_inchi.tsv"), sep = "\t")


# Code for creating cluster of enzyme by enzyme sequence identity. Code was created by Martin Engqvist:
import os
import re
import argparse
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
from Bio import SeqIO
from os.path import join, exists, abspath, isdir, dirname
import subprocess

def remove_header_gaps(folder, infile, outfile):
    '''
    CD-HIT truncates fasta record headers at whitespace,
    need to remove these before I run the algorithm
    '''
    if not exists(outfile):
        with open(join(folder, outfile), 'w') as f:
            for record in SeqIO.parse(join(folder, infile), 'fasta'):
                header = record.description.replace(' ', '_')
                seq = str(record.seq)

                f.write('>%s\n%s\n' % (header, seq))


def run_cd_hit(infile, outfile, cutoff, memory):
    '''
    Run a specific cd-hit command
    '''
    # get the right word size for the cutoff
    if cutoff < 0.5:
        word = 2
    elif cutoff < 0.6:
        word = 3
    elif cutoff < 0.7:
        word = 4
    else:
        word = 5

    mycmd = '%s -i %s -o %s -c %s -n %s -T 1 -M %s -d 0' % ('cd-hit', infile, outfile, cutoff, word, memory)
    print(mycmd)
    process = subprocess.Popen(mycmd, shell=True, stdout=subprocess.PIPE)
    process.wait()
    
    
def cluster_all_levels_60(start_folder, cluster_folder, filename):
    '''
    Run cd-hit on fasta file to cluster on all levels
    '''
    mem = 2000  # memory in mb

    cutoff = 1.0
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(start_folder, '%s.fasta' % filename), outfile=outfile, cutoff=cutoff, memory=mem)

    cutoff = 0.9
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(cluster_folder, '%s_clustered_sequences_100.fasta' % filename), outfile=outfile,
                   cutoff=cutoff, memory=mem)
        
    cutoff = 0.8
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(cluster_folder, '%s_clustered_sequences_90.fasta' % filename), outfile=outfile,
                   cutoff=cutoff, memory=mem)

    cutoff = 0.7
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(cluster_folder, '%s_clustered_sequences_80.fasta' % filename), outfile=outfile,
                   cutoff=cutoff, memory=mem)

    cutoff = 0.6
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(cluster_folder, '%s_clustered_sequences_70.fasta' % filename), outfile=outfile,
                   cutoff=cutoff, memory=mem)
        
        
def cluster_all_levels_80(start_folder, cluster_folder, filename):
    '''
    Run cd-hit on fasta file to cluster on all levels
    '''
    mem = 2000  # memory in mb

    cutoff = 1.0
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(start_folder, '%s.fasta' % filename), outfile=outfile, cutoff=cutoff, memory=mem)

    cutoff = 0.9
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(cluster_folder, '%s_clustered_sequences_100.fasta' % filename), outfile=outfile,
                   cutoff=cutoff, memory=mem)
        
    cutoff = 0.8
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(cluster_folder, '%s_clustered_sequences_90.fasta' % filename), outfile=outfile,
                   cutoff=cutoff, memory=mem)



def cluster_all_levels(start_folder, cluster_folder, filename):
    '''
    Run cd-hit on fasta file to cluster on all levels
    '''
    mem = 2000  # memory in mb

    cutoff = 1.0
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(start_folder, '%s.fasta' % filename), outfile=outfile, cutoff=cutoff, memory=mem)

    cutoff = 0.9
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(cluster_folder, '%s_clustered_sequences_100.fasta' % filename), outfile=outfile,
                   cutoff=cutoff, memory=mem)
        
    cutoff = 0.8
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(cluster_folder, '%s_clustered_sequences_90.fasta' % filename), outfile=outfile,
                   cutoff=cutoff, memory=mem)

    cutoff = 0.7
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(cluster_folder, '%s_clustered_sequences_80.fasta' % filename), outfile=outfile,
                   cutoff=cutoff, memory=mem)

    cutoff = 0.6
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(cluster_folder, '%s_clustered_sequences_70.fasta' % filename), outfile=outfile,
                   cutoff=cutoff, memory=mem)
        
    cutoff = 0.5
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(cluster_folder, '%s_clustered_sequences_60.fasta' % filename), outfile=outfile,
                   cutoff=cutoff, memory=mem)

    cutoff = 0.4
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(cluster_folder, '%s_clustered_sequences_50.fasta' % filename), outfile=outfile,
                   cutoff=cutoff, memory=mem)


def parse_cd_hit(path_to_clstr):
    '''
    Gather the clusters of CD-hit output `path_to_clust` into a dict.
    '''
    # setup regular expressions for parsing
    pat_id = re.compile(r">(.+?)\.\.\.")
    is_center = re.compile(r">(.+?)\.\.\. \*")

    with open(path_to_clstr) as f:
        clusters = {}
        cluster = []
        id_clust = None
        next(f)  # advance first cluster header
        for line in f:
            if line.startswith(">"):
                # if cluster ended, flush seq ids to it
                clusters[id_clust] = cluster
                cluster = []
                continue
            match = pat_id.search(line)
            if match:
                if is_center.search(line):
                    id_clust = match[1]
                else:
                    cluster.append(match[1])
        clusters[id_clust] = cluster
    return clusters


def scale_up_cd_hit(paths_to_clstr):
    '''
    Hierarchically expand CD-hit clusters.

    Parameters
    ----------
    paths_to_clstr: list[str]
        paths to rest of the cd-hit output files, sorted by
        decreasing similarity (first is 100).

    Output
    ------
    clust_now: dict
        id: ids

    '''
    clust_above = parse_cd_hit(paths_to_clstr[0])

    for path in paths_to_clstr[1:]:
        clust_now = parse_cd_hit(path)
        for center in clust_now:
            clust_now[center] += [
                seq
                for a_center in clust_now[center] + [center]
                for seq in clust_above[a_center]
            ]
        clust_above = clust_now

    return clust_above


def find_cluster_members(folder, filename):
    '''
    Go through the cluster files and collect
    all the cluster members, while indicating
    which belongs where.
    '''
    # get a list of filenames
    CLUSTER_FILES = [
        join(folder, filename + f"_clustered_sequences_{sim}.fasta.clstr")
        for sim in [100, 90, 80, 70, 60, 50, 40]
    ]

    # collect all cluster members
    clusters = scale_up_cd_hit(CLUSTER_FILES)
    ind_clusters = {}
    i = 0
    for clus in clusters:
        ind_clusters[i] = [clus] + clusters[clus]
        i += 1

    # convert to format that is suitable for data frames
    clusters_for_df = {'cluster': [], 'member': []}
    for ind in ind_clusters:
        for member in ind_clusters[ind]:
            clusters_for_df['cluster'].append(ind)
            clusters_for_df['member'].append(member)

    df = pd.DataFrame.from_dict(clusters_for_df, orient='columns')

    return df


def find_cluster_members_80(folder, filename):
    '''
    Go through the cluster files and collect
    all the cluster members, while indicating
    which belongs where.
    '''
    # get a list of filenames
    CLUSTER_FILES = [
        join(folder, filename + f"_clustered_sequences_{sim}.fasta.clstr")
        for sim in [100, 90, 80]
    ]

    # collect all cluster members
    clusters = scale_up_cd_hit(CLUSTER_FILES)
    ind_clusters = {}
    i = 0
    for clus in clusters:
        ind_clusters[i] = [clus] + clusters[clus]
        i += 1

    # convert to format that is suitable for data frames
    clusters_for_df = {'cluster': [], 'member': []}
    for ind in ind_clusters:
        for member in ind_clusters[ind]:
            clusters_for_df['cluster'].append(ind)
            clusters_for_df['member'].append(member)

    df = pd.DataFrame.from_dict(clusters_for_df, orient='columns')

    return df

def find_cluster_members_60(folder, filename):
    '''
    Go through the cluster files and collect
    all the cluster members, while indicating
    which belongs where.
    '''
    # get a list of filenames
    CLUSTER_FILES = [
        join(folder, filename + f"_clustered_sequences_{sim}.fasta.clstr")
        for sim in [100, 90, 80,70,60]
    ]

    # collect all cluster members
    clusters = scale_up_cd_hit(CLUSTER_FILES)
    ind_clusters = {}
    i = 0
    for clus in clusters:
        ind_clusters[i] = [clus] + clusters[clus]
        i += 1

    # convert to format that is suitable for data frames
    clusters_for_df = {'cluster': [], 'member': []}
    for ind in ind_clusters:
        for member in ind_clusters[ind]:
            clusters_for_df['cluster'].append(ind)
            clusters_for_df['member'].append(member)

    df = pd.DataFrame.from_dict(clusters_for_df, orient='columns')

    return df


def kfold_by(df, key, k=5):
    """K-Split dataset `df` by values in `key` into `k` groups.

    Parameters
    ----------
    df: pandas.DataFrame
    key: str
        columns to use as splitting
    k: int
        number of groups.

    Returns
    -------
    k*(groups): pandas.DataFrame
        each df is the training set of the fold

    """
    kf = KFold(n_splits=k, random_state=4321, shuffle=True)
    set_keys = np.unique(df[key])
    return [
        df[df[key].isin(set_keys[train_index])]
        for train_index, _ in kf.split(set_keys)
    ]


def split_by(df, key, frac=0.8):
    """Split dataset `df` by values in `key`.

    Parameters
    ----------
    df: pandas.DataFrame
    key: str
        columns to use as splitting
    frac: float
        fraction of `key` groups into `df`.

    Returns
    -------
    (train, test, valid): pandas.DataFrames

    """
    # shuffle the data frame
    df = df.sample(frac=1, random_state=4321).reset_index(drop=True)

    # get all the unique identifiers
    set_keys = np.unique(df[key])

    # get the training identifiers
    train_clusters = np.random.choice(
        set_keys, size=int(len(set_keys) * frac), replace=False
    )
    train = df[df[key].isin(train_clusters)]

    # from the remaining ones, put half as validation and half as test
    remaining = df[~df.index.isin(train.index)]
    # valid and test sets will have equal sizes of 1-frac
    # at this point we are not worried about `key` anymore
    valid = remaining.sample(frac=1 / 2)
    test = remaining[~remaining.index.isin(valid.index)]
    return train, test, valid


def make_splits(folder, df):
    '''
    Takes an input data frame with information on cluster
    belongings and generates train/validation/test splits for DL.
    '''
    # make train/validation/test splits for DL
    train, validation, test = split_by(df, "cluster", frac=0.8)

    train.drop('cluster', axis=1).to_csv(join(folder, f"split_training.tsv"),
                 sep="\t", index=False, header=False)

    validation.drop('cluster', axis=1).to_csv(join(folder, f"split_validation.tsv"),
                      sep="\t", index=False, header=False)

    test.drop('cluster', axis=1).to_csv(join(folder, "split_test.tsv"),
                sep="\t", index=False, header=False)


##########################################

def two_split_report(train_set, test_set):
    # Calculate NaN counts and empty string counts for train and test sets
    nan_check_train_set = train_set.isnull().sum()
    empty_check_train_set = train_set.applymap(lambda x: isinstance(x, str) and x == '').sum()

    nan_check_test_set = test_set.isnull().sum()
    empty_check_test_set = test_set.applymap(lambda x: isinstance(x, str) and x == '').sum()

    # Concatenate results side by side
    result = pd.concat(
        [nan_check_train_set, empty_check_train_set, nan_check_test_set, empty_check_test_set], axis=1)
    result.columns = ['NaNtrain', 'Emptytrain', 'NaNtest', 'Emptytest']
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


def plot_top_keys_values(df, key_column, value_column, xlabel, ylabel, title, color='blue', figsize=(12, 10),
                         top_count=30):
    dict = collections.Counter(df[key_column])
    top = dict.most_common(top_count)
    print(top)
    keys, values = zip(*top)
    plt.figure(figsize=figsize)
    plt.bar(keys, values, color=color, alpha=0.8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=90, fontsize='xx-small')
    plt.subplots_adjust(bottom=0.25)
    plt.legend()
    plt.show()

