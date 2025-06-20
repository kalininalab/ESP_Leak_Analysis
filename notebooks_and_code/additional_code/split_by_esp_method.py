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


"""
All these functions have been taken from ESP repository:
https://github.com/AlexanderKroll/ESP
Code for creating cluster of enzyme by enzyme sequence identity. Code was created by Martin Engqvist:

"""
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
