import pandas as pd
import numpy as np
import random
from os.path import join
import os
import re
import sys
import time
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from Bio import SeqIO
import warnings
#import torch
warnings.filterwarnings("ignore")

sys.path.append('./additional_code')
from data_preprocessing import *

CURRENT_DIR = os.getcwd()
print(CURRENT_DIR)


df_chebi_to_inchi = pd.read_csv(join(CURRENT_DIR, ".." ,"data", "data_ESP", "chebiID_to_inchi.tsv"), sep = "\t")
mol_folder = join(CURRENT_DIR, ".." ,"additional_data_ESP", "mol-files")

count = 0


def get_mol(met_ID):
    is_CHEBI_ID = (met_ID[0:5] == "CHEBI")
    is_InChI = (met_ID[0:5] == "InChI")
    if is_CHEBI_ID:
        try:
            ID = int(met_ID.split(" ")[0].split(":")[-1])
            Inchi = list(df_chebi_to_inchi["Inchi"].loc[df_chebi_to_inchi["ChEBI"] == float(ID)])[0]
            mol = Chem.inchi.MolFromInchi(Inchi)
        except:
            mol = None     
    elif is_InChI:
        try:
            mol = Chem.inchi.MolFromInchi(met_ID)
        except:
            mol = None
        
    else:
        try:
            mol = Chem.MolFromMolFile(mol_folder +  "/mol-files/" + met_ID + '.mol')
        except OSError:
            mol = None
            
    return(mol)


def drop_samples_without_mol_file(df):
    droplist = []
    for ind in df.index:
        if get_mol(met_ID = df["molecule ID"][ind]) is None:
            droplist.append(ind)

    df.drop(droplist, inplace = True)
    return(df)


def get_metabolites_and_similarities(df):
    df_metabolites = pd.DataFrame(data = {"ECFP": df["ECFP"], "ID": df["molecule ID"]})
    df_metabolites = df_metabolites.drop_duplicates()
    df_metabolites.reset_index(inplace = True, drop = True)


    ms = [get_mol(met_ID = df_metabolites["ID"][ind]) for ind in df_metabolites.index]
    fps = [Chem.RDKFingerprint(x) for x in ms]

    similarity_matrix = np.zeros((len(ms), len(ms)))
    for i in range(len(ms)):
        for j in range(len(ms)):
            similarity_matrix[i,j] = DataStructs.FingerprintSimilarity(fps[i],fps[j])
            
    return(df_metabolites, similarity_matrix)



def get_valid_list(met_ID, UID, forbidden_metabolites, df_metabolites, similarity_matrix, lower_bound =0.7, upper_bound =0.9):
    binding_met_IDs = list(df_UID_MID["molecule ID"].loc[df_UID_MID["Uniprot ID"] == UID])
    k = df_metabolites.loc[df_metabolites["ID"] == met_ID].index[0]

    similarities = similarity_matrix[k,:]
    selection = (similarities< upper_bound) * (similarities >lower_bound) 
    metabolites = list(df_metabolites["ID"].loc[selection])
    
    no_mets = list(set(binding_met_IDs + forbidden_metabolites))
    
    metabolites = [met for met in metabolites if (met not in no_mets)]
    return(metabolites)


def create_negative_samples(df, df_metabolites, similarity_matrix):
    start = time.time()
    UID_list = []
    MID_list = []
    Type_list = []
    forbidden_mets = []

    for ind in df.index:
        if ind % 100 ==0:
            print(ind)
            print("Time: %s [min]" % np.round(float((time.time()-start)/60),2))

            df2 = pd.DataFrame(data = {"Uniprot ID": UID_list, "molecule ID" : MID_list, "type" : Type_list})
            df2["Binding"] = 0
            df = pd.concat([df, df2], ignore_index=True)

            UID_list, MID_list, Type_list = [], [], []

            forbidden_mets_old = forbidden_mets.copy()
            all_mets = list(set(df["molecule ID"]))
            all_mets = [met for met in all_mets if not met in forbidden_mets_old]
            forbidden_mets = list(set([met for met in all_mets if 
                                       (np.mean(df["Binding"].loc[df["molecule ID"] == met]) < 1/4)]))
            forbidden_mets = forbidden_mets + forbidden_mets_old
            print(len(forbidden_mets))

        UID = df["Uniprot ID"][ind]
        Type = df["type"][ind]
        met_ID = df["molecule ID"][ind]

        metabolites = get_valid_list(met_ID = met_ID, UID = UID, forbidden_metabolites= forbidden_mets,
                                     df_metabolites = df_metabolites, similarity_matrix = similarity_matrix,
                                     lower_bound =0.7, upper_bound =0.95)
        lower_bound = 0.7
        while len(metabolites) < 2:
            lower_bound = lower_bound - 0.2
            metabolites = get_valid_list(met_ID = met_ID, UID = UID, forbidden_metabolites= forbidden_mets,
                                     df_metabolites = df_metabolites, similarity_matrix = similarity_matrix,
                                     lower_bound =lower_bound, upper_bound =0.95)
            if lower_bound <0:
                break
        
        if lower_bound < 0.7:
            global count
            count +=1
        
        new_metabolites =  random.sample(metabolites, min(3,len(metabolites)))

        for met in new_metabolites:
            UID_list.append(UID), MID_list.append(met), Type_list.append(Type)

    df2 = pd.DataFrame(data = {"Uniprot ID": UID_list, "molecule ID" : MID_list, "type" : Type_list})
    df2["Binding"] = 0

    df = pd.concat([df, df2], ignore_index = True)
    return(df)