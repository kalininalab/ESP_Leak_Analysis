import torch
from torch.utils.data import Dataset
import os
from os.path import join
import pandas as pd
from utils.train_utils import is_cuda


class SMILESProteinDataset(Dataset):
    def __init__(self,
                 split_method,
                 embed_dir,
                 data_path,
                 train: bool,
                 device,
                 gpu,
                 random_state,
                 binary_task: bool,
                 extraction_mode=False):
        self.train = train
        self.device = device
        self.gpu = gpu
        self.random_state = random_state
        self.binary_task = binary_task
        self.max_prot_seq_len = 1018
        self.max_smiles_seq_len = 256
        self.train_or_test = 'train' if train else 'test'
        self.binary_task = binary_task
        self.split_method = split_method

        if self.train:
            self.df = pd.read_pickle(join(data_path, f"train_{self.split_method}_3S.pkl"))
        else:
            self.df = pd.read_pickle(join(data_path, f"val_{self.split_method}_3S.pkl"))

        prot_path = join(embed_dir, "dataESP_enzy_embeddings.pt")
        smiles_path = join(embed_dir, "dataESP_smiles_embeddings.pt")

        # FIX: Use device directly, not gpu index
        map_loc = device
        self.protein_embeds = torch.load(prot_path, map_location=map_loc)
        self.smiles_embeds = torch.load(smiles_path, map_location=map_loc)

        # Filter dataframe to only include IDs present in embeddings
        valid_proteins = set(self.protein_embeds.keys())
        valid_smiles = set(self.smiles_embeds.keys())
        self.df = self.df[self.df["Uniprot ID"].isin(valid_proteins) &
                          self.df["molecule ID"].isin(valid_smiles)].reset_index(drop=True)

        if self.train:
            self.df = self.df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        self.total_datacount = len(self.df)

    def __len__(self):
        return self.total_datacount

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = float(row["Binding"])
        if self.binary_task:
            label = int(label)

        # Ensure tensors and move to float
        uniprot_id = row["Uniprot ID"]
        molecule_id = row["molecule ID"]

        # Retrieve embeddings from dicts
        protein_emb = self.protein_embeds[uniprot_id]
        smiles_emb = self.smiles_embeds[molecule_id]

        # Build attention masks
        smiles_attn_mask = torch.zeros(self.max_smiles_seq_len)
        smiles_attn_mask[:smiles_emb.shape[0]] = 1
        protein_attn_mask = torch.zeros(self.max_prot_seq_len)
        protein_attn_mask[:protein_emb.shape[0]] = 1

        smiles_padding = (0, 0, 0, self.max_smiles_seq_len - smiles_emb.shape[0])
        prot_padding = (0, 0, 0, self.max_prot_seq_len - protein_emb.shape[0])

        smiles_emb = torch.nn.functional.pad(smiles_emb, smiles_padding, mode='constant', value=0)
        protein_emb = torch.nn.functional.pad(protein_emb, prot_padding, mode='constant', value=0)

        labels = torch.tensor([label])
        labels.requires_grad = False
        smiles_emb = smiles_emb.detach()
        protein_emb = protein_emb.detach()
        return smiles_emb, smiles_attn_mask, protein_emb, protein_attn_mask, labels, idx