from tqdm import tqdm
import esm
import torch
from transformers import AutoModel, AutoTokenizer
from rdkit import Chem
import numpy as np
import pandas as pd
import gc
import argparse
import sys
import os
from os.path import join

current_dir = os.getcwd()


def esm2_embeddings_2560(esm, esm_alphabet, peptide_sequence_list, device):
    esm = esm.eval().to(device)
    batch_converter = esm_alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(peptide_sequence_list)
    batch_lens = (batch_tokens != esm_alphabet.padding_idx).sum(1)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = esm(batch_tokens, repr_layers=[36], return_contacts=False)
    token_representations = results["representations"][36].cpu()
    del results, batch_tokens
    torch.cuda.empty_cache()
    gc.collect()
    return token_representations[:, 1:-1, :].mean(1)


def esm1b_embeddings(esm, esm_alphabet, peptide_sequence_list, device):
    esm = esm.eval().to(device)
    batch_converter = esm_alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(peptide_sequence_list)
    batch_lens = (batch_tokens != esm_alphabet.padding_idx).sum(1)
    batch_tokens = batch_tokens.to(device)
    batch_tokens = batch_tokens[:, :1022]
    with torch.no_grad():
        results = esm(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33].cpu()
    del results, batch_tokens
    torch.cuda.empty_cache()
    gc.collect()
    # Return ALL token representations instead of mean pooled
    return token_representations[:, 1:-1, :]  # Remove [CLS] and [SEP] tokens # Shape: [seq_len, 768]


def MolFormer_embedding(model_smiles, tokenizer, SMILES_list, device):
    inputs = tokenizer(SMILES_list, padding=True, return_tensors="pt").to(device)
    model_smiles = model_smiles.to(device)
    with torch.no_grad():
        outputs = model_smiles(**inputs)
    return outputs.pooler_output.cpu()


def chemberta_embedding(model_smiles, tokenizer, SMILES_list, device):
    """ChemBERTa embedding function returning full sequence representations"""
    smiles = SMILES_list[0]
    model_smiles = model_smiles.to(device)
    tokens = tokenizer(
        smiles,
        max_length=500,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model_smiles(**tokens)
        last_hidden_state = outputs.last_hidden_state  # Shape: [1, seq_len, hidden_size]
    # Return the full sequence representation instead of mean pooled
    return last_hidden_state.cpu()  # Shape: [1, seq_len, 768]


def process_enzyme_chunk_esm2(enzy_chunk, model, alphabet, device, chunk_name):
    """Process a chunk of enzyme sequences and return embeddings dictionary"""
    embeddings_enzy_dict = {}
    print(f"Processing enzyme chunk {chunk_name} with {len(enzy_chunk)} sequences...")
    for i in tqdm(range(len(enzy_chunk)), desc=f"Enzymes {chunk_name}"):
        uniprot_id = enzy_chunk['Uniprot ID'].iloc[i]
        seq_enzy = enzy_chunk['Sequence'].iloc[i]
        if len(seq_enzy) < 5500:
            tuple_sequence = tuple(['protein', seq_enzy])
            peptide_sequence_list = [tuple_sequence]
            try:
                embedding = esm2_embeddings_2560(model, alphabet, peptide_sequence_list, device)
                embeddings_enzy_dict[uniprot_id] = embedding.squeeze(0)  # Keep as tensor
            except Exception as e:
                print(f"Error processing enzyme {uniprot_id}: {e}")
                continue
    return embeddings_enzy_dict


def process_enzyme_chunk_esm1b(enzy_chunk, model, alphabet, device, chunk_name):
    """Process a chunk of enzyme sequences and return embeddings dictionary"""
    embeddings_enzy_dict = {}
    print(f"Processing enzyme chunk {chunk_name} with {len(enzy_chunk)} sequences...")
    for i in tqdm(range(len(enzy_chunk)), desc=f"Enzymes {chunk_name}"):
        uniprot_id = enzy_chunk['Uniprot ID'].iloc[i]
        seq_enzy = enzy_chunk['Sequence'].iloc[i]
        tuple_sequence = tuple(['protein', seq_enzy])
        peptide_sequence_list = [tuple_sequence]
        try:
            embedding = esm1b_embeddings(model, alphabet, peptide_sequence_list, device)
            embeddings_enzy_dict[uniprot_id] = embedding.squeeze(0)  # Keep as tensor
        except Exception as e:
            print(f"Error processing enzyme {uniprot_id}: {e}")
            continue
    return embeddings_enzy_dict


def main(args):
    base_filename = args.filename.split('.')[0]
    num_chunks = args.num_chunks
    model_type = args.model

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if device == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif device == 'mps':
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
    # Load data and remove duplicates
    ESP_train_df = pd.read_pickle(join(current_dir, "..", "data", "data_ESP", f"{args.filename}"))

    # Remove duplicates based on Uniprot ID and molecule ID
    unique_enzy_df = ESP_train_df.drop_duplicates(subset=['Uniprot ID'])[['Uniprot ID', 'Sequence']]
    unique_smiles_df = ESP_train_df.drop_duplicates(subset=['molecule ID'])[['molecule ID', 'SMILES']]

    print(f"Original dataset size: {ESP_train_df.shape[0]}")
    print(f"Unique enzymes: {unique_enzy_df.shape[0]}")
    print(f"Unique molecules: {unique_smiles_df.shape[0]}")

    # Load models
    model_smiles=None
    tokenizer=None
    model_pro=None
    alphabet=None
    if model_type == 'FusionESP':
        smiles_model = "ibm/MoLFormer-XL-both-10pct"
        model_smiles = AutoModel.from_pretrained(smiles_model,
                                                 deterministic_eval=True,
                                                 trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(smiles_model,
                                                  trust_remote_code=True)
        model_pro, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    elif model_type == 'ProSmith':
        smiles_model = "DeepChem/ChemBERTa-77M-MTR"
        tokenizer = AutoTokenizer.from_pretrained(smiles_model)
        model_smiles = AutoModel.from_pretrained(smiles_model)  # Changed to AutoModel
        model_pro, alphabet = esm.pretrained.load_model_and_alphabet("esm1b_t33_650M_UR50S")

    # Split enzyme data into chunks

    enzyme_chunks = np.array_split(unique_enzy_df, num_chunks)

    print(f"Splitting {len(unique_enzy_df)} unique enzymes into {num_chunks} chunks...")
    for i, chunk in enumerate(enzyme_chunks):
        print(f"Chunk {i + 1}: {len(chunk)} sequences")

    # Process enzyme chunks separately
    all_enzy_embeddings = {}

    for i, enzy_chunk in enumerate(enzyme_chunks):
        chunk_name = f"chunk_{i + 1}_of_{num_chunks}"

        # Process this chunk
        if model_type=='FusionESP':
            chunk_embeddings = process_enzyme_chunk_esm2(enzy_chunk, model_pro, alphabet, device, chunk_name)
            all_enzy_embeddings.update(chunk_embeddings)
        # Process this chunk
        elif model_type=='ProSmith':
            chunk_embeddings = process_enzyme_chunk_esm1b(enzy_chunk, model_pro, alphabet, device, chunk_name)
            all_enzy_embeddings.update(chunk_embeddings)

        # Save intermediate results
        temp_output_path = join(current_dir, "..", "data", f"embedding_results_{model_type}", "temp")
        os.makedirs(temp_output_path, exist_ok=True)

        temp_file = join(temp_output_path, f"temp_enzy_chunk_{i + 1}.pt")
        torch.save(chunk_embeddings, temp_file)
        print(f"Saved temporary chunk {i + 1} with {len(chunk_embeddings)} embeddings")

        # Clear memory between chunks
        del chunk_embeddings
        torch.cuda.empty_cache()
        gc.collect()

    # Generate SMILES embeddings (less memory intensive, so we can process all at once)
    embeddings_smiles_dict = {}
    print("Generating SMILES embeddings...")

    for i in tqdm(range(unique_smiles_df.shape[0]), desc="Processing Molecules"):
        molecule_id = unique_smiles_df['molecule ID'].iloc[i]
        seq_smiles = unique_smiles_df['SMILES'].iloc[i]

        try:
            canonical_smiles = Chem.CanonSmiles(seq_smiles)
            smiles_list = [canonical_smiles]
            embedding=None
            if model_type == 'FusionESP':
                embedding = MolFormer_embedding(model_smiles, tokenizer, smiles_list, device)
            elif model_type == 'ProSmith':
                embedding = chemberta_embedding(model_smiles, tokenizer, smiles_list, device)
            embeddings_smiles_dict[molecule_id] = embedding.squeeze(0)  # Keep as tensor
        except Exception as e:
            print(f"Error processing molecule {molecule_id}: {e}")
            continue

    # Save final results as .pt files
    output_path = join(current_dir, "..", "data", f"embedding_results_{model_type}")
    os.makedirs(output_path, exist_ok=True)

    # Save as .pt files
    output_file_enzy = f"{base_filename}_enzy_embeddings.pt"
    output_file_smiles = f"{base_filename}_smiles_embeddings.pt"

    torch.save(all_enzy_embeddings, join(output_path, output_file_enzy))
    torch.save(embeddings_smiles_dict, join(output_path, output_file_smiles))

    # Clean up temporary files
    temp_output_path = join(current_dir, "..", "data", f"embedding_results_{model_type}", "temp")
    if os.path.exists(temp_output_path):
        for temp_file in os.listdir(temp_output_path):
            if temp_file.startswith("temp_enzy_chunk_"):
                os.remove(join(temp_output_path, temp_file))
        os.rmdir(temp_output_path)

    print(f"Saved enzyme embeddings for {len(all_enzy_embeddings)} unique proteins")
    print(f"Saved SMILES embeddings for {len(embeddings_smiles_dict)} unique molecules")
    print(f"Results saved to: {output_path}")

    # Print embedding dimensions
    if all_enzy_embeddings:
        sample_enzy_embed = next(iter(all_enzy_embeddings.values()))
        print(f"Enzyme embedding dimension: {sample_enzy_embed.shape}")

    if embeddings_smiles_dict:
        sample_smiles_embed = next(iter(embeddings_smiles_dict.values()))
        print(f"SMILES embedding dimension: {sample_smiles_embed.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate embeddings from a file.")
    parser.add_argument('--filename', type=str, required=True, help="Path to the input file.")
    parser.add_argument('--num_chunks', type=int, default=2, help="Number of chunks to split the data into")
    parser.add_argument('--model', type=str, required=True, help="FusionESP or ProSmith")
    args = parser.parse_args()
    main(args)