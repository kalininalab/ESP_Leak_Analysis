from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from FusionESP_model import Contrastive_learning_layer

import warnings
from tqdm import tqdm
import os
from pathlib import Path
import argparse
import sys
import os
from os.path import join

current_dir = os.getcwd()


def load_embeddings_and_create_datasets(split_method, enzy_embeddings_file, smiles_embeddings_file, batch_size):
    """
    Load dataframes and add embeddings as new columns
    """
    # Load dataframes
    train_df = pd.read_pickle(join(current_dir, "..", "data", "3splits", f"train_{split_method}_3S.pkl"))
    val_df = pd.read_pickle(join(current_dir, "..", "data", "3splits", f"val_{split_method}_3S.pkl"))
    test_df = pd.read_pickle(join(current_dir, "..", "data", "3splits", f"test_{split_method}_3S.pkl"))

    print(f"Original dataset sizes - Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

    # Load embedding dictionaries
    enzy_embeddings_dict = torch.load(enzy_embeddings_file, weights_only=False)
    smiles_embeddings_dict = torch.load(smiles_embeddings_file, weights_only=False)

    print(f"Enzyme embeddings: {len(enzy_embeddings_dict)} unique proteins")
    print(f"SMILES embeddings: {len(smiles_embeddings_dict)} unique molecules")

    # Function to add embeddings to dataframe
    def add_embeddings_to_df(df, enzy_dict, smiles_dict):
        # Add enzyme embeddings
        df['ESM_t36'] = df['Uniprot ID'].map(enzy_dict)
        # Add SMILES embeddings
        df['molformer'] = df['molecule ID'].map(smiles_dict)

        # Check for missing embeddings
        missing_enzy = df['ESM_t36'].isna().sum()
        missing_smiles = df['molformer'].isna().sum()

        if missing_enzy > 0:
            print(f"Warning: {missing_enzy} samples missing enzyme embeddings")
        if missing_smiles > 0:
            print(f"Warning: {missing_smiles} samples missing SMILES embeddings")

        # Remove rows with missing embeddings
        df = df.dropna(subset=['ESM_t36', 'molformer']).reset_index(drop=True)
        return df

    # Add embeddings to all dataframes
    train_df = add_embeddings_to_df(train_df, enzy_embeddings_dict, smiles_embeddings_dict)
    val_df = add_embeddings_to_df(val_df, enzy_embeddings_dict, smiles_embeddings_dict)
    test_df = add_embeddings_to_df(test_df, enzy_embeddings_dict, smiles_embeddings_dict)

    print(f"After embedding mapping - Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

    # Convert to tensors
    def df_to_tensors(df):
        enzy_tensors = torch.stack(df['ESM_t36'].tolist())
        smiles_tensors = torch.stack(df['molformer'].tolist())
        labels = torch.tensor(df['Binding'].values, dtype=torch.float32).unsqueeze(1)
        return enzy_tensors, smiles_tensors, labels

    train_enzy, train_smiles, train_labels = df_to_tensors(train_df)
    val_enzy, val_smiles, val_labels = df_to_tensors(val_df)
    test_enzy, test_smiles, test_labels = df_to_tensors(test_df)

    print(f"Final tensor shapes - Train: {train_enzy.shape}, {train_smiles.shape}, {train_labels.shape}")

    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_enzy, train_smiles, train_labels)
    val_dataset = TensorDataset(val_enzy, val_smiles, val_labels)
    test_dataset = TensorDataset(test_enzy, test_smiles, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def run_validation(model, val_loader, loss_fn, device):
    model.eval()
    loss_sum = 0
    num_batch = len(val_loader)
    total_y_true = []
    total_y_pred = []
    total_y_prob = []
    for ESP_val_df_enzy, ESP_val_df_smiles, y_val in val_loader:
        ESP_val_df_enzy = ESP_val_df_enzy.to(device)
        ESP_val_df_smiles = ESP_val_df_smiles.to(device)
        y_val = y_val.squeeze(1).to(device)

        refined_enzy_embed, refined_smiles_embed = model(ESP_val_df_enzy, ESP_val_df_smiles)
        cos_sim = torch.nn.functional.cosine_similarity(refined_enzy_embed, refined_smiles_embed, dim=1)
        loss = loss_fn(cos_sim, y_val).detach().cpu().numpy()
        loss_sum = loss_sum + loss
        y_pred = (cos_sim > 0.5).float().cpu().numpy()
        total_y_true.append(y_val.cpu().numpy())
        total_y_pred.append(y_pred)
        total_y_prob.append(cos_sim.detach().cpu().numpy())

    loss_sum = loss_sum / num_batch

    arrange_y_true = np.concatenate(total_y_true, axis=0)
    arrange_y_pred = np.concatenate(total_y_pred, axis=0)
    arrange_y_prob = np.concatenate(total_y_prob, axis=0)
    tn, fp, fn, tp = confusion_matrix(arrange_y_true, arrange_y_pred).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    bacc = (sensitivity + specificity) / 2
    MCC = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    AUC = roc_auc_score(arrange_y_true, arrange_y_prob)
    f1 = 2 * precision * recall / (precision + recall)

    print("loss_sum= ", loss_sum, "ACC= ", acc, "bacc= ", bacc, "precision= ", precision,
          "specificity= ", specificity, "sensitivity= ", sensitivity, "recall= ", recall,
          "MCC= ", MCC, "AUC= ", AUC, "f1= ", f1)
    return loss_sum, acc, bacc, arrange_y_true, arrange_y_prob


def train():
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if device == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif device == 'mps':
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
    device = torch.device(device)

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Train model with embeddings.")
    parser.add_argument('--split_method', type=str, required=True, help="Path to train dataframe file.")
    parser.add_argument('--enzy_embeddings', type=str, required=True, help="Path to enzyme embeddings file.")
    parser.add_argument('--smiles_embeddings', type=str, required=True, help="Path to SMILES embeddings file.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size during training")
    parser.add_argument('--learning_rate', type=float, default=1e-03, help="Learning rate for training")

    args = parser.parse_args()

    # Get data loaders
    train_loader, val_loader, test_loader = load_embeddings_and_create_datasets(
        args.split_method,
        args.enzy_embeddings, args.smiles_embeddings,
        args.batch_size
    )

    # Design the model, optimizer and loss function
    model = Contrastive_learning_layer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss().to(device)

    initial_epoch = 0
    best_epoch = -1
    best_accuracy = 0.5

    for epoch in range(initial_epoch, 500):
        torch.cuda.empty_cache()
        model.train()
        with tqdm(train_loader, desc='Processing', unit="batch") as tepoch:
            for ESP_train_df_enzy, ESP_train_df_smiles, y_train in tepoch:
                model.train()
                tepoch.set_description(f"Epoch {epoch}")
                ESP_train_df_enzy = ESP_train_df_enzy.to(device)
                ESP_train_df_smiles = ESP_train_df_smiles.to(device)
                y_train = y_train.squeeze(1).to(device)

                refined_enzy_embed, refined_smiles_embed = model(ESP_train_df_enzy, ESP_train_df_smiles)
                cosine_sim = torch.nn.functional.cosine_similarity(refined_enzy_embed, refined_smiles_embed, dim=1)
                loss = loss_fn(cosine_sim, y_train)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                tepoch.set_postfix(train_loss=loss.item())

            loss_sum_val, acc_val, bacc_val,_,_ = run_validation(model, val_loader, loss_fn, device)
            print('Epoch: %d / %d, ############## the best accuracy in val  %.4f at Epoch: %d  ##############'
                  % (epoch, 500, 100 * best_accuracy, best_epoch))
            print('Performance in Val: Loss: (%.4f); Accuracy (%.2f)' % (loss_sum_val, 100 * acc_val))
            mode_path = join(current_dir, "..", "data", "trained_model")
            os.makedirs(mode_path, exist_ok=True)
            if acc_val > best_accuracy:
                best_accuracy = acc_val
                best_epoch = epoch
                torch.save(model, join(mode_path,f"FusionESP_model_{args.split_method}.pt"))

    # Load and test the best model
    load_path = join(mode_path, f"FusionESP_model_{args.split_method}.pt")
    model_test = torch.load(load_path, weights_only=False)
    print('Model performance in test dataset \n')
    _, _, _, arrange_y_true, arrange_y_prob=run_validation(model_test, test_loader, loss_fn, device)
    results_path = join(current_dir, "..", "data", "training_results_3S")
    os.makedirs(results_path, exist_ok=True)
    np.save(join(results_path, f"y_test_pred_FusionESP_{args.split_method}.npy"), arrange_y_prob)
    np.save(join(results_path, f"y_test_true_FusionESP_{args.split_method}.npy"), arrange_y_true)


if __name__ == '__main__':
    train()