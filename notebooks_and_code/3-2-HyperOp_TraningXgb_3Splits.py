import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix, precision_recall_curve, roc_curve, \
    accuracy_score, log_loss, auc
from hyperopt import fmin, tpe, hp, Trials, rand, space_eval
import logging
from os.path import join
import wandb
import os
import re
import sys
import collections
import warnings
import argparse
import os.path
import gc
from colorama import init, Fore, Style
import torch


sys.path.append("./additional_code")
from additional_code.helper_functions import *
from additional_code.negative_data_generator import *

warnings.filterwarnings("ignore")


def main(args):
    wandb.init(project='ESP', entity='vahid-atabaigi')
    current_dir = os.getcwd()
    split_data = args.split_data
    column_name = args.column_name
    report_path =join(current_dir, "..", "data", "Reports", f"hyperOp_report")
    report_file = f"HOP_ESM1bts_and_{column_name}_{split_data}_3S.log"
    full_report_file = os.path.join(report_path, report_file)
    os.makedirs(os.path.dirname(full_report_file), exist_ok=True)
    if os.path.exists(full_report_file):
        os.remove(full_report_file)
    logging.basicConfig(filename=full_report_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(console_handler)

    def load_data(file_path, column):
        try:
            df = pd.read_pickle(file_path)
            df = df[df["ESM1b_ts"].apply(lambda x: len(x) > 0)]
            df = df.loc[df["type"] != "engqvist"]
            df = df[df[column].apply(lambda x: len(x) > 0)]
            df.reset_index(inplace=True, drop=True)
            return df
        except Exception as e:
            logging.error(f"Error loading data from {file_path}: {e}")
            raise

    df_train = load_data(join(current_dir, "..", "data", f"3splits", f"train_{split_data}_3S.pkl"),
                         column=column_name)
    df_test = load_data(join(current_dir, "..", "data", f"3splits", f"test_{split_data}_3S.pkl"),
                        column=column_name)
    df_val = load_data(join(current_dir, "..", "data", f"3splits", f"val_{split_data}_3S.pkl"),
                       column=column_name)

    def create_input_and_output_data(df):
        x = []
        y = []
        for ind in df.index:
            emb = df["ESM1b_ts"][ind]
            ecfp = np.array(list(df[column_name][ind])).astype(int)
            x.append(np.concatenate([ecfp, emb]))
            y.append(df["Binding"][ind])
        return np.array(x), np.array(y)

    # Prepare input and output data for train, validation, and test sets
    train_x, train_y = create_input_and_output_data(df_train)
    test_X, test_y = create_input_and_output_data(df_test)
    val_X, val_y = create_input_and_output_data(df_val)

    if column_name == "ECFP":
        feature_names = [column_name + "_" + str(i) for i in range(1024)] + ["ESM1b_ts_" + str(i) for i in range(1280)]
    else:
        feature_names = [column_name + "_" + str(i) for i in range(50)] + ["ESM1b_ts_" + str(i) for i in range(1280)]

    def optimize_hyperparameters(param):
        num_round = int(param["num_rounds"])
        param["tree_method"] = "gpu_hist"
        param["device"] = "cuda"
        #param["tree_method"] = "hist"
        param["sampling_method"] = "gradient_based"
        param['objective'] = 'binary:logistic'
        # Define weights for imbalance
        weights = np.array([param["weight"] if binding == 0 else 1.0 for binding in df_train["Binding"]])
        # Remove hyperparameters not needed for training
        del param["num_rounds"]
        del param["weight"]
        # Train XGBoost model
        dtrain = xgb.DMatrix(train_x, weight=weights, label=train_y, feature_names=feature_names)
        dval = xgb.DMatrix(val_X, label=val_y, feature_names=feature_names)
        bst = xgb.train(param, dtrain, num_round, verbose_eval=False, evals=[(dval, 'eval')])
        # Predict on validation set
        y_val_pred = np.round(bst.predict(dval))
        # Calculate loss
        false_positive = 100 * (1 - np.mean(np.array(val_y)[y_val_pred == 1]))
        false_negative = 100 * (np.mean(np.array(val_y)[y_val_pred == 0]))
        logging.info(
            "False positive rate: " + str(false_positive) + "; False negative rate: " + str(false_negative))
        loss = 2 * (false_negative ** 2) + false_positive ** 1.3
        wandb.config.update(param, allow_val_change=True)
        wandb.log({"loss": np.mean(loss)})
        torch.cuda.empty_cache()
        del dtrain, dval, bst
        gc.collect()
        return np.mean(loss)

    # Define search space for hyperparameter optimization
    space = {
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.5),
        "max_depth": hp.choice("max_depth", [9, 10, 11, 12, 13]),
        "reg_lambda": hp.uniform("reg_lambda", 0, 5),
        "reg_alpha": hp.uniform("reg_alpha", 0, 5),
        "max_delta_step": hp.uniform("max_delta_step", 0, 5),
        "min_child_weight": hp.uniform("min_child_weight", 0.1, 15),
        "num_rounds": hp.quniform("num_rounds", 200, 400, 1),
        "weight": hp.uniform("weight", 0.1, 0.33)
    }

    # Perform hyperparameter optimization
    trials = Trials()
    for i in range(1, 2000):
        try:
            best = fmin(fn=optimize_hyperparameters, space=space, algo=rand.suggest, max_evals=i,
                        trials=trials)

            logging.info(f"Iteration {i}")
            logging.info(f"Best loss so far: {trials.best_trial['result']['loss']}")
            logging.info(f"Best hyperparameters so far: {trials.argmin}")
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            logging.error(f"Error during hyperparameter optimization: {e}")
            break

    # Get best hyperparameters
    best_params = space_eval(space, trials.argmin)

    # Train final model using the best hyperparameters on training set
    weights_train = np.array([best_params["weight"] if binding == 0 else 1.0 for binding in df_train["Binding"]])
    dtrain_final = xgb.DMatrix(train_x, weight=weights_train, label=train_y, feature_names=feature_names)

    # Removing unnecessary params for final training
    best_params.pop("weight")
    best_params.pop("num_rounds")

    bst_final = xgb.train(best_params, dtrain_final, int(trials.argmin['num_rounds']), verbose_eval=1)
    # Evaluate final model on validation set
    dval = xgb.DMatrix(val_X, label=val_y, feature_names=feature_names)
    y_val_pred = np.round(bst_final.predict(dval))
    roc_auc_val = roc_auc_score(val_y, bst_final.predict(dval))
    mcc_val = matthews_corrcoef(val_y, y_val_pred)

    # Calculate accuracy and loss for validation set
    accuracy_val = accuracy_score(val_y, y_val_pred)
    loss_val = log_loss(val_y, bst_final.predict(dval))

    print("Final model evaluation on validation set:")
    print("ROC-AUC: %.4f" % roc_auc_val)
    print("MCC: %.4f" % mcc_val)
    print("Accuracy: %.4f" % accuracy_val)
    print("Loss: %.4f" % loss_val)

    # Save validation results
    np.save(join(current_dir, "..", "data", f"training_results_3S",
                 f"y_val_pred_xgboost_ESM1b_ts_{column_name}_{split_data}_3S.npy"),
            bst_final.predict(dval))
    np.save(join(current_dir, "..", "data", f"training_results_3S",
                 f"y_val_true_xgboost_ESM1b_ts_{column_name}_{split_data}_3S.npy"), val_y)
    np.save(join(current_dir, "..", "data", f"training_results_3S",
                 f"roc_auc_val_xgboost_ESM1b_ts_{column_name}_{split_data}_3S.npy"), roc_auc_val)
    np.save(join(current_dir, "..", "data", f"training_results_3S",
                 f"mcc_val_xgboost_ESM1b_ts_{column_name}_{split_data}_3S.npy"), mcc_val)
    np.save(join(current_dir, "..", "data", f"training_results_3S",
                 f"accuracy_val_xgboost_ESM1b_ts_{column_name}_{split_data}_3S.npy"), accuracy_val)
    np.save(join(current_dir, "..", "data", f"training_results_3S",
                 f"loss_val_xgboost_ESM1b_ts_{column_name}_{split_data}_3S.npy"), loss_val)

    # Evaluate final model on test set
    dtest = xgb.DMatrix(test_X, label=test_y, feature_names=feature_names)
    y_test_pred = np.round(bst_final.predict(dtest))
    roc_auc_test = roc_auc_score(test_y, bst_final.predict(dtest))
    mcc_test = matthews_corrcoef(test_y, y_test_pred)

    # Calculate accuracy and loss for test set
    accuracy_test = accuracy_score(test_y, y_test_pred)
    loss_test = log_loss(test_y, bst_final.predict(dtest))

    print("Final model evaluation on test set:")
    print("ROC-AUC: %.4f" % roc_auc_test)
    print("MCC: %.4f" % mcc_test)
    print("Accuracy: %.4f" % accuracy_test)
    print("Loss: %.4f" % loss_test)

    # Save test results
    np.save(join(current_dir, "..", "data", f"training_results_3S",
                 f"y_test_pred_xgboost_ESM1b_ts_{column_name}_{split_data}_3S.npy"),
            bst_final.predict(dtest))
    np.save(join(current_dir, "..", "data", f"training_results_3S",
                 f"y_test_true_xgboost_ESM1b_ts_{column_name}_{split_data}_3S.npy"), test_y)
    np.save(join(current_dir, "..", "data", f"training_results_3S",
                 f"roc_auc_test_xgboost_ESM1b_ts_{column_name}_{split_data}_3S.npy"), roc_auc_test)
    np.save(join(current_dir, "..", "data", f"training_results_3S",
                 f"mcc_test_xgboost_ESM1b_ts_{column_name}_{split_data}_3S.npy"), mcc_test)
    np.save(join(current_dir, "..", "data", f"training_results_3S",
                 f"accuracy_test_xgboost_ESM1b_ts_{column_name}_{split_data}_3S.npy"), accuracy_test)
    np.save(join(current_dir, "..", "data", f"training_results_3S",
                 f"loss_test_xgboost_ESM1b_ts_{column_name}_{split_data}_3S.npy"), loss_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the data preprocessing and negative sample generation script.")
    parser.add_argument('--split-data', type=str, required=True,
                        help="The split-data should be one of [C2,C1e, C1f, I1e, I1f,ESP, ESPC2] to load train,test "
                             "or validation sets for each split method")
    parser.add_argument('--column-name', type=str, required=True,
                        help="This argument selects the embedded vector for molecules to concatenate with the ESM1bts,"
                             " column name should be one of [ ECFP , PreGNN]")
    args = parser.parse_args()
    main(args)
