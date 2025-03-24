import pandas as pd
import numpy as np
import random
import pickle
import sys
import os
import logging
import wandb
import gc
from os.path import join
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from hyperopt import fmin, tpe, hp, Trials, rand, space_eval
import xgboost as xgb
import argparse
sys.path.append("./additional_code")
from additional_code.helper_functions import *
current_dir = os.getcwd()
warnings.filterwarnings("ignore")

"""
All codes in this script has been taken from ESP repository 
"""


def array_column_to_strings(df, column):
    df[column] = [str(list(df[column][ind])) for ind in df.index]
    return df


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


def split_dataframe(df, frac, CV_col):
    try:
        df.drop(columns=["level_0"], inplace=True)
    except:
        pass
    df.reset_index(inplace=True)

    train_indices = []
    test_indices = []
    ind = 0
    while len(train_indices) + len(test_indices) < len(df):
        if ind not in train_indices and ind not in test_indices:
            if ind % frac != 0:
                n_old = len(train_indices)
                train_indices.append(ind)
                train_indices = list(set(train_indices))
                while n_old != len(train_indices):
                    n_old = len(train_indices)
                    training_seqs = list(set(df[CV_col].loc[train_indices]))
                    train_indices = train_indices + (list(df.loc[df[CV_col].isin(training_seqs)].index))
                    train_indices = list(set(train_indices))

            else:
                n_old = len(test_indices)
                test_indices.append(ind)
                test_indices = list(set(test_indices))

                while n_old != len(test_indices):
                    n_old = len(test_indices)

                    testing_seqs = list(set(df[CV_col].loc[test_indices]))

                    test_indices = test_indices + (list(df.loc[df[CV_col].isin(testing_seqs)].index))
                    test_indices = list(set(test_indices))

        ind += 1
    return df.loc[train_indices], df.loc[test_indices]


def create_input_and_output_data(df, column):
    x = ()
    y = ()

    for ind in df.index:
        emb = df["ESM1b_ts"][ind]
        ecfp = np.array(list(df[column][ind])).astype(int)
        x = x + (np.concatenate([ecfp, emb]),)
        y = y + (df["Binding"][ind],)

    return x, y


def customized_cross_validation(train_set, CV_col,split_data,Data_Eng, save_indices=True):
    current_dir = os.getcwd()
    train_indices = [[], [], [], [], []]
    test_indices = [[], [], [], [], []]
    split_file_train = os.path.join(current_dir, "..", "data", "2splits", f"CV_train_indices_{split_data}{Data_Eng}_{CV_col}.pkl")
    split_file_val = os.path.join(current_dir, "..", "data", "2splits", f"CV_test_indices_{split_data}{Data_Eng}_{CV_col}.pkl")
    if save_indices:
        data_train2 = train_set.copy()
        data_train2 = array_column_to_strings(data_train2, column=CV_col)

        data_train2, df_fold = split_dataframe(df=data_train2, frac=5, CV_col=CV_col)
        indices_fold1 = list(df_fold["index"])
        print(len(data_train2), len(indices_fold1))  #

        data_train2, df_fold = split_dataframe(df=data_train2, frac=4, CV_col=CV_col)
        indices_fold2 = list(df_fold["index"])
        print(len(data_train2), len(indices_fold2))

        data_train2, df_fold = split_dataframe(df=data_train2, frac=3, CV_col=CV_col)
        indices_fold3 = list(df_fold["index"])
        print(len(data_train2), len(indices_fold3))

        data_train2, df_fold = split_dataframe(df=data_train2, frac=2, CV_col=CV_col)
        indices_fold4 = list(df_fold["index"])
        indices_fold5 = list(data_train2["index"])
        print(len(data_train2), len(indices_fold4))
        fold_indices = [indices_fold1, indices_fold2, indices_fold3, indices_fold4, indices_fold5]
        for i in range(5):
            for j in range(5):
                if i != j:
                    train_indices[i] = train_indices[i] + fold_indices[j]

            test_indices[i] = fold_indices[i]
    else:
        # Load previously saved indices
        with open(split_file_train, 'rb') as f:
            train_indices = pickle.load(f)
        with open(split_file_val, 'rb') as f:
            test_indices = pickle.load(f)

    return train_indices, test_indices


def cross_validation_neg_acc_gradient_boosting(param, train_indices, test_indices, train_x, train_y, df_train):
    num_round = param["num_rounds"]

    param["tree_method"] = "gpu_hist"
    param["device"] = "cuda"
    param["sampling_method"] = "gradient_based"
    param['objective'] = 'binary:logistic'

    weights = np.array([param["weight"] if binding == 0 else 1.0 for binding in df_train["Binding"]])
    del param["num_rounds"]
    del param["weight"]

    loss = []
    for i in range(5):
        train_index, test_index = train_indices[i], test_indices[i]
        dtrain = xgb.DMatrix(np.array(train_x[train_index]), weight=weights[train_index],
                             label=np.array(train_y[train_index]))
        dvalid = xgb.DMatrix(np.array(train_x[test_index]))
        bst = xgb.train(param, dtrain, int(num_round), verbose_eval=1)
        y_valid_pred = np.round(bst.predict(dvalid))
        validation_y = train_y[test_index]
        false_positive = 100 * (1 - np.mean(np.array(validation_y)[y_valid_pred == 1]))
        false_negative = 100 * (np.mean(np.array(validation_y)[y_valid_pred == 0]))
        logging.info(
            "False positive rate: " + str(false_positive) + "; False negative rate: " + str(false_negative))
        loss.append(2 * (false_negative ** 2) + false_positive ** 1.3)

    wandb.config.update(param, allow_val_change=True)
    wandb.log({"loss": np.mean(loss)})
    return np.mean(loss)



def main(args):
    current_dir = os.getcwd()
    split_data = args.split_data
    Data_Eng = f"_{args.Data_EnCof}" if args.Data_EnCof else ""
    column_name = args.column_name
    wandb.init(project='ESP', entity='vahid-atabaigi', name=f"ESM1b_ts_{column_name}_{split_data}{Data_Eng}_2S_SIP")
    column_cv = None
    if split_data in ["C1f", "I1f", "ESP", "ESPC2"]:
        column_cv = "ESM1b_ts"
    elif split_data in ["C1e", "I1e", "C2"] and column_name == "PreGNN":
        column_cv = "PreGNN"
    elif split_data in ["C1e", "I1e", "C2"] and column_name == "ECFP":
        column_cv = "ECFP"

    logging.basicConfig(filename=join(current_dir, "..", "data", "Reports", "hyperOp_report",
                                      f"HOP_ESM1bts_and_{column_name}_{split_data}{Data_Eng}_2S.log"),
                        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(console_handler)
    df_train = load_data(join(current_dir, "..", "data", "2splits", f"train_{split_data}{Data_Eng}_2S.pkl"), column=column_name)
    df_test = load_data(join(current_dir, "..", "data", "2splits", f"test_{split_data}{Data_Eng}_2S.pkl"), column=column_name)

    train_x, train_y = create_input_and_output_data(df=df_train, column=column_name)
    test_x, test_y = create_input_and_output_data(df=df_test, column=column_name)

    if column_name == "ECFP":
        feature_names = [column_name + "_" + str(i) for i in range(1024)] + ["ESM1b_ts_" + str(i) for i in range(1280)]
    else:
        feature_names = [column_name + "_" + str(i) for i in range(50)] + ["ESM1b_ts_" + str(i) for i in range(1280)]

    train_x = np.array(train_x)
    test_x = np.array(test_x)

    train_y = np.array(train_y)
    test_y = np.array(test_y)
    gc.collect()

    # Load or save CV indices once
    save_indices = not os.path.exists(
        os.path.join(current_dir, "..", "data", "2splits", f"CV_train_indices_{split_data}{Data_Eng}_{column_cv}.pkl"))
    train_indices, test_indices = customized_cross_validation(df_train, column_cv,split_data, Data_Eng, save_indices)
    # Defining search space for hyperparameter optimization
    space = {"learning_rate": hp.uniform("learning_rate", 0.01, 0.5),
             "max_depth": hp.choice("max_depth", [9, 10, 11, 12, 13]),
             "reg_lambda": hp.uniform("reg_lambda", 0, 5),
             "reg_alpha": hp.uniform("reg_alpha", 0, 5),
             "max_delta_step": hp.uniform("max_delta_step", 0, 5),
             "min_child_weight": hp.uniform("min_child_weight", 0.1, 15),
             "num_rounds": hp.uniform("num_rounds", 200, 400),
             "weight": hp.uniform("weight", 0.1, 0.33)}

    # Hyperparameter optimization function
    trials = Trials()
    for i in range(1, 2000):
        try:
            best = fmin(
                fn=lambda param: cross_validation_neg_acc_gradient_boosting(param, train_indices, test_indices, train_x,
                                                                            train_y, df_train),
                space=space,
                algo=rand.suggest,
                max_evals=i,
                trials=trials
            )
            logging.info(f"Iteration {i}")
            logging.info(f"Best loss so far: {trials.best_trial['result']['loss']}")
            logging.info(f"Best hyperparameters so far: {trials.argmin}")
            # Force garbage collection at each iteration
            gc.collect()
        except Exception as e:
            logging.error(f"Error during hyperparameter optimization: {e}")
            break
    best_params = space_eval(space, trials.argmin)
    num_round = best_params["num_rounds"]
    best_params["tree_method"] = "gpu_hist"
    best_params["device"] = "cuda"
    best_params["sampling_method"] = "gradient_based"
    best_params['objective'] = 'binary:logistic'
    weights = np.array([best_params["weight"] if binding == 0 else 1.0 for binding in df_train["Binding"]])
    del best_params["num_rounds"]
    del best_params["weight"]
    loss = []
    accuracy = []
    roc_auc = []
    for i in range(5):
        train_index, test_index = train_indices[i], test_indices[i]
        dtrain = xgb.DMatrix(np.array(train_x[train_index]), weight=weights[train_index],
                             label=np.array(train_y[train_index]))
        dvalid = xgb.DMatrix(np.array(train_x[test_index]))
        bst = xgb.train(best_params, dtrain, int(num_round), verbose_eval=1)
        y_valid_pred = np.round(bst.predict(dvalid))
        validation_y = train_y[test_index]
        false_positive = 100 * (1 - np.mean(np.array(validation_y)[y_valid_pred == 1]))
        false_negative = 100 * (np.mean(np.array(validation_y)[y_valid_pred == 0]))
        logging.info("False positive rate: " + str(false_positive) + "; False negative rate: " + str(false_negative))
        loss.append(2 * (false_negative ** 2) + false_positive ** 1.3)
        accuracy.append(np.mean(y_valid_pred == np.array(validation_y)))
        roc_auc.append(roc_auc_score(np.array(validation_y), bst.predict(dvalid)))
    print("Loss values: %s" % loss)
    print("Accuracies: %s" % accuracy)
    print("ROC-AUC scores: %s" % roc_auc)
    np.save(join(current_dir, "..", "data", "training_results_2S",
                 f"acc_CV_xgboost_ESM1b_ts_{column_name}_{split_data}{Data_Eng}_2S.npy"), np.array(accuracy))
    np.save(join(current_dir, "..", "data", "training_results_2S",
                 f"loss_CV_xgboost_ESM1b_ts_{column_name}_{split_data}{Data_Eng}_2S.npy"), np.array(loss))
    np.save(join(current_dir, "..", "data", "training_results_2S",
                 f"ROC_AUC_CV_xgboost_ESM1b_ts_{column_name}_{split_data}{Data_Eng}_2S.npy"), np.array(roc_auc))
    dtrain = xgb.DMatrix(np.array(train_x), weight=weights, label=np.array(train_y),
                         feature_names=feature_names)
    dtest = xgb.DMatrix(np.array(test_x), label=np.array(test_y),
                        feature_names=feature_names)

    bst = xgb.train(best_params, dtrain, int(num_round), verbose_eval=1)
    y_test_pred = np.round(bst.predict(dtest))
    acc_test = np.mean(y_test_pred == np.array(test_y))
    roc_auc = roc_auc_score(np.array(test_y), bst.predict(dtest))
    mcc = matthews_corrcoef(np.array(test_y), y_test_pred)

    print("Accuracy on test set: %s, ROC-AUC score for test set: %s, MCC: %s" % (acc_test, roc_auc, mcc))

    np.save(join(current_dir, "..", "data", "training_results_2S",
                 f"y_test_pred_xgboost_ESM1b_ts_{column_name}_{split_data}{Data_Eng}_2S.npy"), bst.predict(dtest))
    np.save(join(current_dir, "..", "data", "training_results_2S",
                 f"y_test_true_xgboost_ESM1b_ts_{column_name}_{split_data}{Data_Eng}_2S.npy"), test_y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End to end Hyperparameter tuning and model training for train:test splits")
    parser.add_argument('--split-data', type=str, required=True,
                        help="The split-data should be one of [C2,C1e, C1f, I1e, I1f,ESP, ESPC2] to load train,test "
                             "or validation sets for each split method")
    parser.add_argument('--column-name', type=str, required=True,
                        help="This argument selects the embedded vector for molecules to concatenate with the ESM1bts,"
                             " column name should be one of [ ECFP , PreGNN]")
    parser.add_argument('--Data-EnCof', default="", type=str, required=False,
                        help="The Dataframe without energy (NoEnCof) and its ground true D6964 ")
    args = parser.parse_args()
    main(args)
