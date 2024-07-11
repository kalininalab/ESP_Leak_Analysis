from sklearn.metrics import matthews_corrcoef
import json
import pandas as pd
import numpy as np
import random
import pickle
import sys
import os
import logging
import wandb
from os.path import join
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix, precision_recall_curve, roc_curve
from hyperopt import fmin, tpe, hp, Trials, rand, space_eval
import xgboost as xgb
sys.path.append("./additional_code")
from additional_code.helper_functions import *
from additional_code.negative_data_generator import *
warnings.filterwarnings("ignore")


"""
All codes in this script has been taken from ESP repository 
"""

def main(args):
    wandb.init(project='SIP', entity='vahid-atabaigi')
    CURRENT_DIR = os.getcwd()
    splitted_data = args.splitted_data
    Data_suffix = f"_{args.Data_suffix}" if args.Data_suffix else ""
    column_name = args.column_name
    column_CV = "ESM1b_ts"
    # if splitted_data in ["C1f", "I1f"]:
    #     column_CV = "ESM1b_ts"
    # elif splitted_data in ["C1e", "I1e", "C2"]:
    #     column_CV = "SMILES"

    logging.basicConfig(filename=join(CURRENT_DIR, "..", "data", "Reports", "hyperOp_report",
                                      f"HOP_ESM1bts_and_{column_name}_{splitted_data}{Data_suffix}_2S.log"),
                        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(console_handler)

    def array_column_to_strings(df, column):
        df[column] = [str(list(df[column][ind])) for ind in df.index]
        return (df)

    def load_data(file_path,column):
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
    df_train = load_data(join(CURRENT_DIR, "..", "data", "2splits", f"train_{splitted_data}{Data_suffix}_2S.pkl"),column=column_name)

    df_test = load_data(join(CURRENT_DIR, "..", "data", "2splits", f"test_{splitted_data}{Data_suffix}_2S.pkl"),column=column_name)

    def split_dataframe(df, frac):
        df1 = pd.DataFrame(columns=list(df.columns))
        df2 = pd.DataFrame(columns=list(df.columns))
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

                        training_seqs = list(set(df[column_CV].loc[train_indices]))

                        train_indices = train_indices + (list(df.loc[df[column_CV].isin(training_seqs)].index))
                        train_indices = list(set(train_indices))

                else:
                    n_old = len(test_indices)
                    test_indices.append(ind)
                    test_indices = list(set(test_indices))

                    while n_old != len(test_indices):
                        n_old = len(test_indices)

                        testing_seqs = list(set(df[column_CV].loc[test_indices]))

                        test_indices = test_indices + (list(df.loc[df[column_CV].isin(testing_seqs)].index))
                        test_indices = list(set(test_indices))

            ind += 1
        return (df.loc[train_indices], df.loc[test_indices])

    data_train2 = df_train.copy()
    data_train2 = array_column_to_strings(data_train2, column=column_CV)

    data_train2, df_fold = split_dataframe(df=data_train2, frac=5)
    indices_fold1 = list(df_fold["index"])
    print(len(data_train2), len(indices_fold1))  #

    data_train2, df_fold = split_dataframe(df=data_train2, frac=4)
    indices_fold2 = list(df_fold["index"])
    print(len(data_train2), len(indices_fold2))

    data_train2, df_fold = split_dataframe(df=data_train2, frac=3)
    indices_fold3 = list(df_fold["index"])
    print(len(data_train2), len(indices_fold3))

    data_train2, df_fold = split_dataframe(df=data_train2, frac=2)
    indices_fold4 = list(df_fold["index"])
    indices_fold5 = list(data_train2["index"])
    print(len(data_train2), len(indices_fold4))

    fold_indices = [indices_fold1, indices_fold2, indices_fold3, indices_fold4, indices_fold5]

    train_indices = [[], [], [], [], []]
    test_indices = [[], [], [], [], []]

    for i in range(5):
        for j in range(5):
            if i != j:
                train_indices[i] = train_indices[i] + fold_indices[j]

        test_indices[i] = fold_indices[i]

    # Save indices using pickle
    with open(join(CURRENT_DIR, "..", "data", "2splits", f"CV_train_indices_{splitted_data}{Data_suffix}.pkl"),
              'wb') as f:
        pickle.dump(train_indices, f)

    with open(join(CURRENT_DIR, "..", "data", "2splits", f"CV_test_indices_{splitted_data}{Data_suffix}.pkl"),
              'wb') as f:
        pickle.dump(test_indices, f)

    # Load indices using pickle
    with open(join(CURRENT_DIR, "..", "data", "2splits", f"CV_train_indices_{splitted_data}{Data_suffix}.pkl"),
              'rb') as f:
        train_indices = pickle.load(f)

    with open(join(CURRENT_DIR, "..", "data", "2splits", f"CV_test_indices_{splitted_data}{Data_suffix}.pkl"),
              'rb') as f:
        test_indices = pickle.load(f)

    def create_input_and_output_data(df, column):
        X = ();
        y = ();

        for ind in df.index:
            emb = df["ESM1b_ts"][ind]
            ecfp = np.array(list(df[column][ind])).astype(int)
            X = X + (np.concatenate([ecfp, emb]),);
            y = y + (df["Binding"][ind],);

        return (X, y)

    train_X, train_y = create_input_and_output_data(df=df_train, column=column_name)
    test_X, test_y = create_input_and_output_data(df=df_test, column=column_name)

    if column_name=="ECFP":
        feature_names = [column_name+"_" + str(i) for i in range(1024)] + ["ESM1b_ts_" + str(i) for i in range(1280)]
    else:
        feature_names = [column_name + "_" + str(i) for i in range(50)] + ["ESM1b_ts_" + str(i) for i in range(1280)]

    train_X = np.array(train_X)
    test_X = np.array(test_X)

    train_y = np.array(train_y)
    test_y = np.array(test_y)

    def cross_validation_neg_acc_gradient_boosting(param):
        num_round = param["num_rounds"]
        param["tree_method"] = "gpu_hist"
        param["sampling_method"] = "gradient_based"
        param['objective'] = 'binary:logistic'
        weights = np.array([param["weight"] if binding == 0 else 1.0 for binding in df_train["Binding"]])
        del param["num_rounds"]
        del param["weight"]
        loss = []
        for i in range(5):
            train_index, test_index = train_indices[i], test_indices[i]
            dtrain = xgb.DMatrix(np.array(train_X[train_index]), weight=weights[train_index],
                                 label=np.array(train_y[train_index]))
            dvalid = xgb.DMatrix(np.array(train_X[test_index]))
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
            best = fmin(fn=cross_validation_neg_acc_gradient_boosting, space=space,
                        algo=rand.suggest, max_evals=i, trials=trials)
            logging.info(f"Iteration {i}")
            logging.info(f"Best loss so far: {trials.best_trial['result']['loss']}")
            logging.info(f"Best hyperparameters so far: {trials.argmin}")
        except Exception as e:
            logging.error(f"Error during hyperparameter optimization: {e}")
            break

    best_params = space_eval(space, trials.argmin)
    num_round = best_params["num_rounds"]
    best_params["tree_method"] = "hist"
    best_params["device"] = "cuda"
    best_params["sampling_method"] = "gradient_based"
    best_params['objective'] = 'binary:logistic'
    weights = np.array([best_params["weight"] if binding == 0 else 1.0 for binding in df_train["Binding"]])

    del best_params["num_rounds"]
    del best_params["weight"]

    loss = []
    accuracy = []
    ROC_AUC = []

    for i in range(5):
        train_index, test_index = train_indices[i], test_indices[i]
        dtrain = xgb.DMatrix(np.array(train_X[train_index]), weight=weights[train_index],
                             label=np.array(train_y[train_index]))
        dvalid = xgb.DMatrix(np.array(train_X[test_index]))
        bst = xgb.train(best_params, dtrain, int(num_round), verbose_eval=1)
        y_valid_pred = np.round(bst.predict(dvalid))
        validation_y = train_y[test_index]

        # Calculate loss
        false_positive = 100 * (1 - np.mean(np.array(validation_y)[y_valid_pred == 1]))
        false_negative = 100 * (np.mean(np.array(validation_y)[y_valid_pred == 0]))
        logging.info("False positive rate: " + str(false_positive) + "; False negative rate: " + str(false_negative))
        loss.append(2 * (false_negative ** 2) + false_positive ** 1.3)
        # Calculate accuracy
        accuracy.append(np.mean(y_valid_pred == np.array(validation_y)))
        # Calculate ROC-AUC score
        ROC_AUC.append(roc_auc_score(np.array(validation_y), bst.predict(dvalid)))

    print("Loss values: %s" % loss)
    print("Accuracies: %s" % accuracy)
    print("ROC-AUC scores: %s" % ROC_AUC)

    np.save(join(CURRENT_DIR, "..", "data", "training_results_2S",
                 f"acc_CV_xgboost_ESM1b_ts_{column_name}_{splitted_data}{Data_suffix}_2S.npy"), np.array(accuracy))
    np.save(join(CURRENT_DIR, "..", "data", "training_results_2S",
                 f"loss_CV_xgboost_ESM1b_ts_{column_name}_{splitted_data}{Data_suffix}_2S.npy"), np.array(loss))
    np.save(join(CURRENT_DIR, "..", "data", "training_results_2S",
                 f"ROC_AUC_CV_xgboost_ESM1b_ts_{column_name}_{splitted_data}{Data_suffix}_2S.npy"), np.array(ROC_AUC))

    dtrain = xgb.DMatrix(np.array(train_X), weight=weights, label=np.array(train_y),
                         feature_names=feature_names)
    dtest = xgb.DMatrix(np.array(test_X), label=np.array(test_y),
                        feature_names=feature_names)

    bst = xgb.train(best_params, dtrain, int(num_round), verbose_eval=1)
    y_test_pred = np.round(bst.predict(dtest))
    acc_test = np.mean(y_test_pred == np.array(test_y))
    roc_auc = roc_auc_score(np.array(test_y), bst.predict(dtest))
    mcc = matthews_corrcoef(np.array(test_y), y_test_pred)

    print("Accuracy on test set: %s, ROC-AUC score for test set: %s, MCC: %s" % (acc_test, roc_auc, mcc))

    np.save(join(CURRENT_DIR, "..", "data", "training_results_2S",
                 f"y_test_pred_xgboost_ESM1b_ts_{column_name}_{splitted_data}{Data_suffix}_2S.npy"), bst.predict(dtest))
    np.save(join(CURRENT_DIR, "..", "data", "training_results_2S",
                 f"y_test_true_xgboost_ESM1b_ts_{column_name}_{splitted_data}{Data_suffix}_2S.npy"), test_y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End to end Hyperparameter tuning and model training for train:test splits")
    parser.add_argument('--splitted-data', type=str, required=True,
                        help="The splitted-data should be one of [C2,C1e, C1f, I1e, I1f, ESPC1e, ESPC2]")
    parser.add_argument('--column-name', type=str, required=True,
                        help="The column name should be one of [ ECFP , PreGNN]")
    parser.add_argument('--Data-suffix', default="", type=str, required=False,
                        help="The Dataframe suffix name should be one of [ NoATP ,D3408] ")
    args = parser.parse_args()
    main(args)
