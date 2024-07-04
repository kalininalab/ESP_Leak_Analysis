import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix, precision_recall_curve, roc_curve, \
    accuracy_score, log_loss
from hyperopt import fmin, tpe, hp, Trials, space_eval
import logging
import os
from os.path import join
import wandb

# Setup logging
logging.basicConfig(filename='Hyperparameter_optimization_ESM1bts_and_ECFP_I1e_3S.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Add console handler to also display logs on the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

# Initialize wandb
wandb.init(project='ESP', entity='vahid-atabaigi')

# Load data
CURRENT_DIR = os.getcwd()
logging.info(f"Current directory: {CURRENT_DIR}")


def load_data(file_path):
    try:
        df = pd.read_pickle(file_path)
        df = df.loc[df["ESM1b_ts"] != ""]
        df = df.loc[df["type"] != "engqvist"]
        df = df.loc[df["GNN rep"] != ""]
        df = df.loc[df["ECFP"] != ""]
        df.reset_index(inplace=True, drop=True)
        return df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise


df_train = load_data(join(CURRENT_DIR, "..", "data", "3splits", "df_train_with_ESM1b_ts_GNN_I1e.pkl"))
df_test = load_data(join(CURRENT_DIR, "..", "data", "3splits", "df_test_with_ESM1b_ts_GNN_I1e.pkl"))
df_val = load_data(join(CURRENT_DIR, "..", "data", "3splits", "df_val_with_ESM1b_ts_GNN_I1e.pkl"))


def create_input_and_output_data(df):
    X = []
    y = []
    for ind in df.index:
        emb = df["ESM1b_ts"][ind]
        ecfp = np.array(list(df["ECFP"][ind])).astype(int)
        X.append(np.concatenate([ecfp, emb]))
        y.append(df["Binding"][ind])
    return np.array(X), np.array(y)


# Prepare input and output data for train, validation, and test sets
train_X, train_y = create_input_and_output_data(df_train)
test_X, test_y = create_input_and_output_data(df_test)
val_X, val_y = create_input_and_output_data(df_val)

# Define feature names
feature_names = ["ECFP_" + str(i) for i in range(1024)] + ["ESM1b_ts_" + str(i) for i in range(1280)]


def optimize_hyperparameters(param):
    num_round = int(param["num_rounds"])
    param["tree_method"] = "gpu_hist"
    param["sampling_method"] = "gradient_based"
    param['objective'] = 'binary:logistic'

    # Define weights for imbalance
    weights = np.array([param["weight"] if binding == 0 else 1.0 for binding in df_train["Binding"]])

    # Remove hyperparameters not needed for training
    del param["num_rounds"]
    del param["weight"]

    # Train XGBoost model
    dtrain = xgb.DMatrix(train_X, weight=weights, label=train_y, feature_names=feature_names)
    dval = xgb.DMatrix(val_X, label=val_y, feature_names=feature_names)
    bst = xgb.train(param, dtrain, num_round, verbose_eval=False, evals=[(dval, 'eval')])

    # Predict on validation set
    y_val_pred = np.round(bst.predict(dval))

    # Calculate metrics
    roc_auc = roc_auc_score(val_y, bst.predict(dval))
    logging.info(f"AUC-ROC: {roc_auc}")
    mcc = matthews_corrcoef(val_y, y_val_pred)

    # Log metrics to wandb
    wandb.log({"roc_auc": roc_auc, "mcc": mcc})

    # Log confusion matrix
    confusion_mat = confusion_matrix(val_y, y_val_pred)
    wandb.log({"confusion_matrix": confusion_mat.tolist()})

    # Log ROC curve
    fpr, tpr, _ = roc_curve(val_y, bst.predict(dval))
    wandb.log({"roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()}})

    # Log precision-recall curve
    precision, recall, _ = precision_recall_curve(val_y, bst.predict(dval))
    wandb.log({"precision_recall_curve": {"precision": precision.tolist(), "recall": recall.tolist()}})

    # Return negative MCC as we want to maximize it
    return -mcc


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

# Perform hyperparameter optimization with a high max_evals value
best_params = None
for i in range(1, 501):
    try:
        best = fmin(fn=optimize_hyperparameters,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=i,
                    trials=trials)

        logging.info(f"Iteration {i}")
        logging.info(f"Best -MCC so far: {trials.best_trial['result']['loss']}")
        logging.info(f"Best hyperparameters so far: {trials.argmin}")

        # Update best parameters
        best_params = space_eval(space, trials.argmin)
    except Exception as e:
        logging.error(f"Error during hyperparameter optimization: {e}")
        break

if best_params:
    # Train final model using best hyperparameters on training set
    weights_train = np.array([best_params["weight"] if binding == 0 else 1.0 for binding in df_train["Binding"]])
    dtrain_final = xgb.DMatrix(train_X, weight=weights_train, label=train_y, feature_names=feature_names)

    # Removing unnecessary params for final training
    best_params.pop("weight")
    best_params.pop("num_rounds")

    bst_final = xgb.train(best_params, dtrain_final, int(trials.argmin['num_rounds']), verbose_eval=1)

    # Evaluate final model on validation set
    dval = xgb.DMatrix(val_X, label=val_y, feature_names=feature_names)
    y_val_pred = np.round(bst_final.predict(dval))
    roc_auc_val = roc_auc_score(val_y, bst_final.predict(dval))
    mcc_val = matthews_corrcoef(val_y, y_val_pred)
    accuracy_val = accuracy_score(val_y, y_val_pred)
    loss_val = log_loss(val_y, bst_final.predict(dval))

    print("Final model evaluation on validation set:")
    print(f"ROC-AUC: {roc_auc_val:.4f}")
    print(f"MCC: {mcc_val:.4f}")
    print(f"Accuracy: {accuracy_val:.4f}")
    print(f"Loss: {loss_val:.4f}")

    # Save validation results
    np.save(join(CURRENT_DIR, "..", "data", "training_results_3S", "y_val_pred_xgboost_ESM1b_ts_ECFP_3S_I1e.npy"),
            bst_final.predict(dval))
    np.save(join(CURRENT_DIR, "..", "data", "training_results_3S", "y_val_true_xgboost_ESM1b_ts_ECFP_3S_I1e.npy"),
            val_y)
    np.save(join(CURRENT_DIR, "..", "data", "training_results_3S", "roc_auc_val_xgboost_ESM1b_ts_ECFP_3S_I1e.npy"),
            roc_auc_val)
    np.save(join(CURRENT_DIR, "..", "data", "training_results_3S", "mcc_val_xgboost_ESM1b_ts_ECFP_3S_I1e.npy"), mcc_val)
    np.save(join(CURRENT_DIR, "..", "data", "training_results_3S", "accuracy_val_xgboost_ESM1b_ts_ECFP_3S_I1e.npy"),
            accuracy_val)
    np.save(join(CURRENT_DIR, "..", "data", "training_results_3S", "loss_val_xgboost_ESM1b_ts_ECFP_3S_I1e.npy"),
            loss_val)

    # Evaluate final model on test set
    dtest = xgb.DMatrix(test_X, label=test_y, feature_names=feature_names)
    y_test_pred = np.round(bst_final.predict(dtest))
    roc_auc_test = roc_auc_score(test_y, bst_final.predict(dtest))
    mcc_test = matthews_corrcoef(test_y, y_test_pred)
    accuracy_test = accuracy_score(test_y, y_test_pred)
    loss_test = log_loss(test_y, bst_final.predict(dtest))

    print("Final model evaluation on test set:")
    print(f"ROC-AUC: {roc_auc_test:.4f}")
    print(f"MCC: {mcc_test:.4f}")
    print(f"Accuracy: {accuracy_test:.4f}")
    print(f"Loss: {loss_test:.4f}")

    # Save test results
    np.save(join(CURRENT_DIR, "..", "data", "training_results_3S", "y_test_pred_xgboost_ESM1b_ts_ECFP_3S_I1e.npy"),
            bst_final.predict(dtest))
    np.save(join(CURRENT_DIR, "..", "data", "training_results_3S", "y_test_true_xgboost_ESM1b_ts_ECFP_3S_I1e.npy"),
            test_y)
    np.save(join(CURRENT_DIR, "..", "data", "training_results_3S", "roc_auc_test_xgboost_ESM1b_ts_ECFP_3S_I1e.npy"),
            roc_auc_test)
    np.save(join(CURRENT_DIR, "..", "data", "training_results_3S", "mcc_test_xgboost_ESM1b_ts_ECFP_3S_I1e.npy"),
            mcc_test)
    np.save(join(CURRENT_DIR, "..", "data", "training_results_3S", "accuracy_test_xgboost_ESM1b_ts_ECFP_3S_I1e.npy"),
            accuracy_test)
    np.save(join(CURRENT_DIR, "..", "data", "training_results_3S", "loss_test_xgboost_ESM1b_ts_ECFP_3S_I1e.npy"),
            loss_test)

print("Finished without error")
