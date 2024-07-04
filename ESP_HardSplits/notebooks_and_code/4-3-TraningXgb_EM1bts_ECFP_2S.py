import pandas as pd
import numpy as np
import random
import pickle
import sys
import os
import logging
from os.path import join
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from hyperopt import fmin, tpe, hp, Trials, rand
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef
wandb.init(project='ESP', entity='vahid-atabaigi')
sys.path.append('./additional_code')
CURRENT_DIR = os.getcwd()
print(CURRENT_DIR)


df_train = pd.read_pickle(join(CURRENT_DIR, ".." ,"data","splits", "df_train_with_ESM1b_ts_GNN.pkl"))
df_train = df_train.loc[df_train["ESM1b"] != ""]
df_train = df_train.loc[df_train["type"] != "engqvist"]
df_train = df_train.loc[df_train["GNN rep"] != ""]
df_train.reset_index(inplace = True, drop = True)

df_test  = pd.read_pickle(join(CURRENT_DIR, ".." ,"data","splits", "df_test_with_ESM1b_ts_GNN.pkl"))
df_test = df_test.loc[df_test["ESM1b"] != ""]
df_test = df_test.loc[df_test["type"] != "engqvist"]
df_test = df_test.loc[df_test["GNN rep"] != ""]
df_test.reset_index(inplace = True, drop = True)

train_indices = list(np.load(join(CURRENT_DIR, ".." ,"data","splits", "CV_train_indices.npy"),  allow_pickle=True))
test_indices = list(np.load(join(CURRENT_DIR, ".." ,"data","splits", "CV_test_indices.npy"),  allow_pickle=True))


def create_input_and_output_data(df):
    X = ();
    y = ();
    
    for ind in df.index:
        emb = df["ESM1b_ts"][ind]
        ecfp = np.array(list(df["ECFP"][ind])).astype(int)
                
        X = X +(np.concatenate([ecfp, emb]), );
        y = y + (df["Binding"][ind], );

    return(X,y)

train_X, train_y =  create_input_and_output_data(df = df_train)
test_X, test_y =  create_input_and_output_data(df = df_test)


feature_names =  ["ECFP_" + str(i) for i in range(1024)]
feature_names = feature_names + ["ESM1b_ts_" + str(i) for i in range(1280)]

train_X = np.array(train_X)
test_X  = np.array(test_X)

train_y = np.array(train_y)
test_y  = np.array(test_y)

param = {'learning_rate': 0.31553117247348733,
         'max_delta_step': 1.7726044219753656,
         'max_depth': 10,
         'min_child_weight': 1.3845040588450772,
         'num_rounds': 342.68325188584106,
         'reg_alpha': 0.531395259755843,
         'reg_lambda': 3.744980563764689,
         'weight': 0.26187490421514203}

num_round = param["num_rounds"]
param["tree_method"] = "gpu_hist"
param["sampling_method"] = "gradient_based"
param['objective'] = 'binary:logistic'
weights = np.array([param["weight"] if binding == 0 else 1.0 for binding in df_train["Binding"]])

del param["num_rounds"]
del param["weight"]

loss = []
accuracy = []
ROC_AUC = []

for i in range(5):
    train_index, test_index  = train_indices[i], test_indices[i]
    dtrain = xgb.DMatrix(np.array(train_X[train_index]), weight = weights[train_index],
                     label = np.array(train_y[train_index]))
    dvalid = xgb.DMatrix(np.array(train_X[test_index]))
    bst = xgb.train(param,  dtrain, int(num_round), verbose_eval=1)
    y_valid_pred = np.round(bst.predict(dvalid))
    validation_y = train_y[test_index]

    #calculate loss:
    false_positive = 100*(1-np.mean(np.array(validation_y)[y_valid_pred == 1]))
    false_negative = 100*(np.mean(np.array(validation_y)[y_valid_pred == 0]))
    logging.info("False positive rate: " + str(false_positive)+ "; False negative rate: " + str(false_negative))
    loss.append(2*(false_negative**2) + false_positive**1.3)
    #calculate accuracy:
    accuracy.append(np.mean(y_valid_pred == np.array(validation_y)))
    #calculate ROC-AUC score:
    ROC_AUC.append(roc_auc_score(np.array(validation_y), bst.predict(dvalid)))
    wandb.log({"Test Fold_ESM1bts_ECFP": i, "Loss_ESM1bts_ECFP": loss[i], "Accuracy_ESM1bts_ECFP": accuracy[i], "ROC-AUC_ESM1bts_ECFP": ROC_AUC[i]})
    
print("Loss values: %s" %loss) 
print("Accuracies: %s" %accuracy)
print("ROC-AUC scores: %s" %ROC_AUC)

np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "acc_CV_xgboost_ESM1b_ts_ECFP.npy"), np.array(accuracy))
np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "loss_CV_xgboost_ESM1b_ts_ECFP.npy"), np.array(loss))
np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "ROC_AUC_CV_xgboost_ESM1b_ts_ECFP.npy"), np.array(ROC_AUC))

dtrain = xgb.DMatrix(np.array(train_X), weight = weights, label = np.array(train_y),
                feature_names= feature_names)
dtest = xgb.DMatrix(np.array(test_X), label = np.array(test_y),
                    feature_names= feature_names)

bst = xgb.train(param,  dtrain, int(num_round), verbose_eval=1)
y_test_pred = np.round(bst.predict(dtest))
acc_test = np.mean(y_test_pred == np.array(test_y))
roc_auc = roc_auc_score(np.array(test_y), bst.predict(dtest))
mcc = matthews_corrcoef(np.array(test_y), y_test_pred)

print("Accuracy on test set: %s, ROC-AUC score for test set: %s, MCC: %s"  % (acc_test, roc_auc, mcc))
wandb.log({"Test Accuracy_ESM1bts_ECFP": acc_test, "Test ROC-AUC_ESM1bts_ECFP": roc_auc, "Test MCC_ESM1bts_ECFP": mcc})

np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "y_test_pred_xgboost_ESM1b_ts_ECFP.npy"), bst.predict(dtest))
np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "y_test_true_xgboost_ESM1b_ts_ECFP.npy"), test_y)
