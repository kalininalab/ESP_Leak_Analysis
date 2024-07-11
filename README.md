# Substrate Inhibitor Prediction (SIP)

## Table of Contents
- [Introduction](#introduction)
  - [Setup Instructions](#setup-instructions)
    - [Folder Structure](#folder-structure)
    - [Setting up `SIP` Environment](#setting-up-sip-environment)
- [ESP_Hard Split](#esp_hard-split)
  - [Data Preparation](#data-preparation)
    - [1-DataPreparation.py](#1-datapreparation.py)
  - [Splitting Data](#splitting-data-)
    - [2-1-SplitByDataSAIL.py](#2-1-splitbydatasailpy)
    - [2-2-SplitByESP.py](#2-2-splitbyesppy)
  - [Hyperparameter optimization and model training](#hyperparameter-optimization-and-model-training)
    - [3-1-HyperOp_TraningXgb_2Splits.py](#3-1-hyperop_traningxgb_2splitspy)
    - [3-2-HyperOp_TraningXgb_3Splits.py](#3-2-hyperop_traningxgb_3splitspy)
- [Substrate Inhibitor prediction-SIP](#substrate-inhibitor-prediction-sip)



## Introduction

This project was conducted in two parts. In first part we address data leakage in the ESP model by splitting data by a powerful tools named DataSAIL(short for Data Splitting Against Information Leakage). 

## Setup Instructions
###  Folder structure
```
SIP/
├── ESP_HardSplits/
│ ├── data/
│ │ ├── data_ESP/
│ │ ├── 2splits/
│ │ ├── 3splits/
│ │ ├── data_ProSmith/
│ │ └── Reports/
│ └── notebooks_and_code/
│   └── additional_code/
├── SIP/
├── README.md
└── requirements.txt
```

### Setting up `SIP` Environment
* It is recommended to install the packages in order

* For MacOSX M1 desktop 

      conda create --name SIP python=3.12.0
      conda activate SIP
      conda install mamba -n SIP -c conda-forge
      mamba install -c kalininalab -c conda-forge -c bioconda datasail-lite
      pip install grakel
      pip install xgboost==2.1.0
      pip install biopython==1.84
      pip install torch==2.3.1
      pip install hyperopt==0.2.7
      pip install colorama==0.4.6
      pip install libchebipy==1.0.10

* For server with linux OS

      conda create --name SIP python=3.12.0
      conda activate SIP
      conda install mamba -n SIP -c conda-forge
      mamba install -c kalininalab -c conda-forge -c bioconda datasail-lite
      pip install grakel
      conda install -c bioconda cd-hit
      conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
      conda install conda-forge::xgboost
      pip install biopython==1.84
      pip install hyperopt==0.2.7
      pip install colorama==0.4.6
      pip install libchebipy==1.0.10
      pip install wandb

# -------------------------------***ESP_Hard Split***---------------------------------------
## Data Preparation

### 1-DataPreparation.py 

* After running this script, three different versions of the data will be generated:

        dataESP.pkl: Original ESP data containing only positive data points with experimental evidence.
        dataESP_NoATP.pkl: This dataset excludes all ATP data points from dataESP.pkl.
        dataESP_D3408.pkl: This dataset randomly removes 3408 data points from dataESP.pkl (equivalent to the number of ATP points).


* The reason for randomly deleting 3408 data points is to create a control case to understand the impact of ATP removal on model performance, as approximately 20% of the molecules are ATP. dataESP.pkl: Original ESP data containing only positive data points with experimental evidence.




## Splitting Data 
* This table outlines an overview of all  different split strategies we used in this project.

| split | DataFrame          | 2splits | training            | 3splits | training            |
|-------|--------------------|---------|---------------------|---------|---------------------|
| C1e*  | dataESP.pkl        | Yes     | ESM1bts+PreGNN/ECFP | Yes     | ESM1bts+PreGNN/ECFP |
| C1f   | dataESP.pkl        | Yes     | ESM1bts+PreGNN/ECFP | Yes     | ESM1bts+PreGNN/ECFP |
| I1e   | dataESP.pkl        | Yes     | ESM1bts+PreGNN/ECFP | Yes     | ESM1bts+PreGNN/ECFP |
| I1f   | dataESP.pkl        | Yes     | ESM1bts+PreGNN/ECFP | Yes     | ESM1bts+PreGNN/ECFP |
| C2    | dataESP.pkl        | No      | ESM1bts+PreGNN/ECFP | No      |                     |
| ESP+  | train&test_C1f.pkl | Yes     | ESM1bts+PreGNN/ECFP | Yes     | ESM1bts+PreGNN/ECFP |
| ESP+  | train&test_C2.pkl  | Yes     | ESM1bts+PreGNN/ECFP | No      |                     |
| ESP   | dataESP_NoATP.pkl  | Yes     | ESM1bts+PreGNN/ECFP | No      |                     |
| ESP   | dataESP_D3408.pkl  | Yes     | ESM1bts+PreGNN/ECFP | No      |                     |

* *DataSAIL can split data in 1 and 2 dimensions(1D,2D). The 1D splits are [C1e, C1f, I1e I1f] and the 2D splits are C2 and I2, we used C2 and all 1D splits in this project. To get more information please check the dataSAIL webpage(https://datasail.readthedocs.io/en/latest/index.html).
* +In this project we refer to the split method that used in ESP paper as ESP split
### 2-1-SplitByDataSAIL.py
* This script aims to split (by DataSAIL) and generate negative data for each DataFrame explained in above.

       python 2-1-SplitByDataSAIL.py --split-method [C2, C1e, C1f, I1e I1f] --split-size [8 2, 7 2 1] --Data-suffix ['', _NoATP ,_D3408]

* Explanation of Arguments:

       --split-method [C2, C1e, C1f, I1e, I1f]: Specifies the methods used for splitting the data.
       --split-size [8 2, 7 2 1]: Defines the number of splits for each method.
       --Data-suffix [NoATP, D3408]: Indicates which data files to parse. It is an optinoal argument, if you dont use it,  the original data(data_ESP.pkl) will be parsed.

* Data Suffix Details:

      NoATP: Parses the dataESP_NoATP.pkl file.
      D3408: Parses the dataESP_D3408.pkl file.

* Example:

        python 2-1-SplitByDataSAIL.py --split-method C1e --split-size 8 2 

* Output files:

      ./SIP/data/2splits/train_C1e_2S.pkl
      ./SIP/data/2splits/test_C1e_2S.pkl
      ./SIP/data/Reports/Report_2Splits_C1e.log

### 2-2-SplitByESP.py
* This script aims to generate a control set for each split produced by dataSAIL and also create negative data for each split. The original ESP dataset contains some missing (NaN) data, and for some molecules, we couldn't find the SMILES string. Additionally, during parsing with dataSAIL, some molecules had invalid SMILES strings. Consequently, the size of the dataset is smaller than the original ESP dataset.

* This script accepts the same arguments as 2-1-SplitByDataSAIL.py:

      python 2-2-SplitByESP.py --splitted-data [C2, C1e] --split-size [8 2, 7 2 1] --Data-suffix [NoATP, D3408]

* The splitted-data is an optional argument. However, if specified, should be one of the following: [C2,C1f] to get access to train and test sets related to C1f and C2.
* Since in ESP paper the data have been split based on enzyme and also CV has been done based on sequence's indices(all related indices to an enzyme fall into same fold of CV), we choose train and test resulted to "C1f" split to create control case for all 1D splits.


* Example:

      python 2-2-SplitByESP.py --split-method C1f --split-size 8 2 

* Output files:

      ./SIP/data/2splits/train_ESP-C1e_2S.pkl
      ./SIP/data/2splits/test_ESP-C1e_2S.pkl
      ./SIP/data/Reports/split_report/Report_ESPC1f_2S.log
* The `ESPC1f` emphasizes that the combined data of `C1f` are used to perform the `ESP` split.


## Hyperparameter optimization and model training

### 3-1-HyperOp_TraningXgb_2Splits.py

### 3-2-HyperOp_TraningXgb_3Splits.py


# -----------------------***Substrate Inhibitor prediction-SIP***---------------------