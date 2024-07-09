# Substrate Inhibitor Prediction (SIP)

## Table of Contents

- [Introduction](#introduction)
- [Setup Instructions](#setup-instructions)
  - [Folder Structure](#folder-structure)
  - [Setting up `sail` Environment for DataSAIL](#setting-up-sail-environment-for-datasail)
  - [Setting up `SIP` Environment](#setting-up-sip-environment)
- [Getting Started](#getting-started)
  - [Data Preparation](#1-run-1-datapreparation.py)
  - [Data Splitting](#2-run-2-1-splitbydatasail.py)
- [Contact](#contact)

## Introductions

Welcome to the Substrate Inhibitor Prediction (SIP) project. This project addresses data leakage in the ESP and SIP models. 

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
It is recommended to install the packages in order

For MacOSX M1 desktop 
```bash
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
```
For server with linux OS
```
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
```
### Getting Started

#### 1- Run 1-DataPreparation.py to generate all data set need to perform data split
After running this script, three different versions of the data will be generated:

    dataESP.pkl: Original ESP data containing only positive data points with experimental evidence.

    dataESP_NoATP.pkl: This dataset excludes all ATP data points from dataESP.

    dataESP_D3408.pkl: This dataset randomly removes 3408 data points from dataESP (equivalent to the number of ATP points).


| split          | 2splits  | training      | 3splits    | training       |
|----------------|----------|---------------|------------|----------------|
| C1e            | Yes      | ESM+PGNN/ECFP | Yes        | ESM+PGNN/ECFP  |
| C1f            | Yes      | ESM+PGNN/ECFP | Yes        | ESM+PGNN/ECFP  |
| I1e            | Yes      | ESM+PGNN/ECFP | Yes        | ESM+PGNN/ECFP  |
| I1f            | Yes      | ESM+PGNN/ECFP | Yes        | ESM+PGNN/ECFP  |
| C2             | Yes      | ESM+PGNN/ECFP | No         |                |
| ESP(C1e)       | Yes      | ESM+PGNN/ECFP | No         |                |
| ESP(C2)        | Yes      | ESM+PGNN/ECFP | No         |                |
| ESP(C1e)_NoATP | Yes      | ESM+PGNN/ECFP | No         |                |
| ESP(C1e)_D3408 | Yes      | ESM+PGNN/ECFP | No         |                |




#### 2- Run 2-1-SplitByDataSAIL.py
```
python 2-1-SplitByDataSAIL.py --split-method [C2, C1e, C1f, I1e I1f] --split-size [8 2, 7 2 1] --Data-suffix ['', _NoATP ,_D3408]
```
Explanation of Arguments:

     --split-method [C2, C1e, C1f, I1e, I1f]: Specifies the methods used for splitting the data.
     --split-size [8 2, 7 2 1]: Defines the number of splits for each method.
     --Data-suffix ['', _NoATP, _D3408]: Indicates which data files to parse.

Data Suffix Details:

     '': Parses the dataESP.pkl file.
     _NoATP: Parses the dataESP_NoATP.pkl file.
     _D3408: Parses the dataESP_D3408.pkl file.

##### Example:
```
python 2-1-SplitByDataSAIL.py --split-method C1e --split-size 8 2 --Data-suffix ''
```
Output files:
```
./SIP/data/2splits/train_C1e_2S.pkl
./SIP/data/2splits/test_C1e_2S.pkl
./SIP/data/Reports/Report_2Splits_C1e.log
```
