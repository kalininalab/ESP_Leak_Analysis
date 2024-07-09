# Substrate Inhibitor Prediction (SIP)

## Table of Contents

- [Introduction](#introduction)
- [Setup Instructions](#setup-instructions)
  - [Folder Structure](#folder-structure)
  - [Setting up `SIP` Environment](#setting-up-sip-environment)
- [Data Preparation](#data-preparation)
  - [1-DataPreparation.py](#1-datapreparation.py)
- [Splitting Data](#splitting-data-)
  - [2-1-SplitByDataSAIL.py](#2-1-splitbydatasailpy)
  - [2-2-SplitByESP.py](#2-2-splitbyesppy)
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

### Data Preparation
        
  | data set     | split method         |
  |--------------|----------------------|
  | dataESP.pkl  | DataSAIL*, ESP+      |
  | dataESP_NoATP| ESP                  |
  | dataESP_D3408| ESP                  |

* *DataSAIL can split data in 1 and 2 dimensions(1D,2D). The 1D splits are [C1e, C1f, I1e I1f] and the 2D splits are C2 and I2, we used C2 and all 1D splits in this project. To get more informeation please check the dataSAIL webpage(https://datasail.readthedocs.io/en/latest/index.html).
* +In this project we refer to the split method that used in ESP paper as ESP split

#### 1-DataPreparation.py 
* to generate all data set need to perform data split  After running this script, three different versions of the data will be generated:
* Download the original ESP train and test sets from ESP GitHub. After running this script, three different versions of the data will be generated:

        dataESP.pkl: Original ESP data containing only positive data points with experimental evidence.
        dataESP_NoATP.pkl: This dataset excludes all ATP data points from dataESP.pkl.
        dataESP_D3408.pkl: This dataset randomly removes 3408 data points from dataESP.pkl (equivalent to the number of ATP points).


* The reason for randomly deleting 3408 data points is to create a control case to understand the impact of ATP removal on model performance, as approximately 20% of the molecules are ATP. dataESP.pkl: Original ESP data containing only positive data points with experimental evidence.




### Splitting Data 
* This table outlines the availability of different split methods and the corresponding concatenated embedding vector which was used for 2 splits and 3 splits.

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

#### 2-1-SplitByDataSAIL.py
```
python 2-1-SplitByDataSAIL.py --split-method [C2, C1e, C1f, I1e I1f] --split-size [8 2, 7 2 1] --Data-suffix ['', _NoATP ,_D3408]
```
* Explanation of Arguments:

       --split-method [C2, C1e, C1f, I1e, I1f]: Specifies the methods used for splitting the data.
       --split-size [8 2, 7 2 1]: Defines the number of splits for each method.
       --Data-suffix ['', _NoATP, _D3408]: Indicates which data files to parse.

* Data Suffix Details:

       '': Parses the dataESP.pkl file.
       _NoATP: Parses the dataESP_NoATP.pkl file.
       _D3408: Parses the dataESP_D3408.pkl file.

* Example:

        python 2-1-SplitByDataSAIL.py --split-method C1e --split-size 8 2 --Data-suffix ''

* Output files:

      ./SIP/data/2splits/train_C1e_2S.pkl
      ./SIP/data/2splits/test_C1e_2S.pkl
      ./SIP/data/Reports/Report_2Splits_C1e.log

#### 2-2-SplitByESP.
* This script aims to generate a control set for each split produced by dataSAIL. The original ESP dataset contains some missing (NaN) data, and for some molecules, we couldn't find the SMILES string. Additionally, during parsing with dataSAIL, some molecules had invalid SMILES strings. Consequently, the size of the dataset is smaller than the original ESP dataset.

* For the 1D split, we combine one of the hard split train and test sets and then split them using the method reported in the ESP paper. For the 2D split, we apply the same process to the C2 train and test sets. Since the 2D split is the hardest and we aim to minimize data leakage, some data points are not selected for the final splits. Thus, we combine the C2 split train and test sets and split them again using the ESP split methods.
* This script accepts the same arguments as 2-1-SplitByDataSAIL.py:

      python 2-1-SplitByDataSAIL.py --split-method [C2, C1e, C1f, I1e, I1f] --split-size [8 2, 7 2 1] --Data-suffix ['', _NoATP, _D3408]

* However, as mentioned above, we use the data from the C2 split to create control data for the C2 split and for all 1D splits, since the number of data points is the same, we randomly choose C1e to create control data for all 1D hard splits (C1e, C1f, I1e, I1f).


* Example:

      python 2-2-SplitByESP_Method.py --split-method C1e --split-size 8 2 --Data-suffix ''

* Output files:

      ./SIP/data/2splits/train_ESP(C1e)_2S.pkl
      ./SIP/data/2splits/test_ESP(C1e)_2S.pkl
      ./SIP/data/Reports/split_report/Report_ESP(C1e)_2S.log
* The `ESP(C1e)` emphasizes that the combined data of `C1e are used to perform the ESP split.