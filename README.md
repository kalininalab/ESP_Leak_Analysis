# ESP Leak Analysis (SIP)

## Table of Contents
- [Introduction](#---------------------introduction----------------------)
  - [Setup Instructions](#setup-instructions)
    - [Folder Structure](#folder-structure)
    - [Setting up `SIP` Environment](#setting-up-sip-environment)
- [ESP_Hard Split](#--------------------esp_hard-split---------------------)
  - [Data Preparation](#data-preparation)
    - [1-DataPreparation.py](#data-preparation)
  - [Splitting Data](#splitting-data-)
    - [2-1-SplitByDataSAIL.py](#2-1-splitbydatasailpy)
    - [2-2-SplitByESP.py](#2-2-splitbyesppy)
  - [Hyperparameter optimization and model training](#hyperparameter-optimization-and-model-training)
    - [3-1-HyperOp_TraningXgb_2Splits.py](#3-1-hyperop_traningxgb_2splitspy)
    - [3-2-HyperOp_TraningXgb_3Splits.py](#3-2-hyperop_traningxgb_3splitspy)


# ---------------------***Introduction***----------------------

This project was conducted in two parts. In the first part, we addressed data leakage in the ESP model by splitting the data using a powerful tool named DataSAIL (Data Splitting Against Information Leakage).

In the second part of the project, we extended the ESP dataset. To achieve this, we extracted 3,450 positive data points from the BRENDA database. Instead of generating negative data points, we used inhibitors as negative data points.

Furthermore, we utilized the EMS1b model to embed the enzyme sequences, focusing on the residues on the enzyme's surface. Additionally, we used the ChemBERTa model to embed the structure of the molecules, paying attention to the functional groups.


## Setup Instructions
###  Folder structure
```
├── ESP_Leak_Analysis/
│ ├── data/
│ │ ├── data_ESP/
│ │ ├── 2splits/
│ │ ├── 3splits/
│ │ ├── data_ProSmith/
│ │ ├── Reports/
│ │ ├── training_results_2S
│ │ └── training_results_3S
│ └── notebooks_and_code/
│   └── additional_code/
└── README.md
```

### Setting up `ESL_LA` Environment
* It is recommended to install the packages in order

* For MacOSX M1 desktop 

      conda create --name ESP_LA python=3.12.0
      conda activate ESP_LA
      conda install mamba -n ESP_LA -c conda-forge
      mamba install -c kalininalab -c conda-forge -c bioconda datasail-lite
      pip install grakel
      pip install xgboost==2.1.0
      pip install biopython==1.84
      pip install torch==2.3.1
      pip install hyperopt==0.2.7
      pip install colorama==0.4.6
      pip install libchebipy==1.0.10

* For server with linux OS

      conda create --name ESP_LA python=3.12.0
      conda activate ESP_LA
      conda install mamba -n ESP_LA -c conda-forge
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

# --------------------***ESP_Hard Split***---------------------
## Data Preparation

### 1-DataPreparation.py 

* After running this script, three different versions of the data will be generated:

        dataESP.pkl: Original ESP data containing only positive data points with experimental evidence.


* The reason for randomly deleting 3408 data points is to create a control case to understand the impact of ATP removal on model performance, as approximately 20% of the molecules in the dataESP are ATP.




## Splitting Data 
* This table outlines an overview of all  different split strategies we used in this project.
## Splitting Data 
* This table outlines an overview of all different split strategies we used in this project.

| Method   | Cross-val (Y/N) | Train Samples | Val Samples | Test Samples | Train Neg/Pos | Val Neg/Pos | Test Neg/Pos |
|----------|----------------|---------------|-------------|--------------|---------------|-------------|--------------|
| S1ₗ      | Y              | 55430         | n.a.        | 13901        | 2.78          | n.a.        | 2.80         |
| S1ₚ      | Y              | 55522         | n.a.        | 13904        | 2.79          | n.a.        | 2.80         |
| I1ₗ      | Y              | 55209         | n.a.        | 13845        | 2.77          | n.a.        | 2.78         |
| I1ₚ      | Y              | 55651         | n.a.        | 13864        | 2.80          | n.a.        | 2.79         |
| ESP      | Y              | 55374         | n.a.        | 14056        | 2.79          | n.a.        | 2.81         |
| S2       | Y              | 32455         | n.a.        | 6913         | 2.79          | n.a.        | 2.78         |
| ESPₛ₂    | Y              | 31444         | n.a.        | 7933         | 2.78          | n.a.        | 2.82         |
| S1ₗ      | N              | 48489         | 6825        | 13839        | 2.78          | 2.73        | 2.78         |
| S1ₚ      | N              | 48497         | 6941        | 13919        | 2.78          | 2.79        | 2.80         |
| I1ₗ      | N              | 48938         | 6966        | 12121        | 2.82          | 2.80        | 2.31         |
| I1ₚ      | N              | 48660         | 6982        | 13844        | 2.78          | 2.81        | 2.80         |
| ESP      | N              | 48541         | 7008        | 14034        | 2.79          | 2.82        | 2.82         |
| S2       | N              | 25604         | 1837        | 3622         | 2.77          | 2.73        | 2.75         |
| ESPₛ₂    | N              | 22128         | 3095        | 6344         | 2.81          | 2.90        | 2.83         |

* DataSAIL can split data in 1 and 2 dimensions (1D, 2D). The 1D splits are [S1ₗ, S1ₚ, I1ₗ, I1ₚ] and the 2D splits are S2 and I2. We used S2 and all 1D splits in this project. For more information please check the dataSAIL webpage (https://datasail.readthedocs.io/en/latest/index.html).
* In this project we refer to the split method that was used in the ESP paper as ESP split.

* *DataSAIL can split data in 1 and 2 dimensions(1D,2D). The 1D splits are [C1e, C1f, I1e I1f] and the 2D splits are C2 and I2, we used C2 and all 1D splits in this project. To get more information please check the dataSAIL webpage(https://datasail.readthedocs.io/en/latest/index.html).
* +In this project we refer to the split method that used in ESP paper as ESP split
### 2-1-SplitByDataSAIL.py
* This script aims to split (by DataSAIL) and generate negative data for each DataFrame explained in above.

       python 2-1-SplitByDataSAIL.py --split-method [C2, C1e, C1f, I1e I1f] --split-size [8 2, 7 2 1] --input-path 

* Explanation of Arguments:

       --split-method [C2, C1e, C1f, I1e, I1f]: Specifies the methods used for splitting the data.
       --split-size [8 2, 7 2 1]: Defines the number of splits for each method.
       --input-path  ./../data/data_ESP/dataESP.pkl


* Example:

        python 2-1-SplitByDataSAIL.py --split-method C1e --split-size 8 2 --input-path  ./../data/data_ESP/dataESP.pkl

* Output files:

      ./SIP/data/2splits/train_C1e_2S.pkl
      ./SIP/data/2splits/test_C1e_2S.pkl
      ./SIP/data/Reports/Report_2Splits_C1e.log

### 2-2-SplitByESP.py
* This script aims to generate a control set for 1D abd 2D splits produced by dataSAIL and then creates negative data for each split. The original ESP dataset contains some missing (NaN) data, and for some molecules, we couldn't find the SMILES string. Additionally, during parsing with dataSAIL, some molecules had invalid SMILES strings. Consequently, the size of the dataset is smaller than the original ESP dataset.

* Explanation of Arguments:

      python 2-2-SplitByESP.py  --split-size [8 2, 7 2 1] --input-path 

* `--split-size` and `--input-path ` are same as `2-1-SplitByDataSAIL.py`


* Example:

      python 2-2-SplitByESP.py  --split-size 8 2 --input-path  ./../data/data_ESP/dataESPC2.pkl

* Output files:

      ./SIP/data/2splits/train_ESPC2_2S.pkl
      ./SIP/data/2splits/test_ESPC2_2S.pkl
      ./SIP/data/Reports/split_report/Report_ESPC2_2S.log
* The `ESPC2` emphasizes that the combined data of `C2` are used to perform the `ESP` split.


## Hyperparameter optimization and model training

### 3-1-HyperOp_TraningXgb_2Splits.py
* This script aims to tune the hyperparameters and train xgboost model for each split methods produced under 2 splits (train:test) scenario

       python 3-1-HyperOp_TraningXgb_2Splits.py --split-data [C2,C1e, C1f, I1e, I1f,ESP, ESPC1f, ESPC2] --column-name [ECFP, PreGNN] --Data-ATP [NoATP, D3408]
* Explanation of Arguments:
* `--split-data` and `--Data-ATP` are same as before 
* `--column-name` determines which embedded vector for the molecule should be concatenated with the ESMb1ts vector for hyperparameter optimization and training.
### 3-2-HyperOp_TraningXgb_3Splits.py
* This script aims to tune the hyperparameters and train xgboost model for each split methods produced under 3 splits (train:test:val) scenario
* Explanation of Arguments: The Arguments are same as `3-1-HyperOp_TraningXgb_2Splits.py`
