# Substrate Inhibitor Prediction(SIP)

Addressing data leakage in ESP model


## Setup Instructions
###  Folder structure description
```
SIP/
├── ESP_HardSplits/
│ ├── data/
│ │ ├── data_ESP/
│ │ ├── data_ProSmith/
│ │ └── Reports/
│ └── notebooks_and_code/
│   └── additional_code/
├── SIP/
├── README.md
└── requirements.txt
```

### Setting up `sail` Environment for DataSAIL

```bash
# Create and activate the environment
conda create --name sail python=3.12.0
conda activate sail

# Install Mamba package manager (optional but recommended)
conda install mamba -n sail -c conda-forge

# Install datasail-lite package
mamba install -c kalininalab -c conda-forge -c bioconda datasail-lite

# Install Grakel using pip
pip install grakel
```
### Setting up `SIP` Environment

```bash
# Create and activate the environment
conda create --name SIP python=3.12.0
conda activate SIP

# Install required packages
conda install pandas=2.2.2
conda install numpy=1.26.4
pip install rdkit==2024.3.1
pip install biopython==1.84
conda install matplotlib=3.8.4
conda install scikit-learn=1.4.2 

# Install PyTorch with CPU support
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### Getting Started

#### 1- Run 1-DataPreparation.py to generate all data set need to perform data split
#### 2- Run 2-1-SplitByDataSAIL.py
```
python 2-1-SplitByDataSAIL.py --split-method [C2, C1e, C1f, I1e I1f] --split-number [8 2, 7 2 1] --Data-suffix ['', _NoATP ,_D3408]
```
##### Example:
```
python 2-1-SplitByDataSAIL.py --split-method C1e --split-number 8 2 --Data-suffix ''
```


```
