# Substrate Inhibitor Prediction(SIP)

Addressing data leakage in ESP model

# Substrate Inhibitor Prediction (SIP)

*Addressing data leakage in ESP model*

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://example.com/build-status)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

![Project Logo](https://example.com/logo.png)

## Table of Contents

- [Introduction](#introduction)
- [Setup Instructions](#setup-instructions)
  - [Folder Structure](#folder-structure)
  - [Setting up `sail` Environment for DataSAIL](#setting-up-sail-environment-for-datasail)
  - [Setting up `SIP` Environment](#setting-up-sip-environment)
- [Getting Started](#getting-started)
  - [Data Preparation](#1-run-1-datapreparationpy)
  - [Data Splitting](#2-run-2-1-splitbydatasailpy)
- [Contact](#contact)

## Introduction

Welcome to the Substrate Inhibitor Prediction (SIP) project. This project addresses data leakage in the ESP model by implementing a robust data preparation and splitting strategy.

## Setup Instructions

### Folder Structure



## Setup Instructions
###  Folder structure description
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

### Setting up `sail` Environment for DataSAIL

```bash
conda create --name sail python=3.12.0
conda activate sail
conda install mamba -n sail -c conda-forge
mamba install -c kalininalab -c conda-forge -c bioconda datasail-lite
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
After running this script, three different versions of the data will be generated:

dataESP: Original ESP data containing only positive data points with experimental evidence.

dataESP_NoATP: This dataset excludes all ATP data points from dataESP.

dataESP_D3408: This dataset randomly removes 3408 data points from dataESP (equivalent to the number of ATP points).

#### 2- Run 2-1-SplitByDataSAIL.py
```
python 2-1-SplitByDataSAIL.py --split-method [C2, C1e, C1f, I1e I1f] --split-number [8 2, 7 2 1] --Data-suffix ['', _NoATP ,_D3408]
```
##### Example:
```
python 2-1-SplitByDataSAIL.py --split-method C1e --split-number 8 2 --Data-suffix ''
```


```
