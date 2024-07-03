# Substrate Inhibitor Prediction(SIP)

Brief description of your project.

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

### Setting up `sail` Environment

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
```

```
