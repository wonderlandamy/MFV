# MFV: Enhanced Person Re-Identification through Multi-Scale Feature Fusion and Attention Mechanisms

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-orange.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)

This is the official implementation of the paper **"Enhanced Person Re-Identification through Multi-Scale Feature Fusion and Attention Mechanisms"**.

## Abstract

The field of person re-identification (Re-ID) has witnessed significant advancements with the integration of deep learning technologies. This paper introduces a **Multi-scale Fusion Visual (MFV)** framework that leverages the strengths of both Transformer models and Convolutional Neural Networks (CNNs) to enhance feature extraction and aggregation.

The proposed framework comprises three key components:
1.  **Stratified Deep Feature Aggregation (SDFA):** Progressively integrates and refines feature representations through staged supervision.
2.  **Transformer-based Feature Extraction (TFE):** Incorporates **External Attention (EA)** to mine potential relationships in the dataset with linear complexity, serving as a dictionary for the whole dataset.
3.  **Multi-scale Feature Extractor (MFE):** Uses a ResNet backbone to extract layered features with different scales and semantic information.

Achieving state-of-the-art performance on **Market1501 (Rank-1: 94.70%, mAP: 87.60%)** and other benchmarks.

## Prerequisites

The code is implemented using **PyTorch** and trained on an **NVIDIA RTX 4090 GPU**.

### Installation

# Clone this repository
git clone [https://github.com/wonderlandamy/MFV.git](https://github.com/wonderlandamy/MFV.git)
cd MFV

# Create a conda environment
conda create -n mfv python=3.8
conda activate mfv

# Install dependencies (Example)
pip install torch torchvision --extra-index-url [https://download.pytorch.org/whl/cu113](https://download.pytorch.org/whl/cu113)
pip install -r requirements.txt

### Datasets
We evaluate our method on four large-scale benchmarks: Market1501, DukeMTMC-reID, CUHK03-NP, and MSMT17.

Please download the datasets and organize them as follows:

data
├── market1501
│   ├── bounding_box_train
│   ├── bounding_box_test
│   └── query
├── dukemtmc
│   ├── bounding_box_train
│   ├── ...
├── msmt17
└── cuhk03

Market1501: Collected from 6 cameras, containing 1,501 identities.
DukeMTMC-reID: Contains 36,411 images from 8 cameras.
MSMT17: A larger dataset resembling real scenarios with 126,441 images.


