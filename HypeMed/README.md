# HypeMed: Unleashing the Power of Hypergraph Contrastive Learning for Medication Recommendation
HypeMed is an innovative framework designed for medication recommendations by capturing intricate relationships within Electronic Health Records (EHRs). Leveraging hypergraph contrastive learning, HypeMed considers patient history, medical entity interactions, and prescription patterns across different levels, resulting in highly accurate and balanced medication recommendations. It strikes a fine balance between precision and mitigating medication-related risks, thus enhancing patient safety and treatment efficacy.


## Table of Contents

- [Description](#description)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Description
This project contains the necessary python scripts for HypeMed as well as the directions. 
Considering that unauthorized public access to the MIMIC-III and MIMIC-IV databases is prohibited, we do not provide the associated data. You can use our provided scripts to preprocess the raw data once you have obtained the relevant data.
## Requirements
```text
Python==3.8.36
Torch==1.13.1+cu116
NumPy==1.24.4
```

## Usage
We follow the preprossing procedures of [SafeDrug](https://github.com/ycq091044/SafeDrug/tree/archived).

Below is a guide on how to use the scripts. Before processing, you should put the necessary data in the `data` directory.

```bash
# Data Processing.
python data/processing.py # MIMIC-III
python data/processing_4.py #MIMIC-IV

# Contrastive Learning Pretraing (on MIMIC-III)
python HypeMed/HypeMedPretrain.py --pretrain --pretrain_epoch 300 --pretrain_lr 1e-3 --pretrian_weight_decay 1e-5 --mimic 3 --name example
# Training
python HypeMed/HypeMedPretrain.py --mimic 3 --name example
# Testing
python HypeMed/HypeMedPretrain.py --mimic 3 --Test --name example
# ablation study
python HypeMed/HypeMedPretrain.py --mimic 3 --channel_ablation mem --name example
```
You can explore all adjustable hyperparameters through the `HypeMed/config.py` file.
