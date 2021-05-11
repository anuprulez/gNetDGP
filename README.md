# gNetDGP
An End-to-End Graph Neural Network for Disease Gene Prioritization.

## Table of contents
* [Installation](#installation)
    * [Docker](#using-docker)
    * [Conda](#using-conda)
* [Usage](#usage)
    * [Train the generic model](#train-the-generic-model)
    * [Predict using the generic model](#predict-using-the-generic-model)
    * [Train the specific model](#train-the-specific-model)
    * [Predict using the specific model](#predict-using-the-specific-model)
* [Additional material](#additional-material)

## Installation
### Using docker
We provide a [Dockerfile](Dockerfile) to setup a runtime. To use is run
```bash
docker build -t gNetDGP .
```
 
### Using Conda
```bash
conda env create -f environment.yml
conda activate gnetdgp_env
```

## Usage
### Train the generic model
### Train the specific model
### Predict using the generic model
### Predict using the specific model


## Additional material
The process of the reported experiments is documented in
* [results/disease_gene_prioritization.ipynb for the disease gene prioritization task.](results/disease_gene_prioritization.ipynb)
* [results/disease_gene_classification.ipynb for the disease type classification.](results/disease_gene_classification.ipynb)
