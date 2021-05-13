# gNetDGP
An End-to-End Graph Neural Network for Disease Gene Prioritization.

## Table of contents
* [Installation](#installation)
    * [Docker](#using-docker)
    * [Conda](#using-conda)
* [Test runs](#test-runs)
    * [Train the generic model](#train-the-generic-model)
    * [Predict using the generic model](#predict-using-the-generic-model)
    * [Train the specific model](#train-the-specific-model)
    * [Predict using the specific model](#predict-using-the-specific-model)
* [Additional material](#additional-material)

## Installation
### Using docker
We provide a [Dockerfile](Dockerfile) to setup a runtime. To use it run
```bash
docker build -t gNetDGP .
```
 
### Using Conda
```bash
conda env create -f environment.yml
conda activate gnetdgp_env
```

## Test runs
To get an overview use
```bash
python main.py --help
```

To list the available options on a specific command use
```bash
python main.py [COMMAND] --help
```

### Train the generic model
To train a new generic model use
```bash
python main.py generic-train
```

### Prioritization using the generic model
Provide a input file of gene, disease tuples 
like in the [test/example_input_generic_predict.tsv](test/example_input_generic_predict.tsv)

Then run the command
```bash
python main.py generic-predict test/example_input_generic_predict.tsv
```
This will score the provided disease, gene tuples and return a augmented version of the input file with added scores.
The result is stored in `--out_file`, the default is `./generic_predict_results.tsv` 

To get a list of available genes in the model run
```bash
python main.py generic-predict --get_available_genes
```

To get a list of available diseases in the model run
```bash
python main.py generic-predict --get_available_diseases
```

### Train the specific model
### Predict using the specific model


## Additional material
The process of the reported experiments is documented in
* [results/disease_gene_prioritization.ipynb for the disease gene prioritization task.](results/disease_gene_prioritization.ipynb)
* [results/disease_gene_classification.ipynb for the disease type classification.](results/disease_gene_classification.ipynb)
