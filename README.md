# SubCellAE

SubCellAE is a Python project for learning latent representations of subcellular image patches with autoencoders and variational autoencoders, then analyzing those latent features with embedding, clustering, labeling, and classification utilities.

The repository currently contains:

- data-preparation helpers for extracting and padding microscopy patches
- PyTorch models for autoencoder and VAE training
- dataset utilities for TIFF patch loading
- latent-space analysis utilities for embedding and clustering
- classifier training and evaluation helpers on latent features
- scripts for exporting and combining external label annotations

## Repository Layout

```text
subcellae/
  analysis/         latent extraction and feature analysis helpers
  classification/   classifier training and evaluation
  clustering/       clustering utilities
  dataprep/         patch extraction and preprocessing helpers
  modelling/        AE/VAE models and training scripts
  pipeline/         end-to-end post-training analysis pipeline
  utils/            small utility modules and plotting helpers

config/             analysis configuration templates and constants
scripts/            runnable scripts, including label-processing utilities
Makefile            environment and analysis shortcuts
environment*.yml    Conda environment definitions
pyproject.toml      package metadata
```

## Setup

This project uses Conda environments and installs the package in editable mode.

CPU environment:

```bash
make env
conda activate subcellvae
```

CUDA environment:

```bash
make env-cuda
conda activate subcellvae-cuda
```

Minimal dependencies are defined in:

- `environment.yml`
- `environment-cuda.yml`

A larger pinned environment snapshot is also included in `environment.full.yml`.

## Core Components

### Models

The main model definitions live in `subcellae/modelling/subcellvae.py`:

- `AE`: convolutional autoencoder for image patches
- `VAE32`: variational autoencoder for 32x32 patches

There is also an older training script in `subcellae/modelling/train_AE_multisets.py` that appears to be tailored to local datasets and hard-coded filesystem paths.

### Dataset Loading

`subcellae/modelling/dataset.py` provides `TIFFDataset`, which loads `.tif` and `.tiff` patches into memory and returns:

- image tensor
- label or group id
- original image path

### Data Preparation

`subcellae/dataprep/patch_prep.py` contains reusable helpers for:

- loading CZI images and segmentation masks
- padding images
- computing centered patch grids
- applying optional translation and rotation
- saving extracted TIFF patches
- recording patch metadata rows

### Analysis Pipeline

`subcellae/pipeline/analysis_pipeline.py` is intended to run post-training analysis on a trained model. Based on the code, it performs:

1. latent extraction from a dataloader
2. reconstruction MSE calculation
3. 2D embedding with UMAP and/or PHATE
4. clustering with KMeans and/or DBSCAN
5. merging labels with latent features
6. correlation and label-based visualization outputs
7. class-distribution and crosstab plots
8. per-sample reconstruction error export

The wrapper entry point is `scripts/run_analysis_pipeline.py`, which reads a YAML config and launches the analysis run.

### Classification Utilities

`subcellae/classification/classification.py` provides helpers for:

- merging labels with latent features
- train/validation splitting
- classifier training
- evaluation metrics
- confusion matrix plotting

Supported classifiers currently include:

- logistic regression
- random forest
- gradient boosting

### Label Processing Scripts

The `scripts/labels/` folder contains small scripts for converting and merging external annotation CSVs into a standardized format with fields such as:

- `unique_ID`
- `crop_img_filename`
- `group`
- `group_ID`
- `Position`
- `classification`

These scripts are written for a specific local labeling workflow and use hard-coded input/output paths.

## Running Analyses

The project includes a Makefile with shortcuts for creating environments and running named analysis jobs.

Useful targets:

```bash
make env
make env-cuda
make env-update
make notebook
make help
```

The analysis runner is intended to be used like this:

```bash
python scripts/run_analysis_pipeline.py --config config/analysis_config.yaml
```

The example config in `config/analysis_config.yaml` defines:

- model checkpoint path
- model parameters such as latent dimension
- dataset location
- dataloader settings
- output directory
- optional label CSV
- embedding settings
- clustering settings
- optional label-order configuration

## Current Status Notes

This repository is usable as a research codebase, but a few parts look mid-refactor and are worth knowing before you start:

- `README.md` was originally just a stub; this file reflects the current code layout.
- `scripts/run_analysis_pipeline.py` imports from `core.*`, but the package in this repo is named `subcellae.*`.
- `subcellae/pipeline/analysis_pipeline.py` imports from `utils.*`, while the matching modules currently live under `subcellae.analysis`, `subcellae.clustering`, and `subcellae.utils`.
- `Makefile` analysis targets point to `configs/...`, but the repository currently contains `config/analysis_config.yaml`.
- several scripts contain absolute local paths and appear to be project-specific rather than fully portable command-line tools

If you plan to run the full pipeline on a fresh machine, those import/path mismatches will likely need a cleanup pass first.

## Development

The package metadata in `pyproject.toml` is minimal:

- package name: `subcellvae`
- version: `0.1.0`

Because the environment files install the project with `pip -e .`, local edits are immediately reflected in the active environment.

## License

This project is released under the MIT License. See `LICENSE`.
