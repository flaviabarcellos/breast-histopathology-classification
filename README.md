# Breast Histopathology Image Classification

A data science project for classifying BreaKHis breast histopathology images as **benign** or **malignant**, using an EfficientNetV2B1 backbone and a custom Inception classification head.

The implementation is adapted from `03_PIM_efficient_net_v2_architecture_model_breakhis_classification.ipynb`. It organizes the original experiment into a Python package with patient-based dataset splits, configurable training, saved checkpoints, evaluation reports, and single-image prediction.

## Features

- Binary classification with a fixed class mapping: `benign = 0`, `malignant = 1`.
- Independent experiments at 40×, 100×, 200×, and 400× magnification.
- Stratified train, validation, and test splits grouped by patient.
- ImageNet transfer learning followed by three fine-tuning stages.
- Training augmentation and deterministic preprocessing during evaluation.
- Model selection based on validation loss, with a separate test evaluation command.
- Image-level and patient-level metrics, prediction tables, and evaluation plots.
- Dependencies managed with Poetry and exported to `requirements.txt`.

## Repository structure

```text
breast-histopathology-classification/
├── .github/workflows/checks.yml   # Automated dependency and test checks
├── configs/
│   └── default.json              # Dataset paths and experiment settings
├── data/
│   ├── raw/                      # Original dataset images
│   └── processed/                # Generated dataset manifests
├── docs/
│   └── notebook-migration.md     # Differences from the source notebook
├── models/                       # Checkpoints, logs, and run artifacts
├── notebooks/                    # Exploratory analysis workspace
├── reports/figures/              # Figures selected for reporting
├── src/breakhis/
│   ├── cli.py                    # Command line interface
│   ├── data.py                   # Image indexing and patient splits
│   ├── evaluate.py               # Metrics, predictions, and plots
│   ├── model.py                  # Architecture and fine-tuning setup
│   └── train.py                  # Training and checkpoint selection
├── tests/                        # Data and model regression tests
├── poetry.lock                   # Resolved dependency versions
├── poetry.toml                   # Local virtual environment settings
├── pyproject.toml                # Package metadata and dependencies
└── requirements.txt              # Exported dependencies for pip
```

Dataset images, generated artifacts, local tools, and caches are excluded from Git.

## Installation

Use **Python 3.10–3.12** and **Poetry 2.2 or newer**. Python 3.11 is the version used for local verification. Run the following commands from the repository root.

### Poetry

```bash
poetry env use python3.11
poetry install --all-groups
```

This installs the project together with its runtime, development, and notebook dependencies. Poetry creates the virtual environment in `.venv/`.

In the existing workspace, Poetry is installed at `.tools/poetry/bin/poetry`, and a Python 3.11 interpreter is available under `.tools/python/`. If `poetry` is not on your PATH, use the local executable:

```bash
.tools/poetry/bin/poetry run pytest
```

Those local tools are not included in Git. A fresh clone requires its own Python and Poetry installation.

### pip

For an environment that uses pip, install the exported dependencies and the project:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

The activation command above is for macOS and Linux. On Windows, use `.venv\Scripts\Activate.ps1` in PowerShell.

## Dataset preparation

Obtain BreaKHis separately and extract its images into `data/raw/`, or point `data_dir` to an existing dataset directory. Native nested directories and reorganized directories are supported as long as the original image filenames remain intact.

Example:

```text
data/raw/
└── .../
    ├── SOB_B_A-14-22549AB-40-001.png
    └── SOB_M_DC-14-2985-40-001.png
```

The loader reads diagnosis, patient identity, and magnification from each filename. Unrecognized PNG filenames cause an error instead of assigning an assumed patient identity.

Patients are stratified by diagnosis and assigned to one split only. The default proportions are **70% training**, **10% validation**, and **20% test**. These proportions apply to patients; image proportions can differ because patients have different numbers of images. A fixed seed makes the split reproducible for the same input population.

Generate and inspect a manifest before training:

```bash
poetry run python -m breakhis.cli prepare --config configs/default.json
```

This prints image counts by split and class and saves `data/processed/manifest_40X.csv`. Confirm the filename-derived patient identifiers against your dataset metadata before using the split for an experiment.

## Experiment configuration

Edit [configs/default.json](configs/default.json) before starting a run.

| Setting | Default | Purpose |
| --- | --- | --- |
| `data_dir` | `data/raw` | Directory searched recursively for PNG images |
| `output_dir` | `models/40X` | Directory created for the training run |
| `magnification` | `40` | One of `40`, `100`, `200`, or `400` |
| `seed` | `42` | Split and training random seed |
| `image_size` | `[224, 340]` | Input resize dimensions, height then width |
| `crop_size` | `[224, 224]` | Dimensions presented to the backbone |
| `batch_size` | `128` | Number of images per batch |
| `train_fraction` | `0.7` | Fraction of patients used for training |
| `validation_fraction` | `0.1` | Fraction of patients used for validation |
| `test_fraction` | `0.2` | Fraction of patients reserved for testing |
| `epochs_per_stage` | `50` | Maximum epochs in each training stage |
| `patience` | `10` | Early stopping patience on validation loss |
| `weights` | `imagenet` | Backbone initialization |
| `stages` | Head, then progressively unfrozen blocks | Trainable backbone blocks per stage |

Paths are resolved relative to the working directory. Use a new `output_dir` for each experiment: training refuses to overwrite an existing run directory. Reduce `batch_size` when memory is limited. To run another magnification, change both `magnification` and `output_dir`.

ImageNet initialization downloads pretrained weights on first use. Setting `weights` to `null` allows an offline architecture check with random weights and changes the experiment.

## Model architecture

```text
RGB image resized to 224 × 340
                ↓
Crop to 224 × 224 and training augmentation
                ↓
EfficientNetV2B1 through block3c_add
                ↓
Parallel Inception branches
├── 1 × 1 convolution: 64 filters
├── 3 × 3 convolution: 128 filters
├── 5 × 5 convolution: 32 filters
└── 3 × 3 max pooling
                ↓
Concatenation → Batch normalization → Flatten
                ↓
Dropout (0.4) → Dense (2, softmax)
```

Training augmentation includes crop, horizontal and vertical flips, rotation, translation, and zoom. An explicit center crop ensures the expected input dimensions during inference. Input pixels remain in the `[0, 255]` range because the EfficientNetV2 backbone includes preprocessing.

Training proceeds through four stages:

1. Train the classification head with the backbone frozen.
2. Unfreeze backbone layers in `block3`.
3. Unfreeze backbone layers in `block2` and `block3`.
4. Unfreeze backbone layers in `block1`, `block2`, and `block3`.

Backbone batch normalization remains frozen. Each stage recompiles the optimizer and reloads its best validation-loss checkpoint. The final checkpoint is selected by the lowest validation loss across stages. Test data does not participate in this selection.

## Training

```bash
poetry run python -m breakhis.cli train --config configs/default.json
```

Training regenerates the patient split and saves the exact manifest and configuration in the run directory. With the same input population, configuration, and seed, this matches the preparation command's split. Hardware and library differences can still affect training results.

## Evaluation

Evaluate the selected checkpoint on the saved test split:

```bash
poetry run python -m breakhis.cli evaluate --run models/40X
```

Evaluation reports accuracy, balanced accuracy, ROC AUC, and per-class precision, recall, and F1-score. Patient-level probabilities are calculated by averaging image probabilities for each patient at the selected magnification. The malignant decision threshold is fixed at `0.5`.

## Single-image prediction

```bash
poetry run python -m breakhis.cli predict \
  --run models/40X \
  --image /path/to/image.png
```

The command loads `best.keras`, applies the saved resize configuration and model preprocessing, and prints a JSON object with `benign` and `malignant` probabilities. Use images consistent with the run's magnification and dataset conditions.

## Generated artifacts

```text
models/40X/
├── config.json
├── class_names.json
├── manifest.csv
├── selection.json
├── best.keras
├── stage_0/                      # Additional stage directories follow
│   ├── best.keras
│   ├── history.csv
│   └── tensorboard/
└── evaluation/                   # Created by the evaluation command
    ├── metrics.json
    ├── predictions.csv
    ├── patient_predictions.csv
    ├── confusion_matrix.png
    └── roc_curve.png
```

View training logs with:

```bash
poetry run tensorboard --logdir models/40X
```

## Notebooks

Launch JupyterLab after installing the notebook dependency group:

```bash
poetry run jupyter lab notebooks/
```

Import reusable functions from `breakhis` rather than duplicating the pipeline. See the [notebook guidelines](notebooks/README.md). The original notebook remains at its supplied location and is not bundled in this repository.

## Tests and code checks

```bash
poetry run pytest
poetry run pytest tests/test_data.py -q
poetry run ruff check src tests
poetry check --lock
```

Tests cover filename parsing, invalid inputs, disjoint and reproducible patient splits, rectangular-image inference, fine-tuning activation, and model serialization. The model regression test uses random weights and does not download ImageNet weights. GitHub Actions installs dependencies from Poetry and runs the test suite.

## Dependency updates

Poetry is the source of dependency definitions and resolved versions. After changing dependencies, update `poetry.lock` and regenerate `requirements.txt` with the Poetry export plugin:

```bash
poetry self add poetry-plugin-export
poetry export --all-groups --format requirements.txt --without-hashes --output requirements.txt
```

Commit `pyproject.toml`, `poetry.lock`, and the regenerated `requirements.txt` together.

## Experiment status and limitations

The environment, tests, inference path, and model serialization have been verified locally. Full dataset training and test-set evaluation have not been performed in this repository, so no classification performance is reported.

The implementation differs from the original notebook in split construction, deterministic evaluation, checkpoint selection, fine-tuning behavior, and augmentation. Shear augmentation is omitted. See [notebook migration notes](docs/notebook-migration.md) before comparing results with earlier experiments.

The pipeline does not currently provide cross-validation, confidence intervals, image-content duplicate detection, or automatic training resume. It is intended for research; clinical validation is outside the project's scope.
