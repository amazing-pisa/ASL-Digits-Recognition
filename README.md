# ASL digit recognition with dual CNN branches

Research code, saved dynamic features, fixed data splits, trained models and machine-readable dynamic model-comparison/ablation records for ASL digits 0–20.

## Structure

```text
models/                 static and dynamic architecture definitions
scripts/core/           extraction, static training/evaluation and live routing
scripts/experiments/    dynamic training, ablation and analysis implementations
data/splits/            exact saved-sequence memberships and source hashes
data/dataset/           dynamic feature arrays after artifact extraction
configs/                model, system and recorded training configurations
results/                dynamic architecture-comparison and Delta-ablation records
weights/                model checkpoints after artifact extraction
docs/                   data schemas and reproducibility scope
```

## Data and installation

Extract `dynamic-data.zip` and `model-weights.zip` from the same versioned release into the repository root. The dynamic data contain 976 saved sequences with 30×212 input features and labels. The fixed class-stratified split is **682 training / 98 validation / 196 test**, with split seed **42**. Repeated training uses seeds **2021, 2022 and 2023**. Exact sample identities are in `data/splits/split_manifest.csv`.

Use Python 3.12 and the recorded CPU dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py validate
```

Paths are repository-relative. Optional `ASL_ARTIFACT_ROOT` can point to a separate directory containing `data/dataset/` and `weights/`.

## Evaluation and training

```bash
python reproduce.py evaluate --family architectures
python reproduce.py evaluate --family ablation
python reproduce.py train --family architectures
python reproduce.py train --family ablation
```

Evaluation calculates accuracy and Macro-F1 and checks restored checkpoint predictions against archived per-sample records. Per-run confusion matrices and classification reports are retained under `results/`. New evaluation reports go to `reproduced/`; fresh training writes to each experiment's `outputs/` directory.

The recorded training configuration is Adam learning rate 0.001, batch size 32, at most 200 epochs, learning-rate reduction factor 0.5 with patience 6, early stopping patience 20, and minimum-validation-loss checkpoint selection. Full settings are in `configs/training.json`; exact saved dynamic model configurations are in `configs/models/`. Only training data are augmented. Gaussian noise applies to all feature channels, including reserved zero-valued visibility fields and their Delta channels; scaling applies to the first 84 dimensions. Validation, test and live inputs are not augmented.

## Feature extraction and live use

```bash
python scripts/core/extract_features.py input.mp4 output.npy --digit 10
python reproduce.py camera
```

The live system retains the fixed threshold 0.0005 configured in `configs/system.json`. Static source images are not redistributed; a matching authorized copy is required at `data/Sign-Language-Digits-Dataset/{train,valid,test}/`. Membership and hashes are supplied in `data/splits/static_image_manifest.csv`. Run static train/evaluation scripts from `scripts/core/`.

## Distribution scope

No recorded motion-routing or latency-session videos, photos, audio, logs, annotations, schedules, onset labels, summary results or derived learned-routing outputs/models are included. Their recording/analysis implementations remain available for user-supplied sessions. The reported values from those withheld sessions cannot be recalculated from this distribution alone.

For your own records, use:

```bash
python scripts/experiments/motion_validation/threshold_sensitivity.py --session YOUR_SESSION
python scripts/experiments/latency_validation/latency_test.py analyze --session YOUR_SESSION
```

The optional supervised routing implementation is retained as method code and requires a separately supplied annotated session. It is not integrated into the fixed-threshold live application.

Numerical tables and interpretation are presented in the manuscript. `results/` retains underlying dynamic model-comparison and ablation records for verification. See `docs/DATA.md` and `docs/REPRODUCIBILITY.md` for the scope, `CHECKSUMS.csv` for repository files and `CHECKSUMS.csv` for the release assets.

The provider has authorized public release of the supplied materials. No standard reuse license has been specified; see LICENSE and LICENSE_NOTICE.md. Add actual repository, release and archive links after they exist.

## Compact archived records

Per-epoch CSV/JSON logs and runtime status snapshots are omitted. `results/training_summary.csv` preserves the actual epoch count, selected epoch, minimum validation loss, validation accuracy and learning rate at that epoch, runtime, parameter count and final test metrics for all 18 repeated runs. Training controls remain in `configs/training.json` and the archived protocols. New training still writes its own full logs.

All distributed sample predictions are **test-set only**, with 196 sequences per run. Each distributed repeated-run checkpoint is the final validation-selected checkpoint, not a collection of intermediate epoch weights. FNN and VGG16 remain because they are static comparison models in the manuscript. The static image membership/hash manifest is retained to identify the exact externally supplied images.

`CHECKSUMS.csv` is the single combined repository/artifact checksum table. Its `location` column distinguishes repository files from the two ZIP assets. The operational `data/splits/source_data_manifest.csv` remains separate because loaders use its source filenames, sequence counts and dataset hashes.
