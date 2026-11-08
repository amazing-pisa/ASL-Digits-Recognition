# Data description

## Saved dynamic sequences

The 11 original `seq_*.npy` files contain 976 sequences. Each array has shape `(N,30,213)`. Columns 0–211 are model input features and column 212 repeats the zero-based class label at every frame. Digit = label + 10. The features contain 106 base dimensions and 106 Delta dimensions. Base dimensions comprise 84 joint channels (21 × x/y/z/visibility), 15 angles and seven geometric quantities.

No video frames, face pictures or audio are distributed. Hand-derived coordinates and motion information are still present; these files should not be described as guaranteeing complete anonymity.

`data/splits/split_manifest.csv` identifies every saved sequence by source filename and index. It fixes 682 training, 98 validation and 196 test sequences, with class-stratified split seed 42. No additional windows are generated across splits. Data were collected from one participant; no recording-session grouping is available. The split is sequence-based rather than participant- or session-independent.

## Recorded system-level evaluation data

The recorded motion-routing and latency sessions are not included in this distribution. Their frames, human state labels, onset annotations, trial schedules, timing logs, confusion matrices, summary reports and derived learned-routing outputs are excluded. Analysis and recording implementations remain under `scripts/experiments/` and can process user-supplied sessions. Published values from the withheld sessions cannot be regenerated from this package alone.

## Availability statement for a published release

After the actual repository and archive URLs exist, the manuscript can state that feature-level dynamic data, fixed splits, model code, extraction and routing implementations, seeds, checkpoints, model-comparison and ablation sample-level predictions and analysis scripts are available at those URLs. Raw recordings with identifiable appearance are not included. Do not describe feature-level release as public release of the original video dataset.

## Visibility fields and augmentation

In the 976 saved sequences, all 21 visibility fields and their Delta fields are zero before augmentation. These are reserved channels, not measured hand-visibility confidence. The training augmentation adds Gaussian noise to every feature, including these fields, and then scales the first 84 dimensions. Validation, test and live inputs are not augmented.
