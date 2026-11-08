# Results and commands

| Result | Distributed inputs | Command |
| --- | --- | --- |
| Dynamic architecture comparison | 976 feature sequences, fixed membership, nine checkpoints | `python reproduce.py evaluate --family architectures` |
| Capacity-matched Delta comparison | same features/membership, nine independent checkpoints | `python reproduce.py evaluate --family ablation` |
| New architecture training | features, fixed split seed and model definitions | `python reproduce.py train --family architectures` |
| New Delta training | features and three model-capacity settings | `python reproduce.py train --family ablation` |
| Static image comparisons | included model code/checkpoints; external images matching the manifest required | corresponding `scripts/core/evaluate_0_9_*.py` |

The routing source is `scripts/core/camera_app.py`, class `Recognizer`. It uses a ten-frame variance mean, threshold 0.0005, continuous 30-frame feature buffer, static/dynamic confidence gates 0.8/0.9 and a history of three agreeing eligible predictions. It clears history at branch changes or low confidence and resets buffers when a hand is missing. It does not perform an additional three-frame vote on routing labels.

`results/` contains the completed runs, including sample predictions, model configurations, metrics, confusion matrices and compact training summaries. Original protocol identifiers and source hashes document historical runs; published scripts have path portability changes and therefore different file hashes. Local absolute paths in exported metadata are replaced with a generic source marker. Use the release manifest for checksums of the actual distributed files.

The archived split memberships are identical across architecture and capacity experiments. Their ST-CNN means differ because these are separately trained result families. Evaluation verifies each family against its own saved predictions, without pooling seeds into independent test observations.

An unchanged feature dataset and checkpoint reproduce a saved prediction result, subject to numerical compatibility. Newly training models on another software/hardware stack can differ despite identical random seeds. The release records the chosen software versions and training controls; it does not promise identical live timings on another computer.

Recorded motion-routing and latency evaluation data are excluded from this distribution. The corresponding analysis scripts require user-supplied sessions; they do not reproduce the withheld recorded-session values without those records. Learned-routing training code is retained as optional method code, but its recorded-session outputs and five fitted fold models are not distributed here.
