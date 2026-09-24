# The Computational Theory of Mind in AI

This repository contains the code and saved outputs for the Subjectesis experiment with `Qwen/Qwen3.5-4B`: data preparation, both training conditions, validation, final RecToM testing, OpenToM transfer, statistics and exploratory cognitive analysis.

The study asks whether structured perspective-state supervision and bounded review help computational Theory of Mind. The results are mixed. Improvements, null results and adverse results are retained. The cognitive analysis was added after the benchmark results and is exploratory.

The Python modules consolidate the executed notebooks, statistical analysis scripts and A100 training continuation. Function docstrings explain their purpose, operation and outputs; comments explain experimental choices. Prompt wording, labels and scoring rules match the recorded experiment, apart from the explicit OpenToM metadata correction described below.

## Repository contents

| Location | Contents |
| --- | --- |
| `src/` | Entry scripts for training, evaluation and analysis |
| `src/preparation/` | Revision-4 preparation rules and original tests |
| `src/training/` | Model loading, loss, optimizer, checkpointing and runtime checks |
| `config/` | Settings, recorded environment and source provenance |
| `data/` | Two compressed bundles of frozen data and annotation records |
| `results/` | Benchmark scores, statistics, process counts and training audit |
| `results/cognitive/` | Probes, RSA, process summary and counterfactual predictions |
| `figures/` | Architecture diagrams and benchmark and cognitive-analysis plots |

Saved prediction records needed for CPU analysis are included. Trained adapters, optimizer checkpoints and hidden-state arrays are stored separately. The repository retains the corrected OpenToM scores and PNG figure exports.

## Reproduce results on CPU

Use Python 3.12 in a separate environment. Run from the repository root:

```bash
python -m pip install -r requirements.txt
python src/analyse.py
python src/supplementary.py
python src/test_analysis.py
python src/check_cognitive_results.py
python src/figures.py
```

The scripts extract `data/frozen_analysis_inputs.zip` into `inputs/`, rejecting existing files that differ from the frozen copies. Analysis writes to `results/`. The figure script recreates PNG, PDF and SVG files; only PNG versions are tracked.

| Script | Explanation |
| --- | --- |
| `analyse.py` | Joins predictions to references by record ID; computes accuracy and fixed-class macro F1, cluster bootstrap intervals, paired comparisons, review transitions and checkpoint exposure. |
| `supplementary.py` | Examines task/order subgroups, parser outcomes and state-answer agreement. It generates an error casebook and replays an uncertainty-triggered review rule using saved branches, without new inference. |
| `test_analysis.py` | Tests bootstrap calculations against explicit row expansion, exact paired swaps, invalid predictions and Holm correction. |
| `check_cognitive_results.py` | Recomputes counterfactual metrics and process counts from individual records. Checks probe split separation and final-layer table consistency without claiming to refit probes. |
| `figures.py` | Reads the result tables and draws the benchmark, review, exposure and architecture figures without altering scores. |
| `project_setup.py` | Extracts checked input bundles and stages readable source under the working paths expected by the original notebooks. |

There are 5,000 bootstrap repetitions, resampling dialogues for RecToM and stories for OpenToM. Paired label swaps are exact for at most 18 nonzero clusters, otherwise using 50,000 simulations. Holm correction covers ten post hoc accuracy comparisons. These intervals describe sampled-example uncertainty, not variation across independently trained seeds.

## Data preparation

```bash
python src/train_subjectesis.py --root work/subjectesis --prepare-only
```

Preparation rebuilds revision 4 from the bundled task data, split manifest and annotation decisions. It runs the 31 original preparation tests and verifies the original output hashes. No GPU is needed.

The split contains 235 training, 34 validation and 67 test dialogues. Of 2,270 original training questions, 35 are quarantined, leaving 2,235 eligible questions. Validation has 319 questions; the sealed test has 621.

Both full training schedules contain 30,266 examples with identical question-exposure counts. Answer-only repeats ordinary answer supervision. Subjectesis contains 2,235 answer examples, 2,235 state-building examples and 25,796 field-review examples.

| Preparation module | Explanation |
| --- | --- |
| `convert_v1.py` | Reads tasks and dialogue splits, separates public inputs from references, renders prompts and defines benchmark-state mappings. |
| `build_full.py` | Applies recorded annotations and constructs answer, state and review supervision, including preserve, revise and unknown cases. |
| `dataset_io.py` | Restricts training to eligible records and masks prompt tokens so only completion tokens contribute to loss. |
| `packets.py` | Groups source records used by the preparation and annotation process. |
| `audit_citation_controls.py` | Checks that citation presence alone does not determine a supervision target. |
| `test_full.py` | Checks splits, unchanged labels and held-out data, quarantine, evidence rules, masks and deterministic rebuilding. |

## Training

Model revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Settings: NF4 QLoRA, rank 8, alpha 16, learning rate 0.0001, effective batch size 8, maximum length 2,048 and seed 42. Checkpoint selection uses the mean of validation belief and desire accuracy.

Subjectesis began on two Kaggle T4 GPUs and continued on one Colab A100. Answer-only used one A100. The scripts retain the original environment checks and install their recorded model-library dependencies. They expect the corresponding host Torch/CUDA environment; the CPU `requirements.txt` is not a GPU environment replacement.

```bash
# Start in the recorded two-T4 Kaggle runtime.
python src/train_subjectesis.py --root /kaggle/working/subjectesis --mode t4

# Run the original answer-only sequence in Colab on an A100.
python src/train_answer_only.py

# Continue a historical Subjectesis archive in the recorded A100 runtime.
python src/train_subjectesis.py --root /content/subjectesis_resume --mode a100 --resume /content/drive/MyDrive/Subjectesis/subjectesis_a100_latest.zip
```

`train_subjectesis.py` stages the preparation and training modules, then runs the selected training mode. `train_answer_only.py` follows the original Colab sequence and saves recovery output to Drive. The GPU entry scripts execute stages when run and should not be imported as utilities.

| Training module | Explanation |
| --- | --- |
| `prepare.py` | Rebuilds data and checks the original hashes before training. |
| `tokenize_data.py` | Applies the recorded chat template and writes completion-masked training and validation tokens. |
| `download_model.py`, `model_contract.py` | Download and identify the pinned model and tokenizer. |
| `model_runtime.py` | Loads the text model in NF4, freezes base weights and attaches the specified LoRA parameters. |
| `engine.py` | Computes completion-only next-token loss, normalized by the total supervised-token count, including short final batches. Restores training mode before optimization. |
| `train.py` | Shuffles the fixed schedule, updates adapters, evaluates validation questions, and saves resumable checkpoints and the best adapter. Test labels do not select checkpoints. |
| `a100_answer_only.py`, `a100_subjectesis.py` | Retain the recorded single-GPU optimizer wrappers. Smaller microbatches handle memory limits while keeping the logical batch and token normalization. The Subjectesis wrapper restores gradient checkpointing. |
| `launch.py`, `setup_runtime.py` | Check and set up the original two-T4 environment, launch training, stream progress and package recovery outputs. |
| `common.py` | Provides checked data loading, hashing, batch grouping and recovery packaging. |

Formatting changes source-byte hashes. Historical resume therefore restores exact code from the verified recovery archive and retains the original identity checks. Use a fresh working directory for restoration. A fresh run on different hardware is not claimed to reproduce identical weights.

Answer-only selected step 1,000 after 8,000 example exposures; Subjectesis selected step 2,500 after 20,000 exposures. Full schedules are exposure-matched, but evaluated checkpoints and target-token compute are not. `training_audit.csv` records these differences.

## Validation, testing and transfer

Put the completed `answer_only_a100_latest.zip` and `subjectesis_a100_latest.zip` archives in `MyDrive/Subjectesis/` for the Colab evaluation scripts:

```bash
python src/validate_rectom.py
python src/evaluate_rectom.py
python src/evaluate_opentom.py
python src/rescore_opentom.py
```

All except the rescoring script perform GPU inference. To inspect existing results without inference, use the CPU commands above. Paths are near the top of each script. The original Drive destinations are retained; use separate output paths for a new experiment. Each script retains its saved-metadata checks for prediction reuse.

| Script | Explanation |
| --- | --- |
| `validate_rectom.py` | Evaluates the 319 validation questions and development controller, separately from the sealed test. |
| `evaluate_rectom.py` | Runs the frozen five-condition comparison on 621 questions. Contains the actual state construction, per-field review plan, schema checks, constrained finalizer and evidence-binding audit. |
| `evaluate_opentom.py` | Selects the recorded smoke/final stories from pinned OpenToM commit `3f22b66276b2d7ca5fe573c28c79cc0d077aafc5`, then saves direct and structured predictions. |
| `rescore_opentom.py` | Rebuilds corrected metadata and references and scores already-frozen predictions on CPU, retaining prediction hash checks. Downloads the pinned OpenToM source. |

The five conditions are Base direct, Answer-only direct, Subjectesis direct, Subjectesis no review and Subjectesis review. The structured conditions share the same initial state.

RecToM has a bounded field-review plan, not a universal one-review limit: 1,313 review calls across 621 questions, 48 field-value changes across 45 questions, and no final-answer changes. Citation binding checks text-to-turn correspondence, not semantic entailment.

OpenToM required explicit lookup of `mover`, `observer`, `eoi`, `original_place` and `move_to_place`, rather than dictionary value order. Corrected scoring includes all 621 questions across 27 stories. The old 540-scorable score is obsolete. The correction did not regenerate model answers.

## Results and output meanings

| System | RecToM correct / 621 | RecToM task-mean accuracy | OpenToM correct / 621 | Counterfactual correct / 64 |
| --- | ---: | ---: | ---: | ---: |
| Base direct | 452 | 73.49% | 332 | 40 |
| Answer-only direct | 588 | 94.82% | 326 | 58 |
| Subjectesis direct | 586 | 94.64% | 322 | 61 |
| Subjectesis no review | 585 | 94.39% | 331 | 64 |
| Subjectesis review | 585 | 94.39% | 325 | 63 |

In the system tables, `accuracy` means correct questions divided by all questions. `mean_family_accuracy` weights task families equally. RecToM task mean averages belief and desire accuracy, so it differs from pooled question accuracy. `mean_family_f1` averages the fixed-class macro F1 values across families. Invalid and out-of-target benchmark predictions remain errors.

| Output | Meaning |
| --- | --- |
| `*_systems.csv`, `*_families.csv`, `*_classes.csv` | Overall, family and class scores with relevant uncertainty estimates. |
| `paired_comparisons.csv` | Signed paired differences, cluster intervals, label-swap p-values and Holm adjustment. |
| `*_prediction_matrix.csv` | One aligned row per question for tracing scores back to predictions. |
| `*_process_records.csv`, `*_process_summary.json`, `rectom_review_events.csv` | Answer transitions, state changes, review decisions and evidence changes. OpenToM review corrects 9 answers and damages 15. |
| `*_weighting_sensitivity.csv`, `*_leave_one_cluster_out.csv` | Sensitivity to weighting and removal of individual source clusters. |
| `subgroup_metrics.csv`, `opentom_parser_audit.csv`, `rectom_state_answer_agreement.csv` | Subgroup scores, parser details and agreement between explicit states and answers. |
| `offline_review_policy_replay.csv` | Exploratory selection of saved review branches by an uncertainty rule. |
| `training_audit.csv`, `validation_history.csv` | Exposure, token counts and checkpoint selection. Duplicate resumed log steps are not counted as new optimizer steps. |
| `*_reconciliation.json`, `input_hashes.json` | Connections between recalculated metrics, frozen inputs and saved outputs. |
| `semantic_case_review/notes.json` | Recorded interpretive case notes, not independent human ratings or inter-rater reliability evidence. |

## Cognitive and representational analysis

```bash
python src/cognitive_analysis.py
```

This is the original A100 workflow using the completed archives and saved benchmark records. It uses the same plain RecToM prompt across all three model conditions and extracts the final real prompt-token representation from the embedding output and all 32 layers. Qwen and its adapters are frozen; the small linear probe classifiers are fitted.

The code proceeds through these stages:

1. Build the 2,235 / 319 / 621 question index with dialogue-separated splits.
2. Extract and save hidden states for Base, Answer-only and Subjectesis, with a memory fallback.
3. Fit standardized, balanced logistic regression probes with `C=1.0`. Save layer-wise scores and final-layer dialogue-bootstrap comparisons.
4. Compare cosine-distance geometry to benchmark belief-state geometry using RSA; estimate final-layer uncertainty by deleting one dialogue at a time.
5. Count monitoring/control events in saved records. A revision count is not itself revision correctness.
6. Run 64 information-access cases covering first- and second-order questions and score all five inference conditions.

Hidden states are saved in `MyDrive/Subjectesis/cognitive_analysis/representations/` and are not bundled in this repository. Independently refitting probes or recomputing RSA requires these arrays. The extraction and analysis code, question index, result tables and plots are included. The CPU checker verifies table consistency without repeating the representation analysis.

In `counterfactual_metrics.csv`, paired accuracy means both answers in a pair are correct. Leakage means predicting the new physical location when the target observer missed the move. The original diagnostic excludes missing predictions from these denominators; all saved predictions are valid. The 64 cases are a small post hoc diagnostic, not a large new benchmark.

The probe results do not show a uniform Subjectesis advantage: final-layer `seen` macro F1 is lower than Answer-only by about 0.02465. RSA alignment is higher in the saved Subjectesis results. Neither observation establishes human-like beliefs or a correspondence between transformer layers and brain regions.

## Repository verification

Preparation rebuilt to the original hashes and passed all 31 data tests. CPU benchmark and supplementary analyses reproduced the saved numerical outputs; all eight statistical tests passed. Counterfactual metrics and process counts were recomputed from individual records. Probe split separation and final-layer table consistency were checked.

`config/source_provenance.json` records source hashes and the historical recovery audit. Its historical inventory includes six files that were subsequently removed; those entries document provenance rather than the current file list. The current source contains 31 Python files and 311 documented functions. The recorded syntax-tree comparison, excluding docstrings, found no executable changes in the documentation update. GPU training and inference were not rerun during repository verification, and probe/RSA arrays were not independently recomputed.
