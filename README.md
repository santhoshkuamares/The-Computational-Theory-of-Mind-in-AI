# The Computational Theory of Mind in AI

The repository holds the code and outputs resulting from the Subjectesis experiment `Qwen/Qwen3.5-4B`: Data Preparation, Both Training Conditions, Validation, Final RecToM Test Results, OpenToM Transfer, Statistical Analysis and Exploratory Cognitive Analysis.

This study aims to determine if Structured Perspective-State Supervision and Bounded Review can improve a Computational Theory of Mind. There were varied results. Positive Changes, Null Results and Negative Changes are all included. The Cognitive Analysis was added after the Benchmark Results and thus is exploratory.

The Python modules include the executed Notebooks, Statistical Analysis Scripts and the A100 Training Continuation. Docstrings explain what each function does, how it operates and what output it produces; comments provide explanations for particular experimental decisions. Prompt Wording, Labels and Scoring Rules correspond to those of the recorded experiment, except for the explicitly documented OpenToM Metadata Correction described below.

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

This process rebuilds revision 4 using both of the task data and splits in the bundle along with the decision points to annotate. It will run all 31 of the original preparation tests and compare them against their original hash values. A gpu is not needed for this step.

The split includes 235 dialogue's for training, 34 for validation and 67 for testing. Of the 2270 original training questions, there are 35 that have been quarantined, so only 2235 remain available as a part of the training. There are 319 available validation questions and an additional 621 questions on the sealed test.

Each of the two full training schedules (answer-only and subjectesis) include 30266 examples with equivalent exposure of each question.

| Preparation module | Explanation |
| --- | --- |
| `convert_v1.py` | Reads tasks and dialogue splits, separates public inputs from references, renders prompts and defines benchmark-state mappings. |
| `build_full.py` | Applies recorded annotations and constructs answer, state and review supervision, including preserve, revise and unknown cases. |
| `dataset_io.py` | Restricts training to eligible records and masks prompt tokens so only completion tokens contribute to loss. |
| `packets.py` | Groups source records used by the preparation and annotation process. |
| `audit_citation_controls.py` | Checks that citation presence alone does not determine a supervision target. |
| `test_full.py` | Checks splits, unchanged labels and held-out data, quarantine, evidence rules, masks and deterministic rebuilding. |

## Training

Model revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Settings: NF4 QLoRA with Rank 8, Alpha = 16, Learning Rate = 0.0001, Effective Batch Size = 8, Max Length = 2048. 
Validation belief and desire accuracy were averaged to determine which checkpoint was selected for evaluation.

Subjectesis was run on two Kaggle T4 GPUs and then moved to one Colab A100. Only Answer-Only ran on an A100. Both Subjectesis and Answer-Only retain the original checks against the environment in which they are running and install their respective library/model dependencies as previously captured. In addition, both scripts require the same host environment for Torch/CUDA. There is no replacement for a GPU environment using the CPU `requirements.txt`.

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

Changes to formatting restore byte-hashes. The historical-resume utilizes the recovered archives of exact code and maintains the original identity checks. Restoration occurs in a new working directory. No claim is made that running again with different hardware will produce identical weights.

Step 1,000 after 8,000 examples, Selected Step 2,500 after 20,000 exposures; Exposure matched full schedules are utilized, but checkpoint evaluation and target token computation are not  `training_audit.csv` records the discrepancies.

## Validation, testing and transfer

Put the completed `answer_only_a100_latest.zip` and `subjectesis_a100_latest.zip` archives in `MyDrive/Subjectesis/` for the Colab evaluation scripts:

```bash
python src/validate_rectom.py
python src/evaluate_rectom.py
python src/evaluate_opentom.py
python src/rescore_opentom.py
```

Inference is performed by all scripts except for Rescoring Script. If you want to see pre-existing results from an earlier run w/o performing inference again, you can use the CPU commands listed at the beginning of this document. Near the top of each of these scripts will be where you find your path information. A separate output path should be created when running a different experiment than was originally used in Drive. Each script has retained it's metadata save check that allows for reusing predictions.

| Script | Explanation |
| --- | --- |
| `validate_rectom.py` | Evaluates the 319 validation questions and development controller, separately from the sealed test. |
| `evaluate_rectom.py` | Runs the frozen five-condition comparison on 621 questions. Contains the actual state construction, per-field review plan, schema checks, constrained finalizer and evidence-binding audit. |
| `evaluate_opentom.py` | Selects the recorded smoke/final stories from pinned OpenToM commit `3f22b66276b2d7ca5fe573c28c79cc0d077aafc5`, then saves direct and structured predictions. |
| `rescore_opentom.py` | Rebuilds corrected metadata and references and scores already-frozen predictions on CPU, retaining prediction hash checks. Downloads the pinned OpenToM source. |

There are five conditions in RecToM (Base Direct; Answer-Only Direct; Subjectesis Direct; Subjectesis No Review; and Subjectesis Review). All of the structured conditions start with the same initial state.

RecToM has a bounded field-review plan and is not based on a single universal review call limit: 1,313 review calls were made during 621 questions, and 48 field-value changes occurred over 45 questions. There was also no final answer changes as a result of review.

Citation Binding does correspondences for text-to-turns rather than semantic entailments.

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

This is a new A100 Workflow, which will utilize existing archive files and benchmark results. 
The workflow is utilizing the same "plain RecToM" prompt for testing each of the three models, and extracting the last actual prompt token representation from the output of the embedding representations for all 32 layers. All Qwen & Adapter Layers remain constant (only the Small Linear Probe Classifier Weights are being trained.)

The system will pass through each of the following steps:

1. Create an index of 2,235 / 319 / 621 questions that are split from dialogue as separate.
2. Take the hidden states for Base, Answer only, and Subjectesis and save them for future reference.
3. Use `C=1.0`and fit a standardized, well-balanced set of logistic regression probes. Save both the layer-by-layer scores and a comparison of the final layer dialogue-bootstrap scores. 
4. Compare Cosine-Distance Geometry to Benchmark Belief-State Geometry using RSA; estimate the final layer's uncertainty by removing a single dialogue item at a time. 
5. Record (count) all monitoring/control events recorded in previously-saved files. A revision-count is not necessarily indicative of the correctness of a revision. 
6. Run 64 different information access scenarios. These cover first and second order questions and score each of the five possible inference conditions.

Representational files (hidden state arrays) can be found at `MyDrive/Subjectesis/cognitive_analysis/representations/`  however, they were not included in this version of the repository to prevent large file sizes. These representational files would have to be separately generated if you wanted to either independently fit the probes for your dataset or recompute RSA.

The remaining extracted information (and its associated analyses), question indices, results, and graphs/plots are contained within this repository. Additionally, this repository includes a CPU checking tool that ensures consistent output on all relevant results from the representation analysis without requiring another iteration of the representation analysis.

Paired accuracy in  `counterfactual_metrics.csv`, refers to pairs where both possible answers are correct. In addition, leakage is defined as determining whether an observer will be able to predict the new physical location after missing one of the moves by using the counterfactual. However, the initial diagnostic excluded missing predictions from these denominators. All saved predictions are therefore valid. The 64 cases shown here are a small, post-hoc diagnostic rather than a large, new benchmark.

The subjectesis results from probes did not indicate an advantage for the subjectesis approach across layers. Macro F1 on final layer `seen` was .02465 less than answer-only. Alignment of rsa on the final layers was greater in the saved results of the subjectesis experiment. However, neither finding supports human-like belief or correspondence between layers in transformers with areas of the brain.

## Verification of repository

All source files have been rebuilt to their original hash values and all 31 data tests pass. Numerical output from cpu benchmarks and supplemental analysis has been reproduced. All eight statistical tests are successful. Counterfactual metrics and process counts were recalculated using each individual record. Consistency of separation between the two probe splits and the final-layer table was verified.

Historical recovery audit is in `config/source_provenance.json`. This is along with historical inventory of six files that were since deleted. Entries related to these six files represent provenance, not the present set of files. Presently there are 31 python files containing 311 documented function calls. Comparison of syntax trees (excluding doc strings) indicates there are no functional changes to documentation within the update. Since gpu training/inference was not run again while verifying the repository, nor were the probe/rsa arrays recreated independently.
