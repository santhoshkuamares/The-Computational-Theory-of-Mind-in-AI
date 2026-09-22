# Computational Theory of Mind in AI

This repository contains the code and research artifacts for a master's dissertation investigating a cognitive-science-inspired approach to computational Theory of Mind.

The project operationalises Subjectesis as structured supervision over perspective, evidence and bounded revision, then evaluates whether this changes Theory-of-Mind performance relative to ordinary answer-only fine-tuning.

## Research questions

The main research question is:

> Does training a language model with Subjectesis-style perspective, evidence and revision supervision improve computational Theory-of-Mind performance, and does executing its bounded review procedure provide an additional benefit?

The transfer question is:

> Do these effects extend to a different Theory-of-Mind benchmark without additional fine-tuning?

## Experimental design

The experiments use Qwen3.5-4B with NF4 QLoRA adapters.

Five evaluation conditions are compared:

1. Base model
2. Answer-only QLoRA
3. Subjectesis QLoRA with direct answering
4. Subjectesis QLoRA with an explicit perspective state and no review
5. Subjectesis QLoRA with the perspective state and one bounded review step

The in-domain evaluation uses RecToM. Transfer is evaluated on a fixed OpenToM subset of 27 stories and 621 questions without OpenToM fine-tuning.

## Important training detail

Both training runs completed the same full schedule of 30,266 examples. Validation selected different checkpoints for final evaluation:

- Answer-only: step 1,000, approximately 8,000 training-example exposures
- Subjectesis: step 2,500, approximately 20,000 training-example exposures

The evaluated checkpoints are therefore not exposure-matched. Direct differences between the selected adapters should not be attributed only to supervision structure.

The Subjectesis condition also contains substantially more supervised target tokens because its examples include structured state and review targets.

## Current benchmark results

### RecToM sealed test

| System | Belief accuracy | Desire accuracy | Overall accuracy |
| --- | ---: | ---: | ---: |
| Base | 0.6734 | 0.7964 | 0.7349 |
| Answer-only | 0.9364 | 0.9600 | 0.9482 |
| Subjectesis direct | 0.9220 | 0.9709 | 0.9464 |
| Subjectesis structured, no review | 0.9277 | 0.9600 | 0.9439 |
| Subjectesis structured, review | 0.9277 | 0.9600 | 0.9439 |

Both fine-tuning conditions produced large in-domain gains. The paired comparison between Subjectesis direct and answer-only did not show a clear difference. The bounded review step did not change final RecToM answers on the sealed test.

### OpenToM frozen transfer subset

The corrected evaluator scores all 621 questions.

| System | Accuracy | Mean family macro-F1 |
| --- | ---: | ---: |
| Base | 0.5346 | 0.4559 |
| Answer-only | 0.5250 | 0.4299 |
| Subjectesis direct | 0.5185 | 0.4261 |
| Subjectesis structured, no review | 0.5330 | 0.4501 |
| Subjectesis structured, review | 0.5233 | 0.4415 |

The transfer results are mixed. The in-domain RecToM gains do not robustly transfer to the selected OpenToM stories. The structured no-review condition numerically recovers some performance relative to Subjectesis direct, while review is not consistently helpful.

The OpenToM values above use the corrected evaluator with explicit metadata keys. Earlier values produced by the superseded evaluator are not used.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── config/
│   └── experiment.json
├── src/
│   ├── data_io.py
│   ├── training_config.py
│   ├── subjectesis_controller.py
│   └── statistics.py
├── docs/
│   ├── architecture.md
│   ├── methodology.md
│   └── reproducibility.md
├── results/
│   ├── rectom_summary.csv
│   └── opentom_summary.csv
└── notebooks/
    └── cognitive_representational_analysis.ipynb
```

The repository is intentionally kept small. Temporary debugging scripts, interrupted runs, model checkpoints and superseded evaluation artifacts are not included.

## What each code file does

### `src/data_io.py`

Contains small helper functions for reading JSON, JSONL and CSV data. Keeping input/output code separate makes the analysis scripts easier to read.

### `src/training_config.py`

Stores the final QLoRA settings used for the two completed training conditions. It also documents the validation-selected checkpoints and the effective number of examples seen by those checkpoints.

### `src/subjectesis_controller.py`

Provides a readable implementation of the bounded Subjectesis update rule used during structured evaluation. The controller validates a proposed state update, replaces only the selected field, and performs at most one review step.

### `src/statistics.py`

Contains the clustered bootstrap utilities used to estimate uncertainty while resampling whole dialogues or whole stories instead of treating every question as statistically independent.

### `notebooks/cognitive_representational_analysis.ipynb`

Runs the exploratory post-hoc analysis of internal representations. It includes layer-wise linear probes, representational similarity analysis, monitoring/control analysis and a controlled perspective-access diagnostic. It does not retrain the language model.

## Scope

This work studies computational Theory of Mind. It does not claim that the model is conscious, self-aware, biologically equivalent to a person, or a neural model of the human brain.

The cognitive-science concepts are used as functional design and analysis tools. Neuroscience-inspired representational methods are used to study the model without claiming anatomical correspondence between transformer layers and brain regions.

## Model

- Base model: `Qwen/Qwen3.5-4B`
- Pinned model revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- Training method: NF4 QLoRA
- LoRA rank: 8
- LoRA alpha: 16
- Learning rate: 1e-4
- Effective batch size: 8
- Epochs: 1
- Random seed: 42

Large adapter and optimizer files are not stored in the Git repository. The repository records the configuration and results needed to identify the runs without turning the source repository into checkpoint storage.

## Status

The main training, RecToM evaluation and corrected OpenToM transfer evaluation are complete. The representational analysis is exploratory and is kept separate from the confirmatory benchmark results.
