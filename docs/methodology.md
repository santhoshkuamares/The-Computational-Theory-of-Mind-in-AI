# Methodology

## 1. Data

RecToM is used for training, validation and the sealed in-domain test. The
dialogue-level split contains 235 training dialogues, 34 validation dialogues
and 67 test dialogues.

The transformed Subjectesis training set contains 30,266 examples. These are
made from direct-answer, state-building and review-state examples. The
answer-only condition also contains 30,266 examples so that the full schedules
contain the same number of example presentations.

The conditions are not token-matched. Subjectesis contains substantially more
supervised target tokens because its targets include structured state and review
content.

## 2. Model adaptation

Both conditions start from the same pinned Qwen3.5-4B checkpoint. The base
weights are frozen and adapters are trained with NF4 QLoRA.

The main settings are:

- LoRA rank 8
- LoRA alpha 16
- learning rate 1e-4
- effective batch size 8
- one epoch
- random seed 42
- maximum sequence length 2048

Validation is used for checkpoint selection. The selected answer-only checkpoint
is step 1,000 and the selected Subjectesis checkpoint is step 2,500.

This means the models used for the final comparison had seen different numbers
of examples when selected. The answer-only adapter had seen approximately
8,000 examples and the Subjectesis adapter approximately 20,000. The comparison
should therefore be described as a comparison between validation-selected
models, not as an exposure-matched checkpoint comparison.

## 3. Subjectesis execution

The structured Subjectesis condition uses a per-question state. The state is not
persistent memory across conversations.

The procedure is:

1. construct a perspective-sensitive state from the supplied text;
2. choose one field for review;
3. examine bounded evidence relevant to that field;
4. propose a revised claim;
5. validate and apply the proposed field update;
6. produce the final answer and stop.

The controller owns validation, field replacement and stopping. The language
model supplies the semantic interpretation.

## 4. RecToM evaluation

Five conditions are compared:

1. base direct;
2. answer-only direct;
3. Subjectesis direct;
4. Subjectesis structured without review;
5. Subjectesis structured with one review.

The final RecToM test contains 621 questions from 67 dialogues. Test labels are
not used for checkpoint selection, prompt tuning, controller decisions or
prediction generation. Predictions are generated first and scored afterward.

Uncertainty is estimated by resampling whole dialogues.

## 5. OpenToM transfer

No OpenToM fine-tuning is performed. A deterministic frozen subset contains
27 stories and 621 questions, retaining all 23 official questions for each
selected story.

The corrected evaluator reads the OpenToM metadata with explicit keys. This
replaces an earlier evaluator that relied on dictionary value order. The error
affected scoring metadata only; model prompts used the narrative and question,
so frozen predictions did not need to be regenerated.

All 621 corrected references are scorable.

Uncertainty is estimated by resampling whole stories.

## 6. Exploratory mechanistic analysis

After the benchmark experiments produced mixed evidence for a Subjectesis
advantage, a post-hoc exploratory analysis was added to examine what may have
changed internally.

The analysis studies:

- layer-wise linear decodability of perspective-related variables;
- representational similarity between hidden states and benchmark-encoded
  belief-state structure;
- monitoring and review-state transitions;
- controlled changes in information access.

These analyses are exploratory. They do not replace the confirmatory benchmark
results and are not presented as preregistered hypotheses.
