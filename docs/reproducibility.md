# Reproducibility notes

## Fixed model

The experiments use:

```text
Qwen/Qwen3.5-4B
revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
```

The model revision is pinned because upstream model repositories can change.

## Randomness

The main training seed is 42. Statistical bootstrap analyses also use a fixed
seed so that reported confidence intervals can be regenerated.

## Checkpoint selection

The two runs complete the same full one-epoch schedule of 30,266 examples, but
the final evaluated adapters are selected independently by validation:

```text
answer-only: step 1000, approximately 8000 examples seen
Subjectesis: step 2500, approximately 20000 examples seen
```

This distinction is important when interpreting the comparison.

## Sealed RecToM test

The development protocol was frozen before the final RecToM test:

```text
ffa671e7007de68947288d3043d7975917ec2435c4366b7c212a8e9fcbc090ef
```

Predictions were generated from answer-free inputs and frozen before final
scoring.

## OpenToM transfer

The OpenToM source repository was pinned to:

```text
3f22b66276b2d7ca5fe573c28c79cc0d077aafc5
```

The transfer protocol hash is:

```text
3310bff331d9e098b8e5393b8352f091772f20db43fe6710dd46c65b55b6687c
```

The final transfer subset contains 27 stories and 621 questions.

## Corrected OpenToM scoring

The original transfer scorer unpacked `plot_info.values()`. The current
OpenToM metadata order did not match that assumption. The corrected scorer uses
the explicit keys `mover`, `observer`, `eoi`, `original_place` and
`move_to_place`.

The correction changes reference reconstruction and scoring only. It does not
regenerate model outputs. The corrected run scores 621 of 621 questions.

## Large files

Trained adapter weights and optimizer checkpoints are not committed to this Git
repository. They are large binary artifacts and are not needed for reading the
source code or reported result tables.

If adapters are released later, they should be attached separately as release
artifacts or stored in a model repository together with their hashes and model
revision.
