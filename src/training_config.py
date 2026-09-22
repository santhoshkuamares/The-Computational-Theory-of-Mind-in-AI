"""Training settings used for the completed QLoRA experiments.

This module is documentation in executable form. It records the settings used
for both adapters and makes the checkpoint-selection difference explicit.
"""

MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

TRAINING = {
    "method": "NF4 QLoRA",
    "epochs": 1,
    "rank": 8,
    "alpha": 16,
    "learning_rate": 1e-4,
    "effective_batch_size": 8,
    "max_length": 2048,
    "seed": 42,
}

FULL_TRAINING_EXAMPLES = 30266

SELECTED_CHECKPOINTS = {
    "answer_only": {
        "step": 1000,
        "examples_seen": 8000,
        "validation_selection_score": 0.9377116308370738,
    },
    "subjectesis": {
        "step": 2500,
        "examples_seen": 20000,
        "validation_selection_score": 0.9291873375856367,
    },
}


def print_training_summary():
    """Print the settings and selected checkpoints in a readable form."""
    print("Model:", MODEL_ID)
    print("Revision:", MODEL_REVISION)
    print("Training method:", TRAINING["method"])
    print("Full examples per run:", FULL_TRAINING_EXAMPLES)

    for name, checkpoint in SELECTED_CHECKPOINTS.items():
        print(
            f"{name}: step {checkpoint['step']}, "
            f"{checkpoint['examples_seen']} examples seen"
        )


if __name__ == "__main__":
    print_training_summary()
