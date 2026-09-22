# Subjectesis architecture

The implemented system separates the language model from the deterministic
controller. The model interprets the dialogue and proposes state content. The
controller checks structure, applies a bounded update, and controls stopping.

```mermaid
flowchart LR
    A[Dialogue or story] --> B[Question]
    B --> C[Qwen3.5-4B]
    C --> D[Perspective state]
    D --> E[Identify selected field]
    E --> F[Retrieve relevant evidence]
    F --> G[Propose review update]
    G --> H[Deterministic validation]
    H --> I[Apply valid update or preserve state]
    I --> J[Final answer]
```

## Training and evaluation relationship

```mermaid
flowchart TD
    R[RecToM training split] --> S[Subjectesis structured supervision]
    R --> A[Answer-only supervision]

    S --> SQ[Subjectesis QLoRA adapter]
    A --> AQ[Answer-only QLoRA adapter]

    B[Base Qwen3.5-4B] --> E[Evaluation]
    SQ --> E
    AQ --> E

    E --> RT[RecToM sealed test]
    E --> OT[OpenToM frozen transfer]

    RT --> P[Behavioral and process analysis]
    OT --> P
    P --> M[Exploratory representation analysis]
```

## Cognitive-science interpretation

The design is motivated by a functional distinction between monitoring and
control. The perspective state records what is currently represented. Review
checks a selected part of that state against supplied evidence. The controller
then either preserves or updates the selected field.

This is a computational analogy, not a claim that transformer components map
onto particular brain regions. The dissertation treats perspective tracking,
monitoring and revision as functional constructs that can be implemented and
tested in an AI system.
