"""Readable implementation of the bounded Subjectesis state update.

The language model proposes an update. The controller is deliberately simple:
it checks the proposal, updates only the selected field, and stops after one
bounded review. The controller does not invent semantic content by itself.
"""


def is_valid_update(update):
    """Check that a proposed review update contains the minimum required data."""
    if not isinstance(update, dict):
        return False

    field = update.get("selected_field")
    claim = update.get("updated_claim")

    if not isinstance(field, str) or not field.strip():
        return False

    if claim is None:
        return False

    return True


def apply_update(state, update):
    """Apply one valid field update and otherwise keep the original state."""
    new_state = dict(state)

    if not is_valid_update(update):
        return new_state

    field = update["selected_field"]
    new_state[field] = update["updated_claim"]

    return new_state


def bounded_review(initial_state, review_function):
    """Run at most one review step.

    review_function receives the current state and returns a proposed update.
    The function returns both the final state and the proposal so the review
    process can be audited later.
    """
    proposal = review_function(dict(initial_state))
    final_state = apply_update(initial_state, proposal)

    return {
        "initial_state": initial_state,
        "review_proposal": proposal,
        "final_state": final_state,
    }
