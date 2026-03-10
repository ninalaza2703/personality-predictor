def mbti_accuracies(true_labels, predicted_labels):
    """
    Calculates MBTI classification accuracy at various levels of match granularity.

    Parameters:
        true_labels (list of str): Ground truth MBTI types (4-letter strings).
        predicted_labels (list of str): Predicted MBTI types (4-letter strings).

    Returns:
        dict: Dictionary with accuracy scores for:
              - Exact 4-letter match
              - At least 3-letter match
              - At least 2-letter match
              - At least 1-letter match
    """
    assert len(true_labels) == len(predicted_labels), "Length mismatch between true and predicted labels"

    total = 0
    match_counts = [0, 0, 0, 0, 0]  # Index i stores count of predictions with i letter matches

    for true, pred in zip(true_labels, predicted_labels):
        # Skip invalid entries
        if true is None or pred is None:
            continue
        if not isinstance(true, str) or not isinstance(pred, str):
            continue
        if len(true) != 4 or len(pred) != 4:
            continue

        matches = sum(t == p for t, p in zip(true, pred))
        match_counts[matches] += 1
        total += 1

    if total == 0:
        # Avoid division by zero; return 0.0 for all metrics
        return {
            "Exact 4-letter match": 0.0,
            "At least 3-letter match": 0.0,
            "At least 2-letter match": 0.0,
            "At least 1-letter match": 0.0,
        }

    return {
        "Exact 4-letter match": match_counts[4] / total,
        "At least 3-letter match": sum(match_counts[3:]) / total,
        "At least 2-letter match": sum(match_counts[2:]) / total,
        "At least 1-letter match": sum(match_counts[1:]) / total,
    }
