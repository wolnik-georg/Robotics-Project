#!/usr/bin/env python3
"""
Example: How to add confidence-based prediction filtering to discrimination_analysis.py

This shows how to:
1. Get prediction probabilities
2. Filter low-confidence predictions
3. Report statistics
4. Integrate into existing metrics

Add this helper function to discrimination_analysis.py around line 400 (before run() method)
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def apply_confidence_filtering(
    y_true,
    y_pred,
    probabilities,
    threshold=0.7,
    mode="reject",
    default_class=None,
    logger=None,
):
    """
    Filter predictions based on confidence threshold.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        probabilities: Prediction probabilities from predict_proba (shape: [n_samples, n_classes])
        threshold: Minimum confidence to accept (0.0-1.0)
        mode: "reject" (exclude low-confidence) or "default" (assign default class)
        default_class: Default class to use if mode="default"
        logger: Logger for reporting statistics

    Returns:
        Tuple of (filtered_y_true, filtered_y_pred, confidence_stats)
    """
    # Get maximum probability for each prediction (confidence)
    confidences = np.max(probabilities, axis=1)

    # Identify high-confidence predictions
    high_confidence_mask = confidences >= threshold

    # Statistics
    total_samples = len(y_pred)
    high_conf_count = np.sum(high_confidence_mask)
    low_conf_count = total_samples - high_conf_count

    stats = {
        "total_samples": total_samples,
        "high_confidence": high_conf_count,
        "low_confidence": low_conf_count,
        "high_confidence_pct": 100 * high_conf_count / total_samples,
        "low_confidence_pct": 100 * low_conf_count / total_samples,
        "mean_confidence": np.mean(confidences),
        "median_confidence": np.median(confidences),
        "min_confidence": np.min(confidences),
        "max_confidence": np.max(confidences),
    }

    # Apply filtering based on mode
    if mode == "reject":
        # Reject low-confidence predictions (exclude from evaluation)
        filtered_y_true = y_true[high_confidence_mask]
        filtered_y_pred = y_pred[high_confidence_mask]

        if logger:
            logger.info(f"  📊 Confidence Filtering (threshold={threshold}):")
            logger.info(
                f"    Kept: {high_conf_count}/{total_samples} ({stats['high_confidence_pct']:.1f}%)"
            )
            logger.info(
                f"    Rejected: {low_conf_count}/{total_samples} ({stats['low_confidence_pct']:.1f}%)"
            )
            logger.info(f"    Mean confidence: {stats['mean_confidence']:.3f}")

    elif mode == "default":
        # Assign default class to low-confidence predictions
        filtered_y_true = y_true.copy()
        filtered_y_pred = y_pred.copy()
        filtered_y_pred[~high_confidence_mask] = default_class

        if logger:
            logger.info(f"  📊 Confidence Filtering (threshold={threshold}):")
            logger.info(
                f"    High confidence: {high_conf_count}/{total_samples} ({stats['high_confidence_pct']:.1f}%)"
            )
            logger.info(
                f"    Defaulted to '{default_class}': {low_conf_count}/{total_samples} ({stats['low_confidence_pct']:.1f}%)"
            )
            logger.info(f"    Mean confidence: {stats['mean_confidence']:.3f}")

    else:
        # No filtering
        filtered_y_true = y_true
        filtered_y_pred = y_pred

    return filtered_y_true, filtered_y_pred, stats


# ============================================================================
# INTEGRATION EXAMPLE: Replace prediction code in discrimination_analysis.py
# ============================================================================


def example_integration():
    """
    Example of how to integrate confidence filtering into the existing code.

    REPLACE lines 1000-1010 in discrimination_analysis.py with this pattern:
    """

    # BEFORE (current code):
    # y_train_pred = clf.predict(X_train_scaled)
    # train_accuracy = accuracy_score(y_train_split, y_train_pred)
    # train_f1 = f1_score(y_train_split, y_train_pred, average="weighted")

    # AFTER (with confidence filtering):
    """
    # Get predictions AND probabilities
    y_train_pred = clf.predict(X_train_scaled)
    y_train_proba = clf.predict_proba(X_train_scaled)
    
    # Apply confidence filtering (if enabled)
    conf_config = self.config.get("confidence_filtering", {})
    if conf_config.get("enabled", False):
        threshold = conf_config.get("threshold", 0.7)
        mode = conf_config.get("mode", "reject")
        default_class = conf_config.get("default_class", "no_contact")
        
        y_train_filtered, y_train_pred_filtered, train_conf_stats = apply_confidence_filtering(
            y_train_split,
            y_train_pred,
            y_train_proba,
            threshold=threshold,
            mode=mode,
            default_class=default_class,
            logger=self.logger,
        )
    else:
        # No filtering
        y_train_filtered = y_train_split
        y_train_pred_filtered = y_train_pred
        train_conf_stats = None
    
    # Calculate metrics on filtered predictions
    train_accuracy = accuracy_score(y_train_filtered, y_train_pred_filtered)
    train_f1 = f1_score(y_train_filtered, y_train_pred_filtered, average="weighted")
    
    # Repeat for test and validation sets...
    """
    pass


# ============================================================================
# EXAMPLE: Confidence Analysis Script
# ============================================================================


def analyze_confidence_distribution(model, X, y, class_names):
    """
    Analyze confidence distribution to help choose a good threshold.

    Usage: Run this after training to see confidence distributions.
    """
    # Get predictions and probabilities
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    # Get confidence (max probability)
    confidences = np.max(y_proba, axis=1)

    # Check which predictions are correct
    correct = y_pred == y

    print("\n" + "=" * 70)
    print("CONFIDENCE DISTRIBUTION ANALYSIS")
    print("=" * 70)

    print(f"\nOverall Statistics:")
    print(f"  Total samples: {len(y)}")
    print(f"  Accuracy: {np.mean(correct):.1%}")
    print(f"  Mean confidence: {np.mean(confidences):.3f}")
    print(f"  Median confidence: {np.median(confidences):.3f}")
    print(f"  Std confidence: {np.std(confidences):.3f}")

    print(f"\nCorrect vs Incorrect Predictions:")
    print(
        f"  Correct predictions - mean confidence: {np.mean(confidences[correct]):.3f}"
    )
    print(
        f"  Incorrect predictions - mean confidence: {np.mean(confidences[~correct]):.3f}"
    )

    print(f"\nConfidence Bins:")
    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(len(bins) - 1):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if np.sum(mask) > 0:
            acc = np.mean(correct[mask])
            count = np.sum(mask)
            pct = 100 * count / len(y)
            print(
                f"  [{bins[i]:.1f}-{bins[i+1]:.1f}): {count:4d} samples ({pct:5.1f}%) - Accuracy: {acc:.1%}"
            )

    print(f"\nRecommended Thresholds:")
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        mask = confidences >= threshold
        kept = np.sum(mask)
        kept_pct = 100 * kept / len(y)
        if kept > 0:
            acc = np.mean(correct[mask])
            print(
                f"  {threshold:.1f}: Keep {kept:4d} ({kept_pct:5.1f}%) - Accuracy: {acc:.1%}"
            )

    print("=" * 70)


if __name__ == "__main__":
    # This is just documentation - see the functions above for implementation
    print(__doc__)
    print("\nKey Functions:")
    print("1. apply_confidence_filtering() - Main filtering function")
    print(
        "2. analyze_confidence_distribution() - Analyze confidence to choose threshold"
    )
    print("\nTo integrate:")
    print("1. Add apply_confidence_filtering() to discrimination_analysis.py")
    print("2. Update prediction code to use probabilities")
    print("3. Apply filtering before calculating metrics")
    print("4. Enable in config: confidence_filtering.enabled = true")
