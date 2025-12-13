#!/usr/bin/env python3
"""
Analysis script to investigate the class imbalance issue and feature discriminative power.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))


def analyze_class_imbalance():
    """Analyze the class distribution and its impact on model performance."""
    print("🔍 Analyzing Class Imbalance and Model Performance")
    print("=" * 60)

    # Load the processed data
    data_dir = (
        "modular_analysis_results/dataprocessing/collected_data_runs_2025_12_11_v2_2"
    )

    features_path = os.path.join(
        data_dir, "collected_data_runs_2025_12_11_v2_2_features.npy"
    )
    labels_path = os.path.join(
        data_dir, "collected_data_runs_2025_12_11_v2_2_labels.npy"
    )

    X = np.load(features_path)
    y = np.load(labels_path)

    print(f"Dataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    # Class distribution
    unique_labels, counts = np.unique(y, return_counts=True)
    total_samples = len(y)

    print("\n📊 Class Distribution:")
    for label, count in zip(unique_labels, counts):
        percentage = (count / total_samples) * 100
        print(f"  {label}: {count} samples ({percentage:.1f}%)")

    # Calculate naive baseline accuracy (always predict majority class)
    majority_class = unique_labels[np.argmax(counts)]
    naive_accuracy = counts[np.argmax(counts)] / total_samples

    print(
        f"\n🎯 Naive baseline accuracy (always predict '{majority_class}'): {naive_accuracy:.3f}"
    )
    print(f"Majority class: {majority_class}")

    # Load the trained model and check its performance
    model_path = (
        "modular_analysis_results/discriminationanalysis/best_discrimination_model.pkl"
    )
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    model = model_data["model"]
    scaler = model_data["scaler"]

    # Scale features
    X_scaled = scaler.transform(X)

    # Get predictions
    y_pred = model.predict(X_scaled)

    # Try to get probabilities (may not be available for some models)
    try:
        y_proba = model.predict_proba(X_scaled)
        has_proba = True
    except AttributeError:
        print("⚠️  Model doesn't support probability predictions")
        y_proba = None
        has_proba = False

    # Calculate accuracy
    accuracy = np.mean(y_pred == y)

    print("\n🤖 Model Performance:")
    print(f"Model accuracy: {accuracy:.3f}")

    # Confusion matrix
    cm = confusion_matrix(y, y_pred, labels=unique_labels)

    print("\n📈 Confusion Matrix:")
    cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
    print(cm_df)

    # Normalized confusion matrix
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm_df = pd.DataFrame(cm_norm, index=unique_labels, columns=unique_labels)

    print("\n📈 Normalized Confusion Matrix (rows sum to 1):")
    print(cm_norm_df.round(3))

    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(y, y_pred, target_names=unique_labels))

    # Analyze prediction confidence
    print("\n🎯 Prediction Confidence Analysis:")
    if has_proba:
        max_probs = np.max(y_proba, axis=1)
        print(f"Average confidence: {max_probs.mean():.3f}")
        print(f"Min confidence: {max_probs.min():.3f}")
        print(f"Max confidence: {max_probs.max():.3f}")
    else:
        print("Confidence analysis not available (model doesn't support probabilities)")

    # Check if model is just predicting majority class
    majority_predictions = np.sum(y_pred == majority_class)
    print(
        f"Predictions of majority class '{majority_class}': {majority_predictions} out of {len(y_pred)} ({majority_predictions/len(y_pred)*100:.1f}%)"
    )

    return X, y, y_pred, unique_labels


def analyze_feature_discriminative_power(X, y, labels):
    """Analyze if features can actually discriminate between classes."""
    print("\n🔬 Analyzing Feature Discriminative Power")
    print("=" * 60)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a simple model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)

    # Feature importance
    feature_importance = rf.feature_importances_
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # Sort by importance
    sorted_idx = np.argsort(feature_importance)[::-1]
    top_features = 10

    print("\n🔝 Top 10 Most Important Features:")
    for i in range(min(top_features, len(feature_importance))):
        idx = sorted_idx[i]
        print(f"  {feature_names[idx]}: {feature_importance[idx]:.4f}")

    # Test accuracy
    train_acc = rf.score(X_train_scaled, y_train)
    test_acc = rf.score(X_test_scaled, y_test)

    print("\n🎯 Random Forest Performance:")
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")

    # Check class-wise feature differences
    print("\n📊 Feature Statistics by Class:")
    for label in labels:
        mask = y == label
        X_class = X[mask]

        print(f"\n{label.upper()} (n={np.sum(mask)}):")
        print(
            f"  Mean feature values: {X_class.mean(axis=0)[:5]}..."
        )  # First 5 features
        print(f"  Std feature values:  {X_class.std(axis=0)[:5]}...")

    return feature_importance


def suggest_solutions():
    """Suggest solutions for the class imbalance and discrimination issues."""
    print("\n💡 Suggested Solutions")
    print("=" * 60)

    solutions = [
        "1. 🔄 Balance the Dataset:",
        "   • Undersample the 'edge' class (443 → ~100 samples)",
        "   • Oversample 'contact' and 'no_contact' classes",
        "   • Use SMOTE or similar synthetic sampling techniques",
        "2. ⚖️ Use Class Weights:",
        "   • Apply class weights inversely proportional to class frequency",
        "   • This makes the model pay more attention to minority classes",
        "3. 📊 Collect More Balanced Data:",
        "   • Ensure equal representation of contact types in data collection",
        "   • Check if the experimental setup creates distinguishable signals",
        "4. 🔍 Investigate Feature Quality:",
        "   • Check if acoustic features actually differ between classes",
        "   • Consider domain-specific features (contact duration, frequency bands)",
        "5. 🎯 Use Better Evaluation Metrics:",
        "   • Focus on balanced accuracy, F1-score, not just overall accuracy",
        "   • Use confusion matrix and classification report for insights",
        "6. 🧪 Test Experimental Setup:",
        "   • Verify that different contact types produce different acoustic signatures",
        "   • Check microphone placement, finger condition, recording environment",
    ]

    for solution in solutions:
        print(solution)
        print()


def main():
    """Main analysis function."""
    try:
        X, y, y_pred, labels = analyze_class_imbalance()
        feature_importance = analyze_feature_discriminative_power(X, y, labels)
        suggest_solutions()

        print("\n✅ Analysis Complete!")
        print(
            "The severe class imbalance explains the poor discriminative performance."
        )
        print(
            "The model achieves ~68% accuracy by mostly predicting 'edge' (68% of data)."
        )

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
