#!/usr/bin/env python3
"""
Script to balance the imbalanced dataset and retrain the model with better discrimination.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
import pickle
import os


def load_data():
    """Load the imbalanced dataset."""
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

    print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Original class distribution: {Counter(y)}")

    return X, y


def balance_dataset(X, y, method="hybrid"):
    """Balance the dataset using various techniques."""
    print(f"\n🔄 Balancing dataset using method: {method}")

    if method == "undersample":
        # Undersample majority class
        rus = RandomUnderSampler(random_state=42)
        X_balanced, y_balanced = rus.fit_resample(X, y)
        print(f"After undersampling: {Counter(y_balanced)}")

    elif method == "smote":
        # Oversample minority classes using SMOTE
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        print(f"After SMOTE: {Counter(y_balanced)}")

    elif method == "hybrid":
        # Hybrid: undersample majority, then SMOTE minorities
        # First undersample edge to ~200 samples
        edge_mask = y == "edge"
        non_edge_mask = y != "edge"

        X_edge = X[edge_mask]
        y_edge = y[edge_mask]
        X_non_edge = X[non_edge_mask]
        y_non_edge = y[non_edge_mask]

        # Undersample edge
        n_edge_target = 200
        edge_indices = np.random.choice(len(X_edge), n_edge_target, replace=False)
        X_edge_downsampled = X_edge[edge_indices]
        y_edge_downsampled = y_edge[edge_indices]

        # Combine
        X_combined = np.vstack([X_edge_downsampled, X_non_edge])
        y_combined = np.hstack([y_edge_downsampled, y_non_edge])

        # Then apply SMOTE to balance everything
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X_combined, y_combined)

        print(f"After hybrid balancing: {Counter(y_balanced)}")

    return X_balanced, y_balanced


def train_balanced_model(X, y, model_type="svm"):
    """Train a model on the balanced dataset."""
    print(f"\n🤖 Training {model_type} model on balanced data...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Choose model
    if model_type == "svm":
        model = SVC(kernel="rbf", probability=True, random_state=42)
    elif model_type == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train model
    model.fit(X_train_scaled, y_train)

    # Evaluate
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    balanced_acc = balanced_accuracy_score(y_test, model.predict(X_test_scaled))

    print(".3f")
    print(".3f")
    print(".3f")

    # Detailed metrics
    y_pred = model.predict(X_test_scaled)
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=np.unique(y), columns=np.unique(y))
    print("\n📈 Confusion Matrix:")
    print(cm_df)

    return model, scaler, (X_train_scaled, X_test_scaled, y_train, y_test)


def evaluate_on_original_data(model, scaler, X_original, y_original):
    """Evaluate the balanced model on the original imbalanced data."""
    print("\n🔍 Evaluating balanced model on original imbalanced data...")

    X_scaled = scaler.transform(X_original)
    y_pred = model.predict(X_scaled)

    accuracy = np.mean(y_pred == y_original)
    balanced_acc = balanced_accuracy_score(y_original, y_pred)

    print(".3f")
    print(".3f")

    print("\n📋 Classification Report on Original Data:")
    print(classification_report(y_original, y_pred))

    return accuracy, balanced_acc


def save_balanced_model(model, scaler, class_names, save_path):
    """Save the balanced model."""
    model_data = {
        "model": model,
        "scaler": scaler,
        "classes": class_names,
        "balancing_method": "hybrid",
        "description": "Model trained on balanced dataset using hybrid undersampling + SMOTE",
    }

    with open(save_path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"\n💾 Saved balanced model to: {save_path}")


def main():
    """Main function to balance dataset and retrain model."""
    print("🔄 Dataset Balancing and Model Retraining")
    print("=" * 50)

    # Load original imbalanced data
    X, y = load_data()

    # Balance the dataset
    X_balanced, y_balanced = balance_dataset(X, y, method="hybrid")

    # Train model on balanced data
    model, scaler, test_data = train_balanced_model(
        X_balanced, y_balanced, model_type="svm"
    )

    # Evaluate on original data
    orig_acc, orig_balanced_acc = evaluate_on_original_data(model, scaler, X, y)

    # Save the improved model
    save_path = "modular_analysis_results/discriminationanalysis/balanced_discrimination_model.pkl"
    save_balanced_model(model, scaler, np.unique(y), save_path)

    print("\n✅ Balanced Model Training Complete!")
    print("📊 Summary:")
    print(f"   • Original accuracy: 67.2% (mostly predicting 'edge')")
    print(f"   • New accuracy on original data: {orig_acc:.3f}")
    print(f"   • New balanced accuracy: {orig_balanced_acc:.3f}")
    print("   • Balanced accuracy is a better measure of true discriminative ability")


if __name__ == "__main__":
    main()
