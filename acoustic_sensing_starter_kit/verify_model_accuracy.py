#!/usr/bin/env python3
"""
Quick verification: Does the trained model actually achieve the claimed accuracy?
"""
import pickle
import pandas as pd
import numpy as np
import librosa
from acoustic_sensing.core.feature_extraction import GeometricFeatureExtractor
from pathlib import Path
import sys


def verify_model(model_path, datasets):
    """Verify model accuracy on validation datasets."""

    # Load model
    print(f"Loading model: {model_path}")
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    model = model_data["model"]
    scaler = model_data["scaler"]

    # Create feature extractor
    fe = GeometricFeatureExtractor(
        sr=48000,
        use_workspace_invariant=True,
        use_impulse_features=True,
    )

    # Load ALL validation data
    all_features = []
    all_labels = []

    for dataset_path in datasets:
        dataset = Path(dataset_path)
        if not dataset.exists():
            print(f"⚠ Dataset not found: {dataset}")
            continue

        sweep_df = pd.read_csv(dataset / "sweep.csv")
        label_col = (
            "relabeled_label"
            if "relabeled_label" in sweep_df.columns
            else "original_label"
        )
        data_dir = dataset / "data"

        print(
            f"  Loading {dataset.name}... ({len(sweep_df)} samples)", end="", flush=True
        )

        for idx, row in sweep_df.iterrows():
            audio_path = data_dir / row["filename"]
            try:
                audio, _ = librosa.load(str(audio_path), sr=48000)
                features = fe.extract_features(audio)
                all_features.append(features)
                all_labels.append(row[label_col])
            except Exception as e:
                continue
        print(f" ✓ {len(all_features)} loaded")

    # Convert to arrays
    X = np.array(all_features)
    y = np.array(all_labels)

    print(f"\nTotal validation samples: {len(y)}")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  {cls}: {count}")

    # Scale and predict
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    # Calculate accuracy
    accuracy = np.mean(y == y_pred)
    print(f"\n{'='*60}")
    print(f"VALIDATION ACCURACY: {accuracy*100:.2f}%")
    print(f"{'='*60}")

    # Per-class accuracy
    print("\nPer-class accuracy:")
    for cls in np.unique(y):
        cls_mask = y == cls
        cls_acc = np.mean(y_pred[cls_mask] == y[cls_mask])
        n_correct = np.sum(y_pred[cls_mask] == y[cls_mask])
        n_total = np.sum(cls_mask)
        print(f"  {cls:12s}: {cls_acc*100:5.1f}% ({n_correct}/{n_total})")

    return accuracy


if __name__ == "__main__":
    if len(sys.argv) > 1:
        experiment_dir = sys.argv[1]
    else:
        experiment_dir = "test_pipeline_3class_v1_RETRAIN"

    model_path = f"{experiment_dir}/discriminationanalysis/trained_models/model_rank1_random_forest.pkl"

    # WS2 validation datasets (from Rotation 1)
    ws2_datasets = [
        "data/balanced_workspace_2_3class_squares_cutout",
        "data/balanced_workspace_2_3class_pure_no_contact",
        "data/balanced_workspace_2_3class_pure_contact",
    ]

    print("=" * 60)
    print("MODEL ACCURACY VERIFICATION")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Validation: WS2 (3 datasets)")
    print("=" * 60)
    print()

    try:
        accuracy = verify_model(model_path, ws2_datasets)

        print()
        if accuracy > 0.80:
            print("✅ SUCCESS: Model achieves >80% accuracy!")
            print("   Ready for reconstruction.")
            sys.exit(0)
        elif accuracy > 0.70:
            print("⚠ WARNING: Model achieves 70-80% accuracy.")
            print("  Lower than expected but usable.")
            sys.exit(0)
        else:
            print("❌ FAILURE: Model achieves <70% accuracy!")
            print("   Something is wrong with training/features.")
            sys.exit(1)

    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        print("   Training may still be running...")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
