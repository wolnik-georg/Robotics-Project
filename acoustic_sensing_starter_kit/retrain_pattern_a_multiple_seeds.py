"""
Retrain Pattern A model with multiple random seeds to check stability.

Pattern A: WS2+WS3 train/test → WS1 validation

This will train 5 models with different random seeds and compare:
1. Test accuracy (on WS2+WS3 hold-out)
2. Validation accuracy (on WS1)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path.cwd() / "src"))
from acoustic_sensing.core.feature_extraction import GeometricFeatureExtractor

print("=" * 80)
print("PATTERN A: MULTI-SEED TRAINING EXPERIMENT")
print("  Training: WS2 + WS3 (all surfaces)")
print("  Validation: WS1 (all surfaces)")
print("  Seeds: 5 different random states")
print("=" * 80)

# Dataset paths - using the balanced datasets
TRAINING_DATASETS = [
    "data/balanced_workspace_2_squares_cutout",
    "data/balanced_workspace_2_pure_contact",
    "data/balanced_workspace_2_pure_no_contact",
    "data/balanced_workspace_3_squares_cutout_v1",
    "data/balanced_workspace_3_pure_contact",
    "data/balanced_workspace_3_pure_no_contact",
]

VALIDATION_DATASETS = [
    "data/balanced_workspace_1_squares_cutout_oversample",
    "data/balanced_workspace_1_pure_contact_oversample",
    "data/balanced_workspace_1_pure_no_contact_oversample",
]

# Feature extractor
fe = GeometricFeatureExtractor(
    use_workspace_invariant=True, use_impulse_features=True, sr=48000
)


def load_dataset(dataset_path):
    """Load audio files and extract features from a dataset."""
    import librosa

    data_dir = Path(dataset_path) / "data"
    if not data_dir.exists():
        print(f"  Warning: {data_dir} not found")
        return None, None

    wav_files = list(data_dir.glob("*.wav"))
    if not wav_files:
        print(f"  Warning: No wav files in {data_dir}")
        return None, None

    features_list = []
    labels = []

    for wav_file in wav_files:
        # Extract label from filename (e.g., "123_contact.wav" -> "contact")
        parts = wav_file.stem.split("_")
        if len(parts) >= 2:
            label = "_".join(parts[1:])  # Handle "no_contact"
        else:
            continue

        # Skip edge class
        if label == "edge":
            continue

        try:
            audio, sr = librosa.load(wav_file, sr=48000)
            feat = fe.extract_features(audio)
            features_list.append(feat)
            labels.append(label)
        except Exception as e:
            print(f"  Error loading {wav_file}: {e}")
            continue

    if not features_list:
        return None, None

    X = np.array(features_list)
    y = np.array(labels)
    return X, y


# Load all training data
print("\n📁 Loading training data (WS2 + WS3)...")
X_train_all = []
y_train_all = []

for ds in TRAINING_DATASETS:
    print(f"  Loading {Path(ds).name}...")
    X, y = load_dataset(ds)
    if X is not None:
        X_train_all.append(X)
        y_train_all.append(y)
        print(f"    {len(y)} samples ({dict(zip(*np.unique(y, return_counts=True)))})")

X_train_full = np.vstack(X_train_all)
y_train_full = np.concatenate(y_train_all)
print(f"\n  Total training samples: {len(y_train_full)}")
print(
    f"  Class distribution: {dict(zip(*np.unique(y_train_full, return_counts=True)))}"
)

# Load all validation data
print("\n📁 Loading validation data (WS1)...")
X_val_all = []
y_val_all = []

for ds in VALIDATION_DATASETS:
    print(f"  Loading {Path(ds).name}...")
    X, y = load_dataset(ds)
    if X is not None:
        X_val_all.append(X)
        y_val_all.append(y)
        print(f"    {len(y)} samples ({dict(zip(*np.unique(y, return_counts=True)))})")

X_val_full = np.vstack(X_val_all)
y_val_full = np.concatenate(y_val_all)
print(f"\n  Total validation samples: {len(y_val_full)}")
print(f"  Class distribution: {dict(zip(*np.unique(y_val_full, return_counts=True)))}")

# Train with multiple seeds
print("\n" + "=" * 80)
print("🎲 TRAINING WITH MULTIPLE RANDOM SEEDS")
print("=" * 80)

SEEDS = [42, 123, 456, 789, 2024]
results = []

output_dir = Path("pattern_a_seed_comparison")
output_dir.mkdir(parents=True, exist_ok=True)

for seed in SEEDS:
    print(f"\n--- Seed {seed} ---")

    # Split training data into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=seed,
        stratify=y_train_full,
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val_full)

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100, random_state=seed, n_jobs=-1, class_weight="balanced"
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
    test_f1 = f1_score(y_test, model.predict(X_test_scaled), average="weighted")

    val_pred = model.predict(X_val_scaled)
    val_acc = accuracy_score(y_val_full, val_pred)
    val_f1 = f1_score(y_val_full, val_pred, average="weighted")

    print(f"  TEST (WS2+WS3 holdout): {test_acc*100:.1f}% acc, {test_f1*100:.1f}% F1")
    print(f"  VALIDATION (WS1):       {val_acc*100:.1f}% acc, {val_f1*100:.1f}% F1")
    print(f"  Gap:                    {(test_acc - val_acc)*100:.1f}%")

    results.append(
        {
            "seed": seed,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "gap": test_acc - val_acc,
        }
    )

    # Save model
    model_dict = {
        "model": model,
        "scaler": scaler,
        "classes": model.classes_,
        "classifier_name": "Random Forest",
        "rank": 1,
        "metrics": {
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "validation_accuracy": val_acc,
            "validation_f1": val_f1,
        },
        "seed": seed,
    }

    model_path = output_dir / f"model_seed_{seed}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_dict, f)

# Summary
print("\n" + "=" * 80)
print("📊 SUMMARY ACROSS ALL SEEDS")
print("=" * 80)

df = pd.DataFrame(results)
print(f"\n{'Seed':<10} {'TEST Acc':<12} {'VAL Acc':<12} {'Gap':<10}")
print("-" * 44)
for _, row in df.iterrows():
    print(
        f"{row['seed']:<10} {row['test_acc']*100:>6.1f}%     {row['val_acc']*100:>6.1f}%     {row['gap']*100:>6.1f}%"
    )

print(
    f"\n{'Mean':<10} {df['test_acc'].mean()*100:>6.1f}%     {df['val_acc'].mean()*100:>6.1f}%     {df['gap'].mean()*100:>6.1f}%"
)
print(
    f"{'Std':<10} {df['test_acc'].std()*100:>6.1f}%     {df['val_acc'].std()*100:>6.1f}%     {df['gap'].std()*100:>6.1f}%"
)
print(
    f"{'Min':<10} {df['test_acc'].min()*100:>6.1f}%     {df['val_acc'].min()*100:>6.1f}%"
)
print(
    f"{'Max':<10} {df['test_acc'].max()*100:>6.1f}%     {df['val_acc'].max()*100:>6.1f}%"
)

# Find best seed for validation
best_idx = df["val_acc"].idxmax()
best_seed = df.loc[best_idx, "seed"]
best_val_acc = df.loc[best_idx, "val_acc"]

print(f"\n🏆 Best seed for validation: {best_seed} ({best_val_acc*100:.1f}%)")
print(f"\n✅ Models saved to: {output_dir}/")

# Also save summary
df.to_csv(output_dir / "seed_comparison_results.csv", index=False)
print(f"✅ Results saved to: {output_dir}/seed_comparison_results.csv")
