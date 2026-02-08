"""
Generate proof-of-concept reconstruction figure:
Train on 80% of combined WS1+WS2+WS3 data, test on remaining 20%
"""
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import librosa
from acoustic_sensing.core.feature_extraction import GeometricFeatureExtractor

# Datasets from WS1, WS2, WS3
datasets = [
    "data/balanced_workspace_1_3class_squares_cutout",
    "data/balanced_workspace_1_3class_pure_no_contact", 
    "data/balanced_workspace_1_3class_pure_contact",
    "data/balanced_workspace_2_3class_squares_cutout",
    "data/balanced_workspace_2_3class_pure_no_contact",
    "data/balanced_workspace_2_3class_pure_contact",
    "data/balanced_workspace_3_3class_squares_cutout_v1",
    "data/balanced_workspace_3_3class_pure_no_contact",
    "data/balanced_workspace_3_3class_pure_contact",
]

print("=" * 80)
print("GENERATING PROOF-OF-CONCEPT RECONSTRUCTION")
print("=" * 80)
print("")

# Feature extractor
fe = GeometricFeatureExtractor(
    sr=48000,
    use_workspace_invariant=True,
    use_impulse_features=True,
)

# Collect all data
all_X = []
all_y = []
all_metadata = []

for dataset_path in datasets:
    print(f"Loading: {dataset_path}")
    dataset = Path(dataset_path)
    sweep_df = pd.read_csv(dataset / "sweep.csv")
    label_col = 'relabeled_label' if 'relabeled_label' in sweep_df.columns else 'original_label'
    data_dir = dataset / "data"
    
    for idx, row in sweep_df.iterrows():
        audio_path = data_dir / row['filename']
        try:
            audio, _ = librosa.load(str(audio_path), sr=48000)
            features = fe.extract_features(audio)
            all_X.append(features)
            all_y.append(row[label_col])
            all_metadata.append({
                'dataset': dataset_path,
                'x': row.get('normalized_x', row.get('x', 0)),
                'y': row.get('normalized_y', row.get('y', 0)),
                'label': row[label_col]
            })
        except Exception as e:
            continue

X = np.array(all_X)
y = np.array(all_y)

print(f"\nTotal samples: {len(X)}")
print(f"Classes: {np.unique(y)}")
print(f"Class distribution:")
for cls in np.unique(y):
    print(f"  {cls}: {np.sum(y == cls)}")

# 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {len(X_train)} samples")
print(f"Test: {len(X_test)} samples")

# Train model
print("\nTraining Random Forest...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train_scaled, y_train)

# Evaluate
train_acc = clf.score(X_train_scaled, y_train)
test_acc = clf.score(X_test_scaled, y_test)

print(f"\nTrain accuracy: {train_acc*100:.2f}%")
print(f"Test accuracy: {test_acc*100:.2f}%")

# Save model
output_dir = Path("proof_of_concept_model")
output_dir.mkdir(exist_ok=True)

model_data = {
    'model': clf,
    'scaler': scaler,
    'classes': clf.classes_,
    'classifier_name': 'Random Forest',
    'train_accuracy': train_acc,
    'test_accuracy': test_acc,
}

with open(output_dir / "model.pkl", 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n✓ Model saved to {output_dir}/model.pkl")
print(f"✓ Ready to generate reconstruction figures!")
