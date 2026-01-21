#!/usr/bin/env python3
"""
Hyperparameter tuning for MLP (Medium-HighReg) classifier.

This script searches for optimal hyperparameters using the validation set
(Workspace 1) as the optimization target, since validation accuracy is
what matters for cross-workspace generalization.

Uses Optuna for efficient Bayesian optimization with pruning.
"""

import numpy as np
import pickle
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import warnings

warnings.filterwarnings("ignore")


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TunableMLP(nn.Module):
    """MLP with configurable architecture for hyperparameter tuning."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        num_classes: int,
        dropout: float,
        use_batch_norm: bool,
        activation: str = "relu",
    ):
        super().__init__()

        # Select activation function
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "leaky_relu":
            act_fn = nn.LeakyReLU
        elif activation == "elu":
            act_fn = nn.ELU
        elif activation == "gelu":
            act_fn = nn.GELU
        else:
            act_fn = nn.ReLU

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)


def load_data():
    """Load the preprocessed data from the latest results."""
    # Try to load from results_v10 or find the latest results
    results_dirs = sorted(Path(".").glob("results_v*"), reverse=True)

    for results_dir in results_dirs:
        pkl_path = results_dir / "full_results.pkl"
        if pkl_path.exists():
            print(f"Loading data from {pkl_path}")
            with open(pkl_path, "rb") as f:
                results = pickle.load(f)
            break
    else:
        raise FileNotFoundError(
            "No results pickle file found. Run the main pipeline first."
        )

    # Extract batch results from data processing
    data_processing = results.get("data_processing", {})
    batch_results = data_processing.get("results", {}).get("batch_results", {})

    if not batch_results:
        raise ValueError("No batch results found in the data processing results")

    # Get training and validation dataset names
    training_datasets = data_processing.get("results", {}).get("training_datasets", [])
    validation_datasets = data_processing.get("results", {}).get(
        "validation_datasets", []
    )

    print(f"Training datasets: {len(training_datasets)}")
    print(f"Validation datasets: {len(validation_datasets)}")

    # Combine training data
    X_train_list = []
    y_train_list = []
    for dataset_name in training_datasets:
        if dataset_name in batch_results:
            X_train_list.append(batch_results[dataset_name]["features"])
            y_train_list.append(batch_results[dataset_name]["labels"])

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)

    # Combine validation data
    X_val_list = []
    y_val_list = []
    for dataset_name in validation_datasets:
        if dataset_name in batch_results:
            X_val_list.append(batch_results[dataset_name]["features"])
            y_val_list.append(batch_results[dataset_name]["labels"])

    X_val = np.vstack(X_val_list)
    y_val = np.concatenate(y_val_list)

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Features: {X_train.shape[1]}")

    return X_train, y_train, X_val, y_val


def create_objective(X_train_full, y_train_full, X_val, y_val, device):
    """Create the Optuna objective function."""

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_full)
    y_val_encoded = label_encoder.transform(y_val)
    num_classes = len(label_encoder.classes_)

    # Split training data into train/test for internal validation
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_full,
        y_train_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_train_encoded,
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.LongTensor(y_val_encoded)

    input_dim = X_train.shape[1]

    def objective(trial):
        """Optuna objective function."""

        # Sample hyperparameters
        # Architecture
        n_layers = trial.suggest_int("n_layers", 2, 5)
        hidden_dims = []
        for i in range(n_layers):
            dim = trial.suggest_int(f"hidden_dim_{i}", 32, 256, step=32)
            hidden_dims.append(dim)

        # Regularization
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 0.1, log=True)

        # Training
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        # Architecture options
        use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
        activation = trial.suggest_categorical(
            "activation", ["relu", "leaky_relu", "elu", "gelu"]
        )

        # Create model
        model = TunableMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            activation=activation,
        ).to(device)

        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10
        )

        # Training loop with early stopping
        best_val_acc = 0.0
        patience_counter = 0
        max_patience = 20
        max_epochs = 300

        for epoch in range(max_epochs):
            # Training
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            # Validation (on the held-out workspace)
            model.eval()
            with torch.no_grad():
                X_val_device = X_val_tensor.to(device)
                val_outputs = model(X_val_device)
                _, val_preds = torch.max(val_outputs, 1)
                val_acc = accuracy_score(y_val_tensor.numpy(), val_preds.cpu().numpy())

            scheduler.step(val_acc)

            # Track best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            # Report intermediate value for pruning
            trial.report(val_acc, epoch)

            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Early stopping
            if patience_counter >= max_patience:
                break

        # Final evaluation
        model.eval()
        with torch.no_grad():
            # Test accuracy (same workspace distribution)
            X_test_device = X_test_tensor.to(device)
            test_outputs = model(X_test_device)
            _, test_preds = torch.max(test_outputs, 1)
            test_acc = accuracy_score(y_test_tensor.numpy(), test_preds.cpu().numpy())

            # Validation accuracy (different workspace)
            X_val_device = X_val_tensor.to(device)
            val_outputs = model(X_val_device)
            _, val_preds = torch.max(val_outputs, 1)
            val_acc = accuracy_score(y_val_tensor.numpy(), val_preds.cpu().numpy())

        # Log additional metrics
        trial.set_user_attr("test_accuracy", test_acc)
        trial.set_user_attr("gap", test_acc - val_acc)

        # Optimize for validation accuracy (generalization)
        return val_acc

    return objective


def main():
    print("=" * 80)
    print("MLP Hyperparameter Tuning for Workspace Generalization")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Setup
    device = get_device()
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print("\nLoading data...")
    X_train, y_train, X_val, y_val = load_data()

    # Create Optuna study
    print("\nStarting hyperparameter optimization...")
    print("Optimizing for: VALIDATION ACCURACY (cross-workspace generalization)")
    print("-" * 80)

    # Create study with TPE sampler and median pruner
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=20),
    )

    # Create objective
    objective = create_objective(X_train, y_train, X_val, y_val, device)

    # Run optimization
    n_trials = 50  # Number of trials to run
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    # Results
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)

    print(f"\nBest trial:")
    trial = study.best_trial
    print(f"  Validation Accuracy: {trial.value:.4f}")
    print(f"  Test Accuracy: {trial.user_attrs.get('test_accuracy', 'N/A'):.4f}")
    print(f"  Gap (Test - Val): {trial.user_attrs.get('gap', 'N/A'):.4f}")

    print(f"\nBest hyperparameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    # Save results
    output_dir = Path("hyperparameter_tuning_results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save study
    results = {
        "best_trial": {
            "validation_accuracy": trial.value,
            "test_accuracy": trial.user_attrs.get("test_accuracy"),
            "gap": trial.user_attrs.get("gap"),
            "params": trial.params,
        },
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "test_accuracy": t.user_attrs.get("test_accuracy"),
                "gap": t.user_attrs.get("gap"),
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ],
        "n_trials": len(study.trials),
        "timestamp": timestamp,
    }

    results_path = output_dir / f"tuning_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    # Print top 5 trials
    print("\nTop 5 trials by validation accuracy:")
    print("-" * 80)

    completed_trials = [t for t in study.trials if t.value is not None]
    top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:5]

    for i, t in enumerate(top_trials, 1):
        test_acc = t.user_attrs.get("test_accuracy", 0)
        gap = t.user_attrs.get("gap", 0)
        print(
            f"{i}. Trial {t.number}: Val={t.value:.4f}, Test={test_acc:.4f}, Gap={gap:.4f}"
        )
        print(
            f"   Layers: {t.params.get('n_layers')}, Dropout: {t.params.get('dropout'):.2f}, "
            f"WD: {t.params.get('weight_decay'):.4f}, LR: {t.params.get('learning_rate'):.4f}"
        )

    # Generate code for best model
    print("\n" + "=" * 80)
    print("RECOMMENDED CLASSIFIER CONFIGURATION")
    print("=" * 80)

    best_params = trial.params
    hidden_dims = [
        best_params.get(f"hidden_dim_{i}") for i in range(best_params["n_layers"])
    ]

    print(
        f"""
Add this to discrimination_analysis.py _get_classifiers():

    # Optimized MLP (from hyperparameter tuning)
    classifiers["MLP (Optimized)"] = GPUMLPClassifier(
        hidden_layer_sizes={tuple(hidden_dims)},
        dropout={best_params['dropout']:.3f},
        learning_rate={best_params['learning_rate']:.6f},
        weight_decay={best_params['weight_decay']:.6f},
        batch_size={best_params['batch_size']},
        max_epochs=500,
        early_stopping=True,
        patience=25,
        use_batch_norm={best_params['use_batch_norm']},
        random_state=42,
    )
    """
    )

    return study


if __name__ == "__main__":
    study = main()
