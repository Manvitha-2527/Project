#!/usr/bin/env python3
"""
train_model.py - Training script with Leave-One-Participant-Out Cross-Validation

This script trains a 1D CNN model for sleep apnea classification using
Leave-One-Participant-Out (LOPO) cross-validation.

Usage:
    python train_model.py -dataset "Dataset/breathing_dataset.pkl"
"""

import argparse
import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
from collections import Counter

# Add models directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_dir, 'models'))

from cnn_model import SleepApneaCNN


class SleepApneaDataset(Dataset):
    """
    PyTorch Dataset for sleep apnea classification.
    """

    def __init__(self, windows, label_map):
        """
        Args:
            windows: List of window dictionaries with 'flow', 'thorac', 'spo2', 'label'
            label_map: Dictionary mapping label strings to integers
        """
        self.windows = windows
        self.label_map = label_map

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]

        # Normalize signals (z-score normalization)
        flow = self._normalize(window['flow'])
        thorac = self._normalize(window['thorac'])
        spo2 = self._normalize(window['spo2'])

        # Convert to tensors with channel dimension
        flow_tensor = torch.FloatTensor(flow).unsqueeze(0)  # (1, 960)
        thorac_tensor = torch.FloatTensor(thorac).unsqueeze(0)  # (1, 960)
        spo2_tensor = torch.FloatTensor(spo2).unsqueeze(0)  # (1, 120)

        # Get label
        label = self.label_map[window['label']]

        return {
            'flow': flow_tensor,
            'thorac': thorac_tensor,
            'spo2': spo2_tensor,
            'label': label
        }

    def _normalize(self, signal):
        """Z-score normalization."""
        signal = np.array(signal, dtype=np.float32)
        mean = np.mean(signal)
        std = np.std(signal)
        if std > 0:
            signal = (signal - mean) / std
        return signal


def load_dataset(dataset_path):
    """Load the preprocessed dataset."""
    with open(dataset_path, 'rb') as f:
        windows = pickle.load(f)
    return windows


def get_participants(windows):
    """Get list of unique participants."""
    participants = set()
    for w in windows:
        participants.add(w['participant_id'])
    return sorted(list(participants))


def split_by_participant(windows, test_participant):
    """
    Split data by participant for LOPO cross-validation.

    Args:
        windows: List of all windows
        test_participant: Participant ID to use for testing

    Returns:
        train_windows, test_windows
    """
    train_windows = [w for w in windows if w['participant_id'] != test_participant]
    test_windows = [w for w in windows if w['participant_id'] == test_participant]
    return train_windows, test_windows


def create_label_map(windows):
    """
    Create a mapping from label strings to integers.

    Groups rare events into the most similar category:
    - 'Normal' -> 0
    - 'Hypopnea' -> 1
    - 'Obstructive Apnea', 'Mixed Apnea', 'Central Apnea' -> 2 (Apnea)
    - 'Body event' -> 0 (Normal - not a breathing event)
    """
    # Standard 3-class mapping
    label_map = {
        'Normal': 0,
        'Hypopnea': 1,
        'Obstructive Apnea': 2,
        'Mixed Apnea': 2,  # Group with apnea
        'Central Apnea': 2,  # Group with apnea
        'Body event': 0,  # Not a breathing disorder, treat as normal
    }
    return label_map


def train_epoch(model, dataloader, criterion, optimizer, device, epoch=0, total_epochs=1):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        flow = batch['flow'].to(device)
        thorac = batch['thorac'].to(device)
        spo2 = batch['spo2'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(flow, thorac, spo2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0 or batch_idx == num_batches - 1:
            print(f"    Epoch {epoch+1}/{total_epochs} - Batch {batch_idx+1}/{num_batches} - Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            flow = batch['flow'].to(device)
            thorac = batch['thorac'].to(device)
            spo2 = batch['spo2'].to(device)
            labels = batch['label'].to(device)

            outputs = model(flow, thorac, spo2)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_preds, all_labels


def compute_class_weights(windows, label_map):
    """Compute class weights for imbalanced dataset."""
    labels = [label_map[w['label']] for w in windows]
    counts = Counter(labels)
    total = len(labels)
    num_classes = len(set(counts.keys()))

    weights = []
    for i in range(num_classes):
        count = counts.get(i, 1)
        weight = total / (num_classes * count)
        weights.append(weight)

    return torch.FloatTensor(weights)


def run_lopo_cv(
    windows,
    num_epochs=30,
    batch_size=32,
    learning_rate=0.001,
    device='cpu'
):
    """
    Run Leave-One-Participant-Out cross-validation.

    Args:
        windows: List of all window dictionaries
        num_epochs: Number of training epochs per fold
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: 'cpu' or 'cuda'

    Returns:
        dict: Results including metrics for each fold and overall
    """
    participants = get_participants(windows)
    label_map = create_label_map(windows)
    class_names = ['Normal', 'Hypopnea', 'Apnea']
    num_classes = 3

    print(f"\n{'='*60}")
    print("LEAVE-ONE-PARTICIPANT-OUT CROSS-VALIDATION")
    print(f"{'='*60}")
    print(f"Participants: {participants}")
    print(f"Total windows: {len(windows)}")
    print(f"Classes: {class_names}")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    print(f"Device: {device}")

    all_fold_results = []
    all_preds = []
    all_labels = []

    for fold_idx, test_participant in enumerate(participants):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{len(participants)}: Testing on {test_participant}")
        print(f"{'='*60}")

        # Split data
        train_windows, test_windows = split_by_participant(windows, test_participant)
        print(f"Train: {len(train_windows)} windows, Test: {len(test_windows)} windows")

        # Show class distribution
        train_labels = [label_map[w['label']] for w in train_windows]
        test_labels = [label_map[w['label']] for w in test_windows]
        print(f"Train distribution: {Counter(train_labels)}")
        print(f"Test distribution: {Counter(test_labels)}")

        # Create datasets and dataloaders
        train_dataset = SleepApneaDataset(train_windows, label_map)
        test_dataset = SleepApneaDataset(test_windows, label_map)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Create model (~50K parameters)
        model = SleepApneaCNN(
            flow_length=960,
            thorac_length=960,
            spo2_length=120,
            num_classes=num_classes,
            base_filters=16,  # Compact model
            dropout=0.5
        ).to(device)

        # Compute class weights for imbalanced data
        class_weights = compute_class_weights(train_windows, label_map).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # Training loop
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(num_epochs):
            print(f"\n  --- Epoch {epoch+1}/{num_epochs} ---")
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device,
                epoch=epoch, total_epochs=num_epochs
            )
            val_loss, _, _ = evaluate(model, test_loader, criterion, device)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()

            print(f"  Epoch {epoch+1} Complete: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Val Loss={val_loss:.4f}")

        # Load best model and evaluate
        model.load_state_dict(best_model_state)
        _, fold_preds, fold_labels = evaluate(model, test_loader, criterion, device)

        # Compute metrics for this fold
        fold_acc = accuracy_score(fold_labels, fold_preds)
        fold_precision = precision_score(fold_labels, fold_preds, average='weighted', zero_division=0)
        fold_recall = recall_score(fold_labels, fold_preds, average='weighted', zero_division=0)
        fold_cm = confusion_matrix(fold_labels, fold_preds, labels=list(range(num_classes)))

        print(f"\n  Fold {fold_idx + 1} Results:")
        print(f"    Accuracy:  {fold_acc:.4f}")
        print(f"    Precision: {fold_precision:.4f}")
        print(f"    Recall:    {fold_recall:.4f}")
        print(f"    Confusion Matrix:\n{fold_cm}")

        all_fold_results.append({
            'participant': test_participant,
            'accuracy': fold_acc,
            'precision': fold_precision,
            'recall': fold_recall,
            'confusion_matrix': fold_cm
        })

        all_preds.extend(fold_preds)
        all_labels.extend(fold_labels)

    # Compute overall metrics
    print(f"\n{'='*60}")
    print("OVERALL RESULTS (Aggregated across all folds)")
    print(f"{'='*60}")

    overall_acc = accuracy_score(all_labels, all_preds)
    overall_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    overall_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    overall_cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    print(f"\nOverall Accuracy:  {overall_acc:.4f}")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall:    {overall_recall:.4f}")

    print(f"\nOverall Confusion Matrix:")
    print(f"                Predicted")
    print(f"                {class_names}")
    print(f"Actual")
    for i, row in enumerate(overall_cm):
        print(f"  {class_names[i]:10s}  {row}")

    print(f"\nClassification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=class_names,
        zero_division=0
    ))

    # Per-fold summary
    print(f"\nPer-Fold Summary:")
    print(f"{'Participant':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 50)
    for result in all_fold_results:
        print(f"{result['participant']:<15} {result['accuracy']:>10.4f} "
              f"{result['precision']:>10.4f} {result['recall']:>10.4f}")

    mean_acc = np.mean([r['accuracy'] for r in all_fold_results])
    std_acc = np.std([r['accuracy'] for r in all_fold_results])
    print("-" * 50)
    print(f"{'Mean ± Std':<15} {mean_acc:>10.4f} ± {std_acc:.4f}")

    return {
        'fold_results': all_fold_results,
        'overall_accuracy': overall_acc,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_confusion_matrix': overall_cm,
        'all_predictions': all_preds,
        'all_labels': all_labels
    }


def main():
    parser = argparse.ArgumentParser(
        description='Train 1D CNN with LOPO cross-validation for sleep apnea detection'
    )
    parser.add_argument(
        '-dataset',
        type=str,
        default='Dataset/breathing_dataset.pkl',
        help='Path to the preprocessed dataset pickle file'
    )
    parser.add_argument(
        '-epochs',
        type=int,
        default=30,
        help='Number of training epochs (default: 30)'
    )
    parser.add_argument(
        '-batch_size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '-lr',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )

    args = parser.parse_args()

    # Handle relative paths
    if not os.path.isabs(args.dataset):
        args.dataset = os.path.join(project_dir, args.dataset)

    if not os.path.exists(args.dataset):
        print(f"Error: Dataset not found: {args.dataset}")
        return 1

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    windows = load_dataset(args.dataset)
    print(f"Loaded {len(windows)} windows")

    # Run LOPO cross-validation
    results = run_lopo_cv(
        windows,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device
    )

    print("\nTraining complete!")
    return 0


if __name__ == '__main__':
    exit(main())
