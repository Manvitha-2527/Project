#!/usr/bin/env python3
"""
cnn_model.py - 1D CNN model for sleep apnea classification

This module defines a 1D Convolutional Neural Network for classifying
breathing patterns from physiological signals (Nasal Airflow, Thoracic Movement, SpO2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SignalBranch(nn.Module):
    """
    1D CNN branch for processing a single signal type.

    Architecture:
    - 3 convolutional blocks with batch normalization and max pooling
    - Uses Global Average Pooling at the end to reduce parameters
    """

    def __init__(self, input_length, in_channels=1, base_filters=16):
        """
        Args:
            input_length: Number of samples in the input signal
            in_channels: Number of input channels (default 1 for single signal)
            base_filters: Number of filters in first conv layer
        """
        super(SignalBranch, self).__init__()

        # Block 1
        self.conv1 = nn.Conv1d(in_channels, base_filters, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)

        # Block 2
        self.conv2 = nn.Conv1d(base_filters, base_filters * 2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(base_filters * 2)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)

        # Block 3
        self.conv3 = nn.Conv1d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(base_filters * 4)

        # Global Average Pooling - reduces to single value per channel
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.output_channels = base_filters * 4

    def forward(self, x):
        """Forward pass."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x)  # (batch, channels, 1)

        return x.squeeze(-1)  # (batch, channels)


class SleepApneaCNN(nn.Module):
    """
    Multi-branch 1D CNN for sleep apnea classification.

    Architecture:
    - Separate CNN branches for each signal type (Flow, Thorac, SpO2)
    - Global Average Pooling for compact features
    - Fully connected layers for classification

    Model size: ~50K parameters
    """

    def __init__(
        self,
        flow_length=960,
        thorac_length=960,
        spo2_length=120,
        num_classes=3,
        base_filters=16,
        dropout=0.5
    ):
        """
        Args:
            flow_length: Length of nasal airflow signal (default 960 for 30s @ 32Hz)
            thorac_length: Length of thoracic movement signal (default 960 for 30s @ 32Hz)
            spo2_length: Length of SpO2 signal (default 120 for 30s @ 4Hz)
            num_classes: Number of output classes
            base_filters: Base number of filters for CNN branches (default 16)
            dropout: Dropout rate for regularization
        """
        super(SleepApneaCNN, self).__init__()

        # Signal processing branches
        self.flow_branch = SignalBranch(flow_length, in_channels=1, base_filters=base_filters)
        self.thorac_branch = SignalBranch(thorac_length, in_channels=1, base_filters=base_filters)
        self.spo2_branch = SignalBranch(spo2_length, in_channels=1, base_filters=base_filters)

        # After Global Average Pooling, each branch outputs base_filters * 4 features
        total_features = self.flow_branch.output_channels * 3  # 3 branches = 192 features

        # Fully connected layers
        self.fc1 = nn.Linear(total_features, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, flow, thorac, spo2):
        """
        Forward pass.

        Args:
            flow: Nasal airflow tensor of shape (batch, 1, flow_length)
            thorac: Thoracic movement tensor of shape (batch, 1, thorac_length)
            spo2: SpO2 tensor of shape (batch, 1, spo2_length)

        Returns:
            Tensor of shape (batch, num_classes) with class logits
        """
        # Process each signal branch (outputs are already flattened by GAP)
        flow_features = self.flow_branch(flow)
        thorac_features = self.thorac_branch(thorac)
        spo2_features = self.spo2_branch(spo2)

        # Concatenate all features
        combined = torch.cat([flow_features, thorac_features, spo2_features], dim=1)

        # Fully connected layers
        x = F.relu(self.bn_fc1(self.fc1(combined)))
        x = self.dropout1(x)
        x = self.fc2(x)

        return x


class SimpleCNN1D(nn.Module):
    """
    Simplified 1D CNN that concatenates all signals into a single input.

    This is an alternative architecture that treats the combined signals
    as a multi-channel input.
    """

    def __init__(
        self,
        signal_length=960,
        in_channels=3,
        num_classes=3,
        dropout=0.5
    ):
        """
        Args:
            signal_length: Length of each signal (all signals resampled to same length)
            in_channels: Number of input channels (signals)
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(SimpleCNN1D, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)

        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(2)

        # Calculate flattened size: signal_length / 16 * 256
        self.flat_size = (signal_length // 16) * 256

        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_size, 128)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, length)

        Returns:
            Tensor of shape (batch, num_classes)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)

        return x


def get_model(model_type='multi_branch', num_classes=3, **kwargs):
    """
    Factory function to create a model.

    Args:
        model_type: 'multi_branch' for SleepApneaCNN, 'simple' for SimpleCNN1D
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to model constructor

    Returns:
        nn.Module: The created model
    """
    if model_type == 'multi_branch':
        return SleepApneaCNN(num_classes=num_classes, **kwargs)
    elif model_type == 'simple':
        return SimpleCNN1D(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    # Test model creation and forward pass
    print("Testing SleepApneaCNN...")
    model = SleepApneaCNN(num_classes=3)
    print(model)

    # Test input shapes
    batch_size = 4
    flow = torch.randn(batch_size, 1, 960)
    thorac = torch.randn(batch_size, 1, 960)
    spo2 = torch.randn(batch_size, 1, 120)

    output = model(flow, thorac, spo2)
    print(f"\nInput shapes: flow={flow.shape}, thorac={thorac.shape}, spo2={spo2.shape}")
    print(f"Output shape: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
