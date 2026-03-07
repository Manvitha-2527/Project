# Sleep Apnea Detection from Physiological Signals

This project implements a deep learning pipeline for detecting breathing abnormalities (apnea, hypopnea) during sleep using physiological signals.

## Project Structure

```
Project Root/
├── Data/                    # Raw participant data
│   ├── AP01/
│   ├── AP02/
│   ├── AP03/
│   ├── AP04/
│   └── AP05/
├── Dataset/                 # Processed dataset
│   ├── breathing_dataset.pkl
│   └── breathing_dataset_metadata.csv
├── Visualizations/          # Generated PDF visualizations
├── models/
│   └── cnn_model.py         # 1D CNN model definition
├── scripts/
│   ├── vis.py               # Visualization script
│   ├── create_dataset.py    # Data preprocessing script
│   └── train_model.py       # Training with LOPO CV
├── requirements.txt
└── README.md
```

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Visualization (Task 1)

Generate PDF visualization of signals with breathing events overlaid:

```bash
python scripts/vis.py -name "Data/AP01"
```

Output: `Visualizations/AP01_visualization.pdf`

### 2. Dataset Creation (Task 2)

Preprocess signals and create labeled dataset:

```bash
python scripts/create_dataset.py -in_dir "Data" -out_dir "Dataset"
```

This script:
- Applies bandpass filter (0.17-0.4 Hz) for breathing frequencies
- Creates 30-second windows with 50% overlap
- Labels windows based on >50% overlap with breathing events

### 3. Model Training (Task 3)

Train 1D CNN with Leave-One-Participant-Out cross-validation:

```bash
python scripts/train_model.py -dataset "Dataset/breathing_dataset.pkl" -epochs 30
```

Options:
- `-epochs`: Number of training epochs (default: 30)
- `-batch_size`: Batch size (default: 32)
- `-lr`: Learning rate (default: 0.001)

## Data Description

### Input Signals
- **Nasal Airflow**: 32 Hz sampling rate
- **Thoracic Movement**: 32 Hz sampling rate
- **SpO2 (Oxygen Saturation)**: 4 Hz sampling rate

### Event Types
- **Normal**: No breathing irregularity
- **Hypopnea**: Partial airway obstruction
- **Obstructive Apnea**: Complete airway obstruction

## Model Architecture

The 1D CNN uses:
- **Three parallel branches** for each signal type
- **Global Average Pooling** for efficient feature extraction
- **~40K parameters** (lightweight for CPU training)

```
Flow (960 samples) ──► CNN Branch ──┐
                                    │
Thorac (960 samples) ─► CNN Branch ─┼──► Concat ──► FC ──► Softmax
                                    │
SpO2 (120 samples) ──► CNN Branch ──┘
```

## Evaluation

The model is evaluated using:
- **Leave-One-Participant-Out (LOPO)** cross-validation
- **Metrics**: Accuracy, Precision, Recall, Confusion Matrix

## Results

With 30 epochs of training, typical results:
- Overall Accuracy: ~70-80%
- Challenge: Highly imbalanced dataset (91% Normal class)
