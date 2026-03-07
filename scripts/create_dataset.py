#!/usr/bin/env python3
"""
create_dataset.py - Preprocessing and dataset creation for sleep apnea detection

This script:
1. Applies bandpass filtering to retain breathing frequencies (0.17-0.4 Hz)
2. Splits signals into 30-second windows with 50% overlap
3. Labels windows based on overlap with breathing events
4. Saves the processed dataset

Usage:
    python create_dataset.py -in_dir "Data" -out_dir "Dataset"
"""

import argparse
import os
import pickle
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import signal


def parse_signal_file(filepath):
    """
    Parse a signal file and return metadata and data as a DataFrame.

    Args:
        filepath: Path to the signal file

    Returns:
        dict: Metadata dictionary with 'signal_type', 'start_time', 'sample_rate', 'length', 'unit'
        pd.DataFrame: DataFrame with 'timestamp' and 'value' columns
    """
    metadata = {}
    data_lines = []
    in_data_section = False

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('Data:'):
                in_data_section = True
                continue

            if not in_data_section:
                if line.startswith('Signal Type:'):
                    metadata['signal_type'] = line.split(':', 1)[1].strip()
                elif line.startswith('Start Time:'):
                    metadata['start_time'] = line.split(':', 1)[1].strip()
                elif line.startswith('Sample Rate:'):
                    metadata['sample_rate'] = int(line.split(':', 1)[1].strip())
                elif line.startswith('Length:'):
                    metadata['length'] = int(line.split(':', 1)[1].strip())
                elif line.startswith('Unit:'):
                    metadata['unit'] = line.split(':', 1)[1].strip()
            else:
                if ';' in line:
                    parts = line.split(';')
                    timestamp_str = parts[0].strip()
                    value = float(parts[1].strip())
                    data_lines.append((timestamp_str, value))

    df = pd.DataFrame(data_lines, columns=['timestamp_str', 'value'])

    def parse_timestamp(ts):
        ts = ts.replace(',', '.')
        try:
            return datetime.strptime(ts, '%d.%m.%Y %H:%M:%S.%f')
        except ValueError:
            return datetime.strptime(ts, '%d.%m.%Y %H:%M:%S')

    df['timestamp'] = df['timestamp_str'].apply(parse_timestamp)
    df = df.drop('timestamp_str', axis=1)

    return metadata, df


def parse_events_file(filepath):
    """
    Parse the flow events file and return a DataFrame of events.

    Args:
        filepath: Path to the events file

    Returns:
        pd.DataFrame: DataFrame with 'start_time', 'end_time', 'duration', 'event_type', 'sleep_stage'
    """
    import re
    events = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Data lines: "30.05.2024 23:48:45,119-23:49:01,408; 16;Hypopnea; N1"
            match = re.match(
                r'(\d{2}\.\d{2}\.\d{4})\s+(\d{2}:\d{2}:\d{2},\d+)-(\d{2}:\d{2}:\d{2},\d+);\s*(\d+);([^;]+);\s*(\w+)',
                line
            )
            if match:
                date = match.group(1)
                start_time_str = match.group(2).replace(',', '.')
                end_time_str = match.group(3).replace(',', '.')
                duration = int(match.group(4))
                event_type = match.group(5).strip()
                sleep_stage = match.group(6).strip()

                start_datetime = datetime.strptime(f"{date} {start_time_str}", '%d.%m.%Y %H:%M:%S.%f')
                end_datetime = datetime.strptime(f"{date} {end_time_str}", '%d.%m.%Y %H:%M:%S.%f')

                if end_datetime < start_datetime:
                    end_datetime += timedelta(days=1)

                events.append({
                    'start_time': start_datetime,
                    'end_time': end_datetime,
                    'duration': duration,
                    'event_type': event_type,
                    'sleep_stage': sleep_stage
                })

    return pd.DataFrame(events)


def find_files(participant_dir):
    """
    Find signal and event files in a participant directory.

    Args:
        participant_dir: Path to participant directory

    Returns:
        dict: Dictionary with keys 'flow', 'thorac', 'spo2', 'events', 'sleep_profile'
    """
    files = {}
    all_files = os.listdir(participant_dir)

    for filename in all_files:
        lower_name = filename.lower()
        filepath = os.path.join(participant_dir, filename)

        if 'flow event' in lower_name:
            files['events'] = filepath
        elif 'flow' in lower_name and 'event' not in lower_name:
            files['flow'] = filepath
        elif 'thorac' in lower_name:
            files['thorac'] = filepath
        elif 'spo2' in lower_name:
            files['spo2'] = filepath
        elif 'sleep profile' in lower_name:
            files['sleep_profile'] = filepath

    return files


def apply_bandpass_filter(data, sample_rate, low_freq=0.17, high_freq=0.4):
    """
    Apply a bandpass filter to retain breathing frequency components.

    Human breathing rate: 10-24 breaths/min = 0.17-0.4 Hz

    Args:
        data: 1D numpy array of signal values
        sample_rate: Sampling frequency in Hz
        low_freq: Lower cutoff frequency (default 0.17 Hz)
        high_freq: Upper cutoff frequency (default 0.4 Hz)

    Returns:
        np.array: Filtered signal
    """
    nyquist = sample_rate / 2

    # Ensure frequencies are valid
    if high_freq >= nyquist:
        high_freq = nyquist * 0.95

    if low_freq >= high_freq:
        low_freq = high_freq / 2

    # Design Butterworth bandpass filter
    # Using 4th order for good frequency selectivity while maintaining phase response
    order = 4
    low_normalized = low_freq / nyquist
    high_normalized = high_freq / nyquist

    # Get filter coefficients
    b, a = signal.butter(order, [low_normalized, high_normalized], btype='band')

    # Apply zero-phase filtering to avoid phase distortion
    filtered_data = signal.filtfilt(b, a, data)

    return filtered_data


def create_windows(signal_data, timestamps, sample_rate, window_size_sec=30, overlap=0.5):
    """
    Split signal into windows with specified overlap.

    Args:
        signal_data: 1D numpy array of signal values
        timestamps: Array of datetime timestamps
        sample_rate: Sampling frequency in Hz
        window_size_sec: Window size in seconds (default 30)
        overlap: Overlap fraction (default 0.5 for 50%)

    Returns:
        list: List of dictionaries with 'data', 'start_time', 'end_time', 'start_idx', 'end_idx'
    """
    samples_per_window = int(window_size_sec * sample_rate)
    step_size = int(samples_per_window * (1 - overlap))

    windows = []
    start_idx = 0

    while start_idx + samples_per_window <= len(signal_data):
        end_idx = start_idx + samples_per_window
        window_data = signal_data[start_idx:end_idx]
        window_start_time = timestamps[start_idx]
        window_end_time = timestamps[end_idx - 1]

        windows.append({
            'data': window_data,
            'start_time': window_start_time,
            'end_time': window_end_time,
            'start_idx': start_idx,
            'end_idx': end_idx
        })

        start_idx += step_size

    return windows


def to_datetime(ts):
    """Convert numpy datetime64 or pandas Timestamp to Python datetime."""
    if isinstance(ts, np.datetime64):
        return pd.Timestamp(ts).to_pydatetime()
    elif hasattr(ts, 'to_pydatetime'):
        return ts.to_pydatetime()
    return ts


def calculate_overlap(window_start, window_end, event_start, event_end):
    """
    Calculate the overlap between a window and an event.

    Args:
        window_start, window_end: Window time boundaries
        event_start, event_end: Event time boundaries

    Returns:
        float: Overlap fraction relative to window duration
    """
    # Convert all timestamps to Python datetime
    window_start = to_datetime(window_start)
    window_end = to_datetime(window_end)
    event_start = to_datetime(event_start)
    event_end = to_datetime(event_end)

    # Find the overlap interval
    overlap_start = max(window_start, event_start)
    overlap_end = min(window_end, event_end)

    if overlap_start >= overlap_end:
        return 0.0

    overlap_duration = (overlap_end - overlap_start).total_seconds()
    window_duration = (window_end - window_start).total_seconds()

    return overlap_duration / window_duration


def label_window(window, events_df, overlap_threshold=0.5):
    """
    Assign a label to a window based on event overlap.

    Args:
        window: Window dictionary with 'start_time', 'end_time'
        events_df: DataFrame of events
        overlap_threshold: Minimum overlap fraction to assign event label (default 0.5)

    Returns:
        str: Label ('Normal', 'Hypopnea', 'Obstructive Apnea', etc.)
    """
    if len(events_df) == 0:
        return 'Normal'

    window_start = window['start_time']
    window_end = window['end_time']

    max_overlap = 0
    assigned_label = 'Normal'

    for _, event in events_df.iterrows():
        overlap = calculate_overlap(
            window_start, window_end,
            event['start_time'], event['end_time']
        )

        if overlap > max_overlap:
            max_overlap = overlap
            if overlap > overlap_threshold:
                assigned_label = event['event_type']

    return assigned_label


def process_participant(participant_dir, participant_id):
    """
    Process all signals for a single participant.

    Args:
        participant_dir: Path to participant data directory
        participant_id: Participant identifier string

    Returns:
        list: List of processed window dictionaries
    """
    print(f"\nProcessing {participant_id}...")

    files = find_files(participant_dir)

    required_files = ['flow', 'thorac', 'spo2', 'events']
    for req in required_files:
        if req not in files:
            raise FileNotFoundError(f"Could not find {req} file in {participant_dir}")

    # Load signals
    print(f"  Loading signals...")
    flow_meta, flow_df = parse_signal_file(files['flow'])
    thorac_meta, thorac_df = parse_signal_file(files['thorac'])
    spo2_meta, spo2_df = parse_signal_file(files['spo2'])
    events_df = parse_events_file(files['events'])

    print(f"  Flow: {len(flow_df)} samples at {flow_meta['sample_rate']} Hz")
    print(f"  Thorac: {len(thorac_df)} samples at {thorac_meta['sample_rate']} Hz")
    print(f"  SpO2: {len(spo2_df)} samples at {spo2_meta['sample_rate']} Hz")
    print(f"  Events: {len(events_df)} breathing events")

    # Apply bandpass filter to respiration signals
    print(f"  Applying bandpass filter (0.17-0.4 Hz) to respiration signals...")
    flow_filtered = apply_bandpass_filter(
        flow_df['value'].values,
        flow_meta['sample_rate']
    )
    thorac_filtered = apply_bandpass_filter(
        thorac_df['value'].values,
        thorac_meta['sample_rate']
    )

    # SpO2 doesn't need breathing frequency filtering (it's oxygen saturation, not waveform)
    # But we'll still normalize it for the model
    spo2_values = spo2_df['value'].values

    # Create windows for each signal (30 seconds, 50% overlap)
    print(f"  Creating 30-second windows with 50% overlap...")

    flow_windows = create_windows(
        flow_filtered,
        flow_df['timestamp'].values,
        flow_meta['sample_rate'],
        window_size_sec=30,
        overlap=0.5
    )

    thorac_windows = create_windows(
        thorac_filtered,
        thorac_df['timestamp'].values,
        thorac_meta['sample_rate'],
        window_size_sec=30,
        overlap=0.5
    )

    spo2_windows = create_windows(
        spo2_values,
        spo2_df['timestamp'].values,
        spo2_meta['sample_rate'],
        window_size_sec=30,
        overlap=0.5
    )

    print(f"  Flow windows: {len(flow_windows)}")
    print(f"  Thorac windows: {len(thorac_windows)}")
    print(f"  SpO2 windows: {len(spo2_windows)}")

    # Align windows by time (use the minimum number of windows)
    n_windows = min(len(flow_windows), len(thorac_windows), len(spo2_windows))

    print(f"  Aligning to {n_windows} windows...")

    # Create dataset entries
    processed_windows = []

    for i in range(n_windows):
        # Label based on flow window timing (primary signal)
        label = label_window(flow_windows[i], events_df, overlap_threshold=0.5)

        window_entry = {
            'participant_id': participant_id,
            'window_idx': i,
            'start_time': flow_windows[i]['start_time'],
            'end_time': flow_windows[i]['end_time'],
            'flow': flow_windows[i]['data'],
            'thorac': thorac_windows[i]['data'],
            'spo2': spo2_windows[i]['data'],
            'label': label
        }

        processed_windows.append(window_entry)

    # Count labels
    label_counts = {}
    for w in processed_windows:
        label = w['label']
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"  Label distribution: {label_counts}")

    return processed_windows


def create_dataset(input_dir, output_dir):
    """
    Process all participants and create the dataset.

    Args:
        input_dir: Path to Data directory containing participant folders
        output_dir: Path to output directory for dataset files
    """
    # Find all participant directories
    participant_dirs = []
    for item in sorted(os.listdir(input_dir)):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path) and item.startswith('AP'):
            participant_dirs.append((item, item_path))

    print(f"Found {len(participant_dirs)} participants: {[p[0] for p in participant_dirs]}")

    all_windows = []

    for participant_id, participant_path in participant_dirs:
        try:
            windows = process_participant(participant_path, participant_id)
            all_windows.extend(windows)
        except Exception as e:
            print(f"  Error processing {participant_id}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*50}")
    print(f"Total windows: {len(all_windows)}")

    # Overall label distribution
    label_counts = {}
    for w in all_windows:
        label = w['label']
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"Overall label distribution:")
    for label, count in sorted(label_counts.items()):
        pct = 100 * count / len(all_windows)
        print(f"  {label}: {count} ({pct:.1f}%)")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save dataset as pickle (preserves numpy arrays efficiently)
    pickle_path = os.path.join(output_dir, 'breathing_dataset.pkl')
    print(f"\nSaving dataset to {pickle_path}...")

    with open(pickle_path, 'wb') as f:
        pickle.dump(all_windows, f)

    # Also save a CSV with metadata (without the signal arrays)
    csv_data = []
    for w in all_windows:
        csv_data.append({
            'participant_id': w['participant_id'],
            'window_idx': w['window_idx'],
            'start_time': w['start_time'],
            'end_time': w['end_time'],
            'label': w['label'],
            'flow_samples': len(w['flow']),
            'thorac_samples': len(w['thorac']),
            'spo2_samples': len(w['spo2'])
        })

    csv_df = pd.DataFrame(csv_data)
    csv_path = os.path.join(output_dir, 'breathing_dataset_metadata.csv')
    csv_df.to_csv(csv_path, index=False)
    print(f"Saved metadata to {csv_path}")

    print(f"\nDataset creation complete!")
    print(f"  Total windows: {len(all_windows)}")
    print(f"  Flow samples per window: {len(all_windows[0]['flow'])} (30s @ 32Hz = 960)")
    print(f"  Thorac samples per window: {len(all_windows[0]['thorac'])} (30s @ 32Hz = 960)")
    print(f"  SpO2 samples per window: {len(all_windows[0]['spo2'])} (30s @ 4Hz = 120)")


def main():
    parser = argparse.ArgumentParser(
        description='Create preprocessed dataset for sleep apnea detection'
    )
    parser.add_argument(
        '-in_dir',
        type=str,
        required=True,
        help='Input directory containing participant data folders'
    )
    parser.add_argument(
        '-out_dir',
        type=str,
        required=True,
        help='Output directory for processed dataset'
    )

    args = parser.parse_args()

    # Handle relative paths
    if not os.path.isabs(args.in_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        args.in_dir = os.path.join(project_dir, args.in_dir)

    if not os.path.isabs(args.out_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        args.out_dir = os.path.join(project_dir, args.out_dir)

    if not os.path.exists(args.in_dir):
        print(f"Error: Input directory not found: {args.in_dir}")
        return 1

    try:
        create_dataset(args.in_dir, args.out_dir)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
