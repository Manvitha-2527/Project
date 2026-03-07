#!/usr/bin/env python3
"""
vis.py - Visualization script for sleep apnea physiological signals

This script generates PDF visualizations of Nasal Airflow, Thoracic Movement,
and SpO2 signals with annotated breathing events overlaid.

Usage:
    python vis.py -name "Data/AP01"
"""

import argparse
import os
import glob
import re
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages


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
                # Parse metadata
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
                # Parse data line: "30.05.2024 20:59:00,000; 120"
                if ';' in line:
                    parts = line.split(';')
                    timestamp_str = parts[0].strip()
                    value = float(parts[1].strip())
                    data_lines.append((timestamp_str, value))

    # Convert to DataFrame
    df = pd.DataFrame(data_lines, columns=['timestamp_str', 'value'])

    # Parse timestamps - handle format "DD.MM.YYYY HH:MM:SS,mmm"
    def parse_timestamp(ts):
        # Replace comma with dot for milliseconds
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
    events = []
    in_data_section = False

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Skip header lines (lines that contain ':' at the start for metadata)
            if ':' in line and not any(c.isdigit() for c in line.split(':')[0][:2]):
                if line.startswith('Signal ID:') or line.startswith('Start Time:') or \
                   line.startswith('Unit:') or line.startswith('Signal Type:'):
                    continue

            # Data lines have format: "30.05.2024 23:48:45,119-23:49:01,408; 16;Hypopnea; N1"
            # Pattern: DATE TIME1-TIME2; DURATION;EVENT_TYPE; SLEEP_STAGE
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

                # Handle end time - might be next day if time is earlier
                end_datetime = datetime.strptime(f"{date} {end_time_str}", '%d.%m.%Y %H:%M:%S.%f')
                if end_datetime < start_datetime:
                    # Crossed midnight
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


def create_visualization(participant_dir, output_dir):
    """
    Create a PDF visualization of signals with event overlays.

    Args:
        participant_dir: Path to participant data directory
        output_dir: Path to output directory for PDF
    """
    # Get participant name
    participant_name = os.path.basename(os.path.normpath(participant_dir))

    print(f"Processing participant: {participant_name}")

    # Find all required files
    files = find_files(participant_dir)

    required_files = ['flow', 'thorac', 'spo2', 'events']
    for req in required_files:
        if req not in files:
            raise FileNotFoundError(f"Could not find {req} file in {participant_dir}")

    print(f"  Loading Flow signal...")
    flow_meta, flow_df = parse_signal_file(files['flow'])

    print(f"  Loading Thoracic signal...")
    thorac_meta, thorac_df = parse_signal_file(files['thorac'])

    print(f"  Loading SpO2 signal...")
    spo2_meta, spo2_df = parse_signal_file(files['spo2'])

    print(f"  Loading Events...")
    events_df = parse_events_file(files['events'])

    print(f"  Loaded {len(flow_df)} flow samples, {len(thorac_df)} thorac samples, "
          f"{len(spo2_df)} SpO2 samples, {len(events_df)} events")

    # Convert timestamps to hours from start for plotting
    start_time = flow_df['timestamp'].min()

    flow_df['hours'] = (flow_df['timestamp'] - start_time).dt.total_seconds() / 3600
    thorac_df['hours'] = (thorac_df['timestamp'] - start_time).dt.total_seconds() / 3600
    spo2_df['hours'] = (spo2_df['timestamp'] - start_time).dt.total_seconds() / 3600

    if len(events_df) > 0:
        events_df['start_hours'] = (events_df['start_time'] - start_time).dt.total_seconds() / 3600
        events_df['end_hours'] = (events_df['end_time'] - start_time).dt.total_seconds() / 3600

    # Define colors for event types
    event_colors = {
        'Hypopnea': 'orange',
        'Obstructive Apnea': 'red',
        'Central Apnea': 'purple',
        'Mixed Apnea': 'brown'
    }

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f'Sleep Study Signals - Participant {participant_name}', fontsize=14, fontweight='bold')

    # Plot Nasal Airflow
    ax1 = axes[0]
    ax1.plot(flow_df['hours'], flow_df['value'], 'b-', linewidth=0.3, alpha=0.8)
    ax1.set_ylabel('Nasal Airflow\n(arbitrary units)', fontsize=10)
    ax1.set_title('Nasal Airflow (32 Hz)', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot Thoracic Movement
    ax2 = axes[1]
    ax2.plot(thorac_df['hours'], thorac_df['value'], 'g-', linewidth=0.3, alpha=0.8)
    ax2.set_ylabel('Thoracic Movement\n(arbitrary units)', fontsize=10)
    ax2.set_title('Thoracic Movement (32 Hz)', fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Plot SpO2
    ax3 = axes[2]
    ax3.plot(spo2_df['hours'], spo2_df['value'], 'r-', linewidth=0.5, alpha=0.8)
    ax3.set_ylabel('SpO2 (%)', fontsize=10)
    ax3.set_xlabel('Time (hours from start)', fontsize=10)
    ax3.set_title('Oxygen Saturation (4 Hz)', fontsize=11)
    ax3.set_ylim([70, 100])  # Typical SpO2 range
    ax3.grid(True, alpha=0.3)

    # Overlay events on all plots
    if len(events_df) > 0:
        for ax in axes:
            for _, event in events_df.iterrows():
                event_type = event['event_type']
                color = event_colors.get(event_type, 'gray')
                ax.axvspan(event['start_hours'], event['end_hours'],
                          alpha=0.3, color=color, linewidth=0)

    # Create legend for events
    legend_patches = [mpatches.Patch(color=color, alpha=0.3, label=event_type)
                     for event_type, color in event_colors.items()
                     if len(events_df) > 0 and event_type in events_df['event_type'].values]

    if legend_patches:
        fig.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(0.99, 0.98),
                  title='Breathing Events', fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save to PDF
    output_path = os.path.join(output_dir, f'{participant_name}_visualization.pdf')

    print(f"  Saving visualization to {output_path}")
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, dpi=150, bbox_inches='tight')

    plt.close(fig)
    print(f"  Done!")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate PDF visualization of sleep study signals with breathing events'
    )
    parser.add_argument(
        '-name',
        type=str,
        required=True,
        help='Path to participant data folder (e.g., "Data/AP01")'
    )
    parser.add_argument(
        '-output',
        type=str,
        default='Visualizations',
        help='Output directory for PDF files (default: Visualizations)'
    )

    args = parser.parse_args()

    # Handle relative paths
    if not os.path.isabs(args.name):
        # Check if the path exists as-is or needs to be relative to script location
        if not os.path.exists(args.name):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(script_dir)
            args.name = os.path.join(project_dir, args.name)

    if not os.path.isabs(args.output):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        args.output = os.path.join(project_dir, args.output)

    if not os.path.exists(args.name):
        print(f"Error: Participant directory not found: {args.name}")
        return 1

    try:
        output_path = create_visualization(args.name, args.output)
        print(f"\nVisualization saved to: {output_path}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
