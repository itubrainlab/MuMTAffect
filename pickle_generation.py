#!/usr/bin/env python3
"""
pickle_generation.py

This script processes three different sensor data types – eye tracking, AU (facial expression), and shimmer (GSR, temperature, accelerometer) – using provided processing functions.
It reads participant information from "participants.tsv" in the dataset path, processes each participant’s files, merges the computed features, and saves the merged DataFrame as "dataset.pkl".
"""

import os
import json
import random
import argparse
import pandas as pd
import numpy as np
import neurokit2 as nk
from sklearn.cluster import KMeans

# Global constants
FIXED_LENGTH = 400
CONFIDENCE_THRESHOLD = 0.7  # Only include frames with confidence >= this value

# ====================================================================
# Helper Functions – Eye Processing Section
# ====================================================================

# Use this function for eye data downsampling (it drops unwanted columns).
def downsample_eye_data(df, target_length):
    """
    Downsamples a DataFrame for eye tracking data to a fixed number of rows.
    
    For numerical columns, linear interpolation is used.
    For categorical columns, the nearest neighbor (rounded index) is taken.
    Before downsampling, this function drops unwanted columns.
    """
    drop_columns = [
        'FPOGV', 'LPOGV', 'RPOGV', 'BPOGV',
        'LPV', 'RPV',
        'RPCX', 'RPCY', 'RPD', 'RPS',  # from right eye pupil data
        'LPUPILV', 'RPUPILV',
        'LPD', 'LPS',
        'LPUPILD', 'RPUPILD'
    ]
    df = df.drop(columns=[col for col in drop_columns if col in df.columns])
    
    current_length = len(df)
    if current_length == target_length:
        return df.copy()
    
    old_indices = np.arange(current_length)
    new_indices = np.linspace(0, current_length - 1, target_length)
    new_indices_int = np.round(new_indices).astype(int)
    
    data = {}
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            data[col] = np.interp(new_indices, old_indices, df[col].values)
        else:
            data[col] = df[col].iloc[new_indices_int].values
    return pd.DataFrame(data)

def extract_segment(dataframe, stim_file, start_flag, end_flag):
    filtered_df = dataframe[(dataframe['stim_file'] == stim_file)]
    if filtered_df.empty:
        return None
    initial_timestamp = filtered_df[filtered_df['flag'] == start_flag]['onset'].iloc[0]
    duration = 0
    start = 0
    for flag in filtered_df['flag'].unique():
        if start == 0 and flag != start_flag:
            start = 1
            continue
        flag_filtered_df = filtered_df[filtered_df['flag'] == flag]
        duration += flag_filtered_df.iloc[0]['duration']
        if start == 1 and flag == end_flag:
            break
    end_timestamp = initial_timestamp + duration
    result_df = dataframe[(dataframe['onset'] >= initial_timestamp) & (dataframe['onset'] <= end_timestamp)]
    return result_df

def compute_features(eye_data):
    """
    Compute fixation, blink, saccade and pupil features from the given eye_data.
    """
    features = {}
    trial_time = eye_data['onset'].iloc[-1] - eye_data['onset'].iloc[0]
    # --- FIXATION FEATURES ---
    fix_data = eye_data[eye_data['FPOGV'] == 1].copy()
    if not fix_data.empty:
        features['num_fixations'] = fix_data['FPOGID'].nunique() / trial_time
        features['fixation_duration_mean'] = fix_data['FPOGD'].mean()
        features['fixation_duration_std'] = fix_data['FPOGD'].std()
        dispersion_list = []
        for fixation_id, group in fix_data.groupby('FPOGID'):
            x_std = group['FPOGX'].std()
            y_std = group['FPOGY'].std()
            dispersion = np.sqrt(x_std**2 + y_std**2)
            dispersion_list.append(dispersion)
        features['fixation_dispersion_mean'] = np.mean(dispersion_list) if dispersion_list else np.nan
        features['fixation_dispersion_std'] = np.std(dispersion_list) if dispersion_list else np.nan
    else:
        features['num_fixations'] = 0
        features['fixation_duration_mean'] = np.nan
        features['fixation_duration_std'] = np.nan
        features['fixation_dispersion_mean'] = np.nan
        features['fixation_dispersion_std'] = np.nan

    # --- BLINK FEATURES ---
    # Compute merged pupil sizes (actual_size = diameter * scale)
    eye_data['actual_left_size'] = eye_data['LPD'] * eye_data['LPS']
    eye_data['actual_right_size'] = eye_data['RPD'] * eye_data['RPS']
    eye_data['actual_avg_size'] = np.where(
        (eye_data['RPUPILV'] == 1),
        (eye_data['actual_left_size'] + eye_data['actual_right_size']) / 2,
        np.nan
    )
    blinks = []
    current_blink_start = None
    for idx, row in eye_data.iterrows():
        if row['RPUPILV'] == 0:
            if current_blink_start is None:
                current_blink_start = row['onset']
        else:
            if current_blink_start is not None:
                blink_duration = row['onset'] - current_blink_start
                blinks.append(blink_duration)
                current_blink_start = None
    if blinks:
        features['blink_duration_mean'] = np.mean(blinks)
        features['blink_duration_std'] = np.std(blinks)
        total_time = eye_data['onset'].iloc[-1] - eye_data['onset'].iloc[0]
        features['blink_rate'] = len(blinks) / total_time
    else:
        features['blink_duration_mean'] = np.nan
        features['blink_duration_std'] = np.nan
        features['blink_rate'] = 0

    # --- SACCADE FEATURES ---
    saccades = []
    fix_ids = fix_data['FPOGID'].unique()
    fix_list = []
    for fix_id in fix_ids:
        group = fix_data[fix_data['FPOGID'] == fix_id]
        mean_x = group['BPOGX'].mean() if 'BPOGX' in group.columns else group['FPOGX'].mean()
        mean_y = group['BPOGY'].mean() if 'BPOGY' in group.columns else group['FPOGY'].mean()
        start_time = group['FPOGS'].iloc[0]
        end_time = start_time + group['FPOGD'].iloc[0]
        fix_list.append({'id': fix_id, 'x': mean_x, 'y': mean_y, 'start': start_time, 'end': end_time})
    fix_list = sorted(fix_list, key=lambda x: x['start'])
    for i in range(len(fix_list) - 1):
        current_fix = fix_list[i]
        next_fix = fix_list[i + 1]
        amplitude = np.sqrt((next_fix['x'] - current_fix['x'])**2 + (next_fix['y'] - current_fix['y'])**2)
        duration = next_fix['start'] - current_fix['end']
        velocity = amplitude / duration if duration > 0 else np.nan
        saccades.append({'amplitude': amplitude, 'duration': duration, 'velocity': velocity})
    if saccades:
        amplitudes = [s['amplitude'] for s in saccades]
        durations = [s['duration'] for s in saccades]
        velocities = [s['velocity'] for s in saccades]
        accelerations = []
        for i in range(len(velocities) - 1):
            dt = fix_list[i + 2]['start'] - fix_list[i + 1]['start'] if i + 2 < len(fix_list) else durations[i]
            acc = (velocities[i + 1] - velocities[i]) / dt if dt > 0 else np.nan
            accelerations.append(acc)
        features['saccade_amplitude_mean'] = np.mean(amplitudes)
        features['saccade_duration_mean'] = np.mean(durations)
        features['saccade_velocity_mean'] = np.mean(velocities)
        features['saccade_acceleration_mean'] = np.mean(accelerations) if accelerations else np.nan
        total_time = eye_data['onset'].iloc[-1] - eye_data['onset'].iloc[0]
        features['saccade_rate'] = len(saccades) / total_time
    else:
        features['saccade_amplitude_mean'] = np.nan
        features['saccade_duration_mean'] = np.nan
        features['saccade_velocity_mean'] = np.nan
        features['saccade_acceleration_mean'] = np.nan
        features['saccade_rate'] = 0

    # --- PUPIL FEATURES ---
    eye_data = eye_data.dropna(subset=['actual_avg_size'])
    features['pupil_size_mean'] = eye_data['actual_avg_size'].mean()
    features['pupil_size_std'] = eye_data['actual_avg_size'].std()
    features['pupil_size_min'] = eye_data['actual_avg_size'].min()
    features['pupil_size_max'] = eye_data['actual_avg_size'].max()

    return features

# ====================================================================
# Helper Functions – AU Processing Section (Facial Expression)
# ====================================================================

# Use the generic downsampling function (for AU and shimmer data, we assume the DataFrame contains only the channels to keep).
def downsample_interpolate_numeric_sample_categorical(df, target_length):
    """
    Downsamples a DataFrame to a fixed number of rows.
    Numeric columns are interpolated linearly; non-numeric columns use nearest neighbor.
    """
    current_length = len(df)
    if current_length == target_length:
        return df.copy()

    old_indices = np.arange(current_length)
    new_indices = np.linspace(0, current_length - 1, target_length)
    new_indices_int = np.round(new_indices).astype(int)

    data = {}
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            data[col] = np.interp(new_indices, old_indices, df[col].values)
        else:
            data[col] = df[col].iloc[new_indices_int].values
    return pd.DataFrame(data)

def get_blink_metrics(trial_df):
    """
    Computes blink-related metrics from the binary blink column AU45_c.
    Identifies consecutive frames where AU45_c == 1 as one blink event.
    Returns blink count, blink rate, and average blink duration.
    """
    if "AU45_c" not in trial_df.columns or trial_df.empty:
        return {"blink_count": 0, "blink_rate": 0.0, "blink_duration_mean": 0.0}
    
    blink_col = trial_df["AU45_c"].values
    onset_col = trial_df["onset"].values
    blink_events = []
    in_blink = False
    start_idx = 0
    for i in range(len(blink_col)):
        if blink_col[i] == 1 and not in_blink:
            in_blink = True
            start_idx = i
        elif blink_col[i] == 0 and in_blink:
            in_blink = False
            blink_events.append((start_idx, i - 1))
    if in_blink:
        blink_events.append((start_idx, len(blink_col) - 1))
    
    blink_count = len(blink_events)
    total_duration = onset_col[-1] - onset_col[0] if len(onset_col) > 1 else 1e-6
    durations = []
    for (start, end) in blink_events:
        blink_start_time = onset_col[start]
        blink_end_time = onset_col[end]
        durations.append(blink_end_time - blink_start_time)
    
    avg_blink_duration = np.mean(durations) if durations else 0.0
    blink_rate = blink_count / (total_duration if total_duration > 0 else 1e-6)
    
    return {
        "blink_count": blink_count,
        "blink_rate": blink_rate,
        "blink_duration_mean": avg_blink_duration
    }

def get_dynamic_features(trial_df, intensity_cols):
    """
    Computes dynamic descriptors for each intensity AU column:
      - Peak counts (local maxima)
      - Maximum slopes
    """
    features = {}
    if len(trial_df) < 2:
        for col in intensity_cols:
            features[f"{col}_peak_count"] = 0
            features[f"{col}_max_slope"] = 0.0
        return features
    
    onset = trial_df["onset"].values
    dt = np.diff(onset)
    dt[dt == 0] = 1e-6  # avoid division by zero
    
    for col in intensity_cols:
        series = trial_df[col].values
        peak_count = 0
        for i in range(1, len(series) - 1):
            if series[i] > series[i - 1] and series[i] > series[i + 1]:
                peak_count += 1
        
        slopes = (series[1:] - series[:-1]) / dt
        max_slope = np.max(slopes) if len(slopes) > 0 else 0.0
        
        features[f"{col}_peak_count"] = peak_count
        features[f"{col}_max_slope"] = max_slope
    
    return features

def get_intensity_correlations(trial_df, intensity_cols):
    """
    Computes average correlation among intensity AU channels.
    Returns the mean of the upper-triangular correlation coefficients.
    """
    corr_dict = {}
    if len(intensity_cols) < 2 or trial_df.empty:
        corr_dict["mean_intensity_corr"] = 0.0
        return corr_dict

    data = trial_df[intensity_cols].dropna()
    if data.shape[0] < 2:
        corr_dict["mean_intensity_corr"] = 0.0
        return corr_dict

    valid_cols = [col for col in intensity_cols if data[col].std() > 1e-6]
    if len(valid_cols) < 2:
        corr_dict["mean_intensity_corr"] = 0.0
        return corr_dict

    data_valid = data[valid_cols].values
    with np.errstate(divide='ignore', invalid='ignore'):
        corr_matrix = np.corrcoef(data_valid, rowvar=False)
    corr_matrix = np.nan_to_num(corr_matrix)
    
    idxs = np.triu_indices_from(corr_matrix, k=1)
    mean_corr = np.mean(corr_matrix[idxs])
    corr_dict["mean_intensity_corr"] = mean_corr
    return corr_dict

def get_composite_expressions(trial_df):
    """
    Computes composite expression rates, e.g. smile (AU06_c and AU12_c) and frown (AU04_c and AU15_c).
    Returns the fraction of frames where each expression is active.
    """
    frames_count = len(trial_df)
    if frames_count == 0:
        return {"smile_rate": 0.0, "frown_rate": 0.0}
    
    def safe_col(col):
        return trial_df[col].values if col in trial_df.columns else np.zeros(frames_count)
    
    au06 = safe_col("AU06_c")
    au12 = safe_col("AU12_c")
    smile_frames = np.sum((au06 == 1) & (au12 == 1))

    au04 = safe_col("AU04_c")
    au15 = safe_col("AU15_c")
    frown_frames = np.sum((au04 == 1) & (au15 == 1))
    
    return {
        "smile_rate": smile_frames / frames_count,
        "frown_rate": frown_frames / frames_count
    }

def compute_au_features(trial_df):
    """
    Computes trial-level AU features:
      - Basic statistics for intensity AUs.
      - Activation rates for binary AUs.
      - Blink metrics.
      - Dynamic descriptors.
      - Average correlation among intensity AUs.
      - Composite expression rates.
    """
    features = {}
    
    intensity_cols = [col for col in trial_df.columns if col.endswith("_r")]
    binary_cols = [col for col in trial_df.columns if col.endswith("_c")]
    
    for col in intensity_cols:
        features[f"{col}_mean"] = trial_df[col].mean()
        features[f"{col}_std"]  = trial_df[col].std()
        features[f"{col}_min"]  = trial_df[col].min()
        features[f"{col}_max"]  = trial_df[col].max()
    
    for col in binary_cols:
        features[f"{col}_activation_rate"] = trial_df[col].mean()
    
    if "confidence" in trial_df.columns:
        features["confidence_mean"] = trial_df["confidence"].mean()
    features["n_frames"] = len(trial_df)
    
    blink_features = get_blink_metrics(trial_df)
    features.update(blink_features)
    
    dynamic_features = get_dynamic_features(trial_df, intensity_cols)
    features.update(dynamic_features)
    
    corr_features = get_intensity_correlations(trial_df, intensity_cols)
    features.update(corr_features)
    
    composite_feats = get_composite_expressions(trial_df)
    features.update(composite_feats)
    
    return features

def compute_au_clusters(trial_df, n_clusters=6):
    """
    Computes cluster proportions and the dominant cluster on intensity channels using KMeans.
    """
    intensity_cols = [col for col in trial_df.columns if col.endswith("_r")]
    if len(intensity_cols) == 0:
        return {"cluster_proportions": np.zeros(n_clusters), "dominant_cluster": None}
    
    X = trial_df[intensity_cols].dropna().values
    if X.shape[0] == 0:
        return {"cluster_proportions": np.zeros(n_clusters), "dominant_cluster": None}
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    
    proportions = np.zeros(n_clusters)
    for i in range(n_clusters):
        proportions[i] = np.sum(labels == i) / float(len(labels))
    dominant = int(np.argmax(proportions))
    return {"cluster_proportions": proportions, "dominant_cluster": dominant}

def compute_frame_cluster_onehot(trial_df, n_clusters=6, target_length=FIXED_LENGTH):
    """
    Computes frame-level cluster labels using KMeans on intensity channels,
    downsamples the labels to target_length, and returns one-hot encoded clusters.
    """
    intensity_cols = [col for col in trial_df.columns if col.endswith("_r")]
    if len(intensity_cols) == 0 or trial_df.empty:
        return pd.DataFrame(np.zeros((target_length, n_clusters)), 
                            columns=[f"cluster_{i}_onehot" for i in range(n_clusters)])
    
    X = trial_df[intensity_cols].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    frame_labels = kmeans.fit_predict(X)
    
    down_indices = np.round(np.linspace(0, len(frame_labels) - 1, target_length)).astype(int)
    downsampled_labels = frame_labels[down_indices]
    
    onehot = np.zeros((target_length, n_clusters))
    for i, label in enumerate(downsampled_labels):
        onehot[i, label] = 1
    onehot_df = pd.DataFrame(onehot, columns=[f"cluster_{i}_onehot" for i in range(n_clusters)])
    return onehot_df

# Define columns used in AU processing
FINAL_AU_COLUMNS = [
    "onset", "confidence", "success",
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r",
    "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r",
    "AU25_r", "AU26_r", "AU45_r",
    "AU01_c", "AU02_c", "AU04_c", "AU05_c", "AU06_c", "AU07_c", "AU09_c",
    "AU10_c", "AU12_c", "AU14_c", "AU15_c", "AU17_c", "AU20_c", "AU23_c",
    "AU25_c", "AU26_c", "AU28_c", "AU45_c",
    "duration", "trial_type", "flag", "subject", "run", "trial", "local_time", "stim_file"
]
AU_CHANNELS = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r",
    "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r",
    "AU25_r", "AU26_r", "AU45_r",
    "AU01_c", "AU02_c", "AU04_c", "AU05_c", "AU06_c", "AU07_c", "AU09_c",
    "AU10_c", "AU12_c", "AU14_c", "AU15_c", "AU17_c", "AU20_c", "AU23_c",
    "AU25_c", "AU26_c", "AU28_c", "AU45_c"
]

# ====================================================================
# Helper Functions – Shimmer (GSR, Temperature, Accelerometer) Processing
# ====================================================================

def index_finder(data, start, end):
    start_ind = 0
    end_ind = len(data['SCR_Onsets']) if 'SCR_Onsets' in data else len(next(iter(data.values())))
    for i, peak in enumerate(data.get('SCR_Onsets', [])):
        if peak < start:
            start_ind = i
        if peak < end:
            end_ind = i
    return start_ind, end_ind

def safe_stat(func, arr):
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return np.nan
    return func(valid)

def generate_gsr_report(results, start_ind, end_ind):
    scr_onsets = results.get('SCR_Onsets', np.array([]))
    scr_peaks = results.get('SCR_Peaks', np.array([]))
    scr_height = results.get('SCR_Height', np.array([]))
    scr_amplitude = results.get('SCR_Amplitude', np.array([]))
    scr_rise_time = results.get('SCR_RiseTime', np.array([]))
    scr_recovery = results.get('SCR_Recovery', np.array([]))
    scr_recovery_time = results.get('SCR_RecoveryTime', np.array([]))
    sampling_rate = results.get('sampling_rate', np.nan)
    
    if scr_onsets.size > 0:
        scr_onsets = (scr_onsets - start_ind) / (end_ind - start_ind)
    
    num_peaks = len(scr_onsets)
    
    report = {
        "Number of Peaks": num_peaks,
        "SCR_Onsets mean": safe_stat(np.nanmean, scr_onsets),
        "SCR_Onsets median": safe_stat(np.nanmedian, scr_onsets),
        "SCR_Onsets min": safe_stat(np.nanmin, scr_onsets),
        "SCR_Onsets max": safe_stat(np.nanmax, scr_onsets),
        "SCR_Onsets STD": safe_stat(np.nanstd, scr_onsets),
        
        "SCR_Amplitude mean": safe_stat(np.nanmean, scr_amplitude),
        "SCR_Amplitude median": safe_stat(np.nanmedian, scr_amplitude),
        "SCR_Amplitude min": safe_stat(np.nanmin, scr_amplitude),
        "SCR_Amplitude max": safe_stat(np.nanmax, scr_amplitude),
        "SCR_Amplitude STD": safe_stat(np.nanstd, scr_amplitude),
        
        "SCR_Height mean": safe_stat(np.nanmean, scr_height),
        "SCR_Height median": safe_stat(np.nanmedian, scr_height),
        "SCR_Height min": safe_stat(np.nanmin, scr_height),
        "SCR_Height max": safe_stat(np.nanmax, scr_height),
        "SCR_Height STD": safe_stat(np.nanstd, scr_height),
        
        "SCR_RiseTime mean": safe_stat(np.nanmean, scr_rise_time),
        "SCR_RiseTime median": safe_stat(np.nanmedian, scr_rise_time),
        "SCR_RiseTime min": safe_stat(np.nanmin, scr_rise_time),
        "SCR_RiseTime max": safe_stat(np.nanmax, scr_rise_time),
        "SCR_RiseTime STD": safe_stat(np.nanstd, scr_rise_time),
        
        "SCR_Recovery mean": safe_stat(np.nanmean, scr_recovery),
        "SCR_Recovery median": safe_stat(np.nanmedian, scr_recovery),
        "SCR_Recovery min": safe_stat(np.nanmin, scr_recovery),
        "SCR_Recovery max": safe_stat(np.nanmax, scr_recovery),
        "SCR_Recovery STD": safe_stat(np.nanstd, scr_recovery),
        
        "SCR_RecoveryTime mean": safe_stat(np.nanmean, scr_recovery_time),
        "SCR_RecoveryTime median": safe_stat(np.nanmedian, scr_recovery_time),
        "SCR_RecoveryTime min": safe_stat(np.nanmin, scr_recovery_time),
        "SCR_RecoveryTime max": safe_stat(np.nanmax, scr_recovery_time),
        "SCR_RecoveryTime STD": safe_stat(np.nanstd, scr_recovery_time),
        
        "Sampling Rate": sampling_rate
    }
    return report

def calculate_gsr_metrics_with_dynamic_range(data, flag, sampling_rate, start_delay=2, dwel_time=5):
    samplerate = int(sampling_rate)
    try:
        start_ind = int(start_delay * samplerate) + int(flag.index[flag == "video"][0])
        end_ind = int(flag.index[flag == "last_frame_video"][0]) + int(dwel_time * samplerate)
    except ValueError as e:
        return {"error": f"Could not find required flags: {str(e)}"}
    start_index, end_index = index_finder(data, start_ind, end_ind)
    sliced_data = {key: value[int(start_index):int(end_index)+1] for key, value in data.items() if isinstance(value, np.ndarray)}
    sliced_data['sampling_rate'] = samplerate
    results = generate_gsr_report(sliced_data, start_ind, end_ind)
    return results

def add_shimmer_feature_columns(trial_df):
    trial_df = trial_df.copy()
    trial_df["Acc_mag"] = np.sqrt(
        trial_df["Low_Noise_Accelerometer_X_cal"]**2 +
        trial_df["Low_Noise_Accelerometer_Y_cal"]**2 +
        trial_df["Low_Noise_Accelerometer_Z_cal"]**2
    )
    return trial_df

def compute_temperature_report(trial_df):
    temp = trial_df["Temperature_cal"].dropna().values
    report = {
        "Temperature_mean": np.mean(temp) if len(temp) > 0 else np.nan,
        "Temperature_std": np.std(temp) if len(temp) > 0 else np.nan,
        "Temperature_min": np.min(temp) if len(temp) > 0 else np.nan,
        "Temperature_max": np.max(temp) if len(temp) > 0 else np.nan
    }
    return report

def compute_accelerometer_report(trial_df):
    acc = trial_df["Acc_mag"].dropna().values
    report = {
        "Acc_mag_mean": np.mean(acc) if len(acc) > 0 else np.nan,
        "Acc_mag_std": np.std(acc) if len(acc) > 0 else np.nan,
        "Acc_mag_min": np.min(acc) if len(acc) > 0 else np.nan,
        "Acc_mag_max": np.max(acc) if len(acc) > 0 else np.nan
    }
    return report

# ====================================================================
# Data Processing Functions – One per Sensor Modality
# ====================================================================

def process_eye_data(root_path):
    """
    Processes eye tracking data.
    Reads gaze and pupil JSON/TSV files, merges them with events and labels,
    computes features and downsamples the trial data.
    Returns a DataFrame with computed eye data metrics.
    """
    # Initial fixed header for eye data (original header list)
    orig_headers = ['onset', 'TIME', 'FPOGX', 'FPOGY', 'FPOGS', 'FPOGD', 'FPOGID', 'FPOGV',
                    'LPOGX', 'LPOGY', 'LPOGV', 'RPOGX', 'RPOGY', 'RPOGV', 'BPOGX', 'BPOGY',
                    'BPOGV', 'LPCX', 'LPCY', 'LPD', 'LPS', 'LPV', 'RPCX', 'RPCY', 'RPD',
                    'RPS', 'RPV', 'LEYEX', 'LEYEY', 'LEYEZ', 'LPUPILD', 'LPUPILV', 'REYEX',
                    'REYEY', 'REYEZ', 'RPUPILD', 'RPUPILV', 'duration', 'trial_type',
                    'flag', 'subject', 'run', 'trial', 'local_time', 'stim_file']
    # New simplified header for downsampled eye data
    new_headers = [
        "onset", "TIME", "FPOGX", "FPOGY", "FPOGS", "FPOGD", "FPOGID",
        "LPOGX", "LPOGY",
        "RPOGX", "RPOGY",
        "BPOGX", "BPOGY",
        "LPCX", "LPCY",
        "LEYEX", "LEYEY", "LEYEZ",
        "REYEX", "REYEY", "REYEZ",
        "duration", "trial_type", "flag", "subject", "run", "trial", "local_time", "stim_file",
        "actual_left_size", "actual_right_size", "actual_avg_size"
    ]
    eye_metrics_all = []
    participants_path = os.path.join(root_path, "participants.tsv")
    participants_df = pd.read_csv(participants_path, sep="\t")
    for user in participants_df['participant_id'].unique():
        user_df = participants_df[participants_df['participant_id'] == user]
        openness = user_df['O'].values[0]
        conscientiousness = user_df['C'].values[0]
        extraversion = user_df['E'].values[0]
        agreeableness = user_df['A'].values[0]
        neuroticism = user_df['N'].values[0]
        for run in range(4):
            print(f"Processing eye data for user {user}, run {run}")
            gaze_json_path = os.path.join(root_path, user, "beh", f"{user}_task-fer_run-{run}_recording-gaze_physio.json")
            pupil_json_path = os.path.join(root_path, user, "beh", f"{user}_task-fer_run-{run}_recording-pupil_physio.json")
            gaze_path = os.path.join(root_path, user, "beh", f"{user}_task-fer_run-{run}_recording-gaze_physio.tsv.gz")
            pupil_path = os.path.join(root_path, user, "beh", f"{user}_task-fer_run-{run}_recording-pupil_physio.tsv.gz")
            events_path = os.path.join(root_path, user, f"{user}_task-fer_run-{run}_events.tsv")
            labels_path = os.path.join(root_path, user, "beh", f"{user}_task-fer_run-{run}_beh.tsv")
            if os.path.exists(gaze_json_path) and os.path.exists(pupil_json_path):
                # Read JSON headers for gaze and pupil data
                with open(gaze_json_path, 'r') as file:
                    gaze_json_data = json.load(file)
                gaze_headers = gaze_json_data['Columns']
                with open(pupil_json_path, 'r') as file:
                    pupil_json_data = json.load(file)
                pupil_headers = pupil_json_data['Columns']
                # Load the TSV data
                gaze_data = pd.read_csv(gaze_path, sep='\t', compression='gzip', names=gaze_headers)
                pupil_data = pd.read_csv(pupil_path, sep='\t', compression='gzip', names=pupil_headers).drop(columns=["TIME"])
                # Normalize onset timestamps
                gaze_data['onset'] = gaze_data['onset'] - gaze_data['onset'].iloc[0]
                pupil_data['onset'] = pupil_data['onset'] - pupil_data['onset'].iloc[0]
                gaze_data = gaze_data.dropna(subset=['onset']).sort_values('onset')
                pupil_data = pupil_data.dropna(subset=['onset']).sort_values('onset')
                # Merge gaze and pupil data
                eye_data_merged = pd.merge_asof(gaze_data, pupil_data, on='onset', direction='backward')
                events = pd.read_csv(events_path, sep='\t').dropna(subset=['onset'])
                labels = pd.read_csv(labels_path, sep='\t')
                eye_data_merged = pd.merge_asof(eye_data_merged, events, on='onset', direction='backward')
                eye_data_merged.columns = orig_headers
                # Compute merged pupil sizes
                eye_data_merged['actual_left_size'] = eye_data_merged['LPD'] * eye_data_merged['LPS']
                eye_data_merged['actual_right_size'] = eye_data_merged['RPD'] * eye_data_merged['RPS']
                eye_data_merged['actual_avg_size'] = np.where(
                    (eye_data_merged['RPUPILV'] == 1),
                    (eye_data_merged['actual_left_size'] + eye_data_merged['actual_right_size']) / 2,
                    np.nan
                )
                for stim_file in eye_data_merged['stim_file'].unique():
                    try:
                        metrics = {}
                        # Select trial segment and filter flags
                        eye_trial = eye_data_merged[eye_data_merged['stim_file'] == stim_file].reset_index(drop=True)
                        eye_trial = eye_trial[eye_trial['flag'].isin(
                            ['trial', 'first_fix', 'scenario', 'second_fix', 'video', 'last_frame_video']
                        )]
                        # Compute features and downsample the trial
                        eye_features = compute_features(eye_trial)
                        metrics["Eye_Data_features"] = eye_features
                        eye_trial_down = downsample_eye_data(eye_trial, FIXED_LENGTH)
                        eye_trial_down.columns = new_headers
                        metrics["Eye_Data"] = eye_trial_down

                        metrics['user'] = user
                        metrics['run'] = run
                        metrics['stim_file'] = stim_file
                        label_line = labels[labels['stim_file'] == stim_file]
                        metrics['trial'] = label_line['trial'].values[0]
                        metrics['stim_emo'] = label_line['trial_type'].values[0]
                        metrics['preceived_arousal'] = label_line['p_emotion_a'].values[0]
                        metrics['preceived_valance'] = label_line['p_emotion_v'].values[0]
                        metrics['felt_arousal'] = label_line['f_emotion_a'].values[0]
                        metrics['felt_valance'] = label_line['f_emotion_v'].values[0]
                        metrics['openness'] = openness
                        metrics['conscientiousness'] = conscientiousness
                        metrics['extraversion'] = extraversion
                        metrics['agreeableness'] = agreeableness
                        metrics['neuroticism'] = neuroticism

                        eye_metrics_all.append(metrics)
                    except Exception as e:
                        print(f"Error processing {stim_file} for user {user} run {run}: {e}")
            else:
                print(f"Missing eye data for user {user} run {run}")
    eye_df = pd.DataFrame(eye_metrics_all)
    return eye_df

def process_au_data(root_path):
    """
    Processes AU (facial expression) data.
    Reads the videostream JSON/TSV files, merges with events and labels,
    computes AU-level features, downsamples the data, and collects trial metadata.
    Returns a DataFrame with the computed AU metrics.
    """
    au_metrics_all = []
    participants_path = os.path.join(root_path, "participants.tsv")
    participants_df = pd.read_csv(participants_path, sep="\t")
    for user in participants_df['participant_id'].unique():
        user_df = participants_df[participants_df['participant_id'] == user]
        openness = user_df['O'].values[0]
        conscientiousness = user_df['C'].values[0]
        extraversion = user_df['E'].values[0]
        agreeableness = user_df['A'].values[0]
        neuroticism = user_df['N'].values[0]
        for run in range(4):
            print(f"Processing AU data for user {user}, run {run}")
            json_path = os.path.join(root_path, user, "beh", f"{user}_task-fer_run-{run}_recording-videostream_physio.json")
            if os.path.exists(json_path):
                au_path = os.path.join(root_path, user, "beh", f"{user}_task-fer_run-{run}_recording-videostream_physio.tsv.gz")
                events_path = os.path.join(root_path, user, f"{user}_task-fer_run-{run}_events.tsv")
                labels_path = os.path.join(root_path, user, "beh", f"{user}_task-fer_run-{run}_beh.tsv")
                
                with open(json_path, 'r') as file:
                    json_data = json.load(file)
                headers = json_data['Columns']
                
                au_data = pd.read_csv(au_path, sep='\t', compression='gzip', names=headers)
                au_data["onset"] = pd.to_numeric(au_data["onset"], errors="coerce")
                for col in AU_CHANNELS + ["confidence"]:
                    au_data[col] = pd.to_numeric(au_data[col], errors="coerce")
                
                au_data = au_data.dropna(subset=["onset"]).sort_values("onset")
                if au_data.empty:
                    print(f"No valid onset data for AU user {user} run {run}. Skipping.")
                    continue
                
                au_data["onset"] = au_data["onset"] - au_data["onset"].iloc[0]
                events = pd.read_csv(events_path, sep="\t")
                events["onset"] = pd.to_numeric(events["onset"], errors="coerce")
                events = events.dropna(subset=["onset"]).sort_values("onset")
                labels = pd.read_csv(labels_path, sep='\t')
                
                try:
                    au_data_merged = pd.merge_asof(au_data, events, on="onset", direction="backward")
                except Exception as e:
                    print(f"Merge error for AU user {user} run {run}: {e}")
                    continue
                
                au_data_clean = au_data_merged[FINAL_AU_COLUMNS].reset_index(drop=True)
                
                for stim_file in au_data_clean["stim_file"].unique():
                    try:
                        metrics = {}
                        au_trial = au_data_clean[au_data_clean["stim_file"] == stim_file].reset_index(drop=True)
                        au_trial = au_trial[au_trial["flag"].isin(
                            ["trial", "first_fix", "scenario", "second_fix", "video", "last_frame_video"]
                        )].reset_index(drop=True)
                        if au_trial.empty:
                            print(f"No trial data for {stim_file} AU user {user} run {run}. Skipping trial.")
                            continue
                        
                        au_features = compute_au_features(au_trial)
                        metrics["AUs_features"] = au_features
                        
                        au_trial_down = downsample_interpolate_numeric_sample_categorical(au_trial, FIXED_LENGTH)
                        metrics["AUs"] = au_trial_down
                        
                        metrics["user"] = user
                        metrics["run"] = run
                        metrics["stim_file"] = stim_file
                        label_line = labels[labels["stim_file"] == stim_file]
                        if label_line.empty:
                            print(f"No label found for {stim_file} AU user {user} run {run}. Skipping trial.")
                            continue
                        metrics["trial"] = label_line["trial"].values[0]
                        metrics["stim_emo"] = label_line["trial_type"].values[0]
                        metrics["preceived_arousal"] = label_line["p_emotion_a"].values[0]
                        metrics["preceived_valance"] = label_line["p_emotion_v"].values[0]
                        metrics["felt_arousal"] = label_line["f_emotion_a"].values[0]
                        metrics["felt_valance"] = label_line["f_emotion_v"].values[0]
                        metrics["openness"] = openness
                        metrics["conscientiousness"] = conscientiousness
                        metrics["extraversion"] = extraversion
                        metrics["agreeableness"] = agreeableness
                        metrics["neuroticism"] = neuroticism
                        
                        au_metrics_all.append(metrics)
                    except Exception as e:
                        print(f"Error processing {stim_file} for AU user {user} run {run}: {e}")
            else:
                print(f"Missing AU data for user {user} run {run}")
    au_df = pd.DataFrame(au_metrics_all)
    return au_df

def process_shimmer_data(root_path):
    """
    Processes shimmer data (GSR, Temperature, Accelerometer).
    Reads shimmer JSON/TSV files, merges with events/labels, processes sensor data via NeuroKit2,
    computes sensor reports, downsamples the data, and collects trial-level metadata.
    Returns a DataFrame with the computed shimmer metrics.
    """
    shimmer_metrics_all = []
    FINAL_SHIMMER_COLUMNS = [
        "onset", "Timestamp_raw", "Timestamp_cal", "System_Timestamp_cal",
        "Low_Noise_Accelerometer_X_cal", "Low_Noise_Accelerometer_Y_cal", "Low_Noise_Accelerometer_Z_cal",
        "Wide_Range_Accelerometer_X_cal", "Wide_Range_Accelerometer_Y_cal", "Wide_Range_Accelerometer_Z_cal",
        "Gyroscope_X_cal", "Gyroscope_Y_cal", "Gyroscope_Z_cal",
        "Magnetometer_X_cal", "Magnetometer_Y_cal", "Magnetometer_Z_cal",
        "VSenseBatt_cal",
        "External_ADC_A7_cal", "Internal_ADC_A13_cal",
        "Pressure_cal", "Temperature_cal", 
        "GSR_raw", "GSR_cal", "GSR_Conductance_cal",
        "duration", "trial_type", "flag", "subject", "run", "trial", "local_time", "stim_file"
    ]
    participants_path = os.path.join(root_path, "participants.tsv")
    participants_df = pd.read_csv(participants_path, sep="\t")
    for user in participants_df['participant_id'].unique():
        print(f"Processing shimmer data for user {user}")
        user_df = participants_df[participants_df['participant_id'] == user]
        openness = user_df['O'].values[0]
        conscientiousness = user_df['C'].values[0]
        extraversion = user_df['E'].values[0]
        agreeableness = user_df['A'].values[0]
        neuroticism = user_df['N'].values[0]
        for run in range(4):
            print(f"Processing shimmer data for user {user}, run {run}")
            json_path = os.path.join(root_path, user, "beh", f"{user}_task-fer_run-{run}_recording-gsr_physio.json")
            if os.path.exists(json_path):
                shimmer_path = os.path.join(root_path, user, "beh", f"{user}_task-fer_run-{run}_recording-gsr_physio.tsv.gz")
                events_path = os.path.join(root_path, user, f"{user}_task-fer_run-{run}_events.tsv")
                labels_path = os.path.join(root_path, user, "beh", f"{user}_task-fer_run-{run}_beh.tsv")
                with open(json_path, 'r') as file:
                    json_data = json.load(file)
                headers = json_data['Columns']
                
                shimmer_data = pd.read_csv(shimmer_path, sep='\t', compression='gzip', names=headers)
                shimmer_data["onset"] = pd.to_numeric(shimmer_data["onset"], errors="coerce")
                shimmer_data["Temperature_cal"] = pd.to_numeric(shimmer_data["Temperature_cal"], errors="coerce")
                shimmer_data["GSR_Conductance_cal"] = pd.to_numeric(shimmer_data["GSR_Conductance_cal"], errors="coerce")
                for col in ["Low_Noise_Accelerometer_X_cal", "Low_Noise_Accelerometer_Y_cal", "Low_Noise_Accelerometer_Z_cal"]:
                    shimmer_data[col] = pd.to_numeric(shimmer_data[col], errors="coerce")
                
                shimmer_data = shimmer_data.dropna(subset=["onset"]).sort_values("onset")
                if shimmer_data.empty:
                    print(f"No valid onset data for shimmer user {user} run {run}. Skipping.")
                    continue
                shimmer_data["onset"] = shimmer_data["onset"] - shimmer_data["onset"].iloc[0]
                
                events = pd.read_csv(events_path, sep="\t")
                events["onset"] = pd.to_numeric(events["onset"], errors="coerce")
                events = events.dropna(subset=["onset"]).sort_values("onset")
                labels = pd.read_csv(labels_path, sep='\t')
                
                try:
                    shimmer_data_merged = pd.merge_asof(shimmer_data, events, on="onset", direction="backward")
                except Exception as e:
                    print(f"Merge error for shimmer user {user} run {run}: {e}")
                    continue
                
                shimmer_data_clean = shimmer_data_merged[FINAL_SHIMMER_COLUMNS]
                
                for stim_file in shimmer_data_clean["stim_file"].unique():
                    try:
                        metrics = {}
                        trial_df = shimmer_data_clean[shimmer_data_clean["stim_file"] == stim_file].reset_index(drop=True)
                        trial_df = trial_df[trial_df["flag"].isin(
                            ["trial", "first_fix", "scenario", "second_fix", "video", "last_frame_video"]
                        )]
                        if trial_df.empty:
                            print(f"No trial data for {stim_file} shimmer user {user} run {run}. Skipping trial.")
                            continue
                        
                        # Process GSR using NeuroKit2
                        gsr_signal = nk.standardize(trial_df["GSR_Conductance_cal"].values)
                        trial_duration = trial_df["onset"].iloc[-1] - trial_df["onset"].iloc[0]
                        sampling_rate = int(len(trial_df["onset"]) / trial_duration) if trial_duration > 0 else 100
                        data, result = nk.eda_process(gsr_signal, sampling_rate=sampling_rate, method='neurokit')
                        tonic = data["EDA_Tonic"]
                        phasic = data["EDA_Phasic"]
                        trial_df["EDA_Tonic"] = tonic
                        trial_df["EDA_Phasic"] = phasic
                        gsr_report = calculate_gsr_metrics_with_dynamic_range(result, trial_df["flag"], sampling_rate, start_delay=0, dwel_time=10)
                        
                        # Compute Temperature and Accelerometer reports
                        temp_report = compute_temperature_report(trial_df)
                        trial_df = add_shimmer_feature_columns(trial_df)
                        acc_report = compute_accelerometer_report(trial_df)
                        
                        sensor_reports = {
                            "GSR_report": gsr_report,
                            "Temperature_report": temp_report,
                            "Accelerometer_report": acc_report
                        }
                        
                        trial_down = downsample_interpolate_numeric_sample_categorical(trial_df, FIXED_LENGTH)
                        
                        metrics["Shimmer_features"] = sensor_reports
                        metrics["Shimmer"] = trial_down
                        metrics["user"] = user
                        metrics["run"] = run
                        metrics["stim_file"] = stim_file
                        label_line = labels[labels["stim_file"] == stim_file]
                        if label_line.empty:
                            print(f"No label found for {stim_file} shimmer user {user} run {run}. Skipping trial.")
                            continue
                        metrics["trial"] = label_line["trial"].values[0]
                        metrics["stim_emo"] = label_line["trial_type"].values[0]
                        metrics["preceived_arousal"] = label_line["p_emotion_a"].values[0]
                        metrics["preceived_valance"] = label_line["p_emotion_v"].values[0]
                        metrics["felt_arousal"] = label_line["f_emotion_a"].values[0]
                        metrics["felt_valance"] = label_line["f_emotion_v"].values[0]
                        metrics["openness"] = openness
                        metrics["conscientiousness"] = conscientiousness
                        metrics["extraversion"] = extraversion
                        metrics["agreeableness"] = agreeableness
                        metrics["neuroticism"] = neuroticism
                        
                        shimmer_metrics_all.append(metrics)
                    except Exception as e:
                        print(f"Error processing {stim_file} for shimmer user {user} run {run}: {e}")
            else:
                print(f"Missing shimmer data for user {user} run {run}")
    shimmer_df = pd.DataFrame(shimmer_metrics_all)
    return shimmer_df

# ====================================================================
# Merge and Save Dataset
# ====================================================================

def merge_data(eye_df, au_df, shimmer_df):
    """
    Merges the eye, AU, and shimmer DataFrames.
    Drops personality columns from eye and AU data and then performs inner merges based on shared columns.
    Returns the merged DataFrame.
    """
    eye_df_pure = eye_df.drop(columns=['openness', 'conscientiousness', 'extraversion',
                                         'agreeableness', 'neuroticism'])
    au_df_pure = au_df.drop(columns=['openness', 'conscientiousness', 'extraversion',
                                       'agreeableness', 'neuroticism'])
    merge_keys = ['user', 'run', 'stim_file', 'trial', 'stim_emo',
                  'preceived_arousal', 'preceived_valance', 'felt_arousal', 'felt_valance']
    merged = pd.merge( pd.merge(eye_df_pure, au_df_pure, on=merge_keys, how='inner'),
                       shimmer_df, on=['user', 'run', 'stim_file', 'stim_emo',
                                        'preceived_arousal', 'preceived_valance', 'felt_arousal', 'felt_valance'],
                       how='inner')
    return merged

# ====================================================================
# Main function and Argument Parsing
# ====================================================================

def main(dataset_path):
    # Process each sensor modality
    print("Starting eye data processing...")
    eye_df = process_eye_data(dataset_path)
    print("Completed eye data processing.\n")
    
    print("Starting AU data processing...")
    au_df = process_au_data(dataset_path)
    print("Completed AU data processing.\n")
    
    print("Starting shimmer data processing...")
    shimmer_df = process_shimmer_data(dataset_path)
    print("Completed shimmer data processing.\n")
    
    # Merge the data
    print("Merging data...")
    merged_df = merge_data(eye_df, au_df, shimmer_df)
    
    # Save to pickle in the dataset directory
    pickle_path = os.path.join(dataset_path, "dataset.pkl")
    merged_df.to_pickle(pickle_path)
    print(f"Pickle file saved to: {pickle_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate merged pickle dataset from sensor data.")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the dataset folder (should contain participants.tsv and participant subdirectories)")
    args = parser.parse_args()
    main(args.dataset_path)
