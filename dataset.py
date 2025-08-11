# dataset.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.signal import resample  # For time warping



# Example column definitions (customize as needed)
def process_stim_emo(stim_emo):
    stim_emo_categories = ['neutral','sad','happy','angry','fear','disgust']
    vec = np.zeros(len(stim_emo_categories), dtype=np.float32)
    if stim_emo in stim_emo_categories:
        vec[stim_emo_categories.index(stim_emo)] = 1.0
    return vec

# Example column definitions (customize as needed)
common_flag_categories = ['trial','first_fix','scenario','second_fix','video','last_frame_video',
                          'f_emotion_labelling','p_emotion_labelling']
gaze_cols = ['onset','FPOGX','FPOGY','FPOGS','FPOGD','FPOGID','LPOGX','LPOGY','RPOGX','RPOGY',
             'BPOGX','BPOGY','LPCX','LPCY','LEYEX','LEYEY','LEYEZ','REYEX','REYEY','REYEZ','flag']
pupil_cols = ['onset','actual_left_size','actual_right_size','actual_avg_size','flag']
desired_au_cols = ['onset','confidence','success','AU01_r','AU02_r','AU04_r','AU05_r','AU06_r',
                   'AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r',
                   'AU25_r','AU26_r','AU45_r','AU01_c','AU02_c','AU04_c','AU05_c','AU06_c','AU07_c',
                   'AU09_c','AU10_c','AU12_c','AU14_c','AU15_c','AU17_c','AU20_c','AU23_c','AU25_c',
                   'AU26_c','AU28_c','AU45_c','flag']
desired_gsr_cols = ['onset','Pressure_cal','Temperature_cal','GSR_raw','GSR_cal','EDA_Tonic',
                    'EDA_Phasic','flag']

def process_modal_data(modal_df, desired_cols, flag_categories):
    """
    Extracts the desired columns, fills missing values with 0,
    and expands the 'flag' column into one-hot columns if present.
    Returns a NumPy array of type float32.
    """
    df = modal_df[desired_cols].copy()
    df.fillna(0, inplace=True)
    if 'flag' in df.columns:
        flag_dummies = pd.get_dummies(df['flag'], prefix='flag')
        expected_flag_cols = [f'flag_{cat}' for cat in flag_categories]
        flag_dummies = flag_dummies.reindex(columns=expected_flag_cols, fill_value=0)
        df.drop(columns=['flag'], inplace=True)
        df = pd.concat([df, flag_dummies], axis=1)
    return df.values.astype(np.float32)

def flatten_dict_values(d):
    flat_vals = []
    for key in sorted(d.keys()):
        val = d[key]
        if isinstance(val, (list, np.ndarray)):
            flat_vals.append(np.mean(val))
        else:
            flat_vals.append(val)
    return np.array(flat_vals, dtype=np.float32)

def flatten_dict_values(d):
    flat_vals = []
    for key in sorted(d.keys()):
        val = d[key]
        if isinstance(val, (list, np.ndarray)):
            flat_vals.append(np.mean(val))
        else:
            flat_vals.append(val)
    return np.array(flat_vals, dtype=np.float32)

def process_stim_emo(stim_emo):
    stim_emo_categories = ['neutral','sad','happy','angry','fear','disgust']
    vec = np.zeros(len(stim_emo_categories), dtype=np.float32)
    if stim_emo in stim_emo_categories:
        vec[stim_emo_categories.index(stim_emo)] = 1.0
    return vec

# ------------------------------
# Augmentation Functions
# ------------------------------
def time_warp(signal, warp_factor_range=(0.9, 1.1)):
    """
    Speed up or slow down the sequence.
    signal: np.array of shape (T, feature_dim)
    warp_factor_range: tuple with min and max scaling factors.
    """
    warp_factor = np.random.uniform(*warp_factor_range)
    new_length = max(1, int(signal.shape[0] * warp_factor))
    warped_signal = resample(signal, new_length, axis=0)
    return warped_signal

def random_crop(signal, crop_size):
    """
    Randomly crop a contiguous segment from the sequence.
    signal: np.array of shape (T, feature_dim)
    crop_size: desired output length.
    """
    if signal.shape[0] <= crop_size:
        return signal
    start = np.random.randint(0, signal.shape[0] - crop_size)
    return signal[start:start+crop_size]

def noise_injection(signal, noise_std=0.01):
    """
    Add Gaussian noise to the signal.
    signal: np.array of shape (T, feature_dim)
    noise_std: standard deviation of the noise.
    """
    noise = np.random.normal(loc=0.0, scale=noise_std, size=signal.shape)
    return signal + noise

# ------------------------------
# Mixup Augmentation Function
# ------------------------------
def mixup_samples(sample1, sample2, lam):
    """
    Combines two samples with mixing coefficient lam.
    sample1 and sample2 are tuples containing:
      (eye_seq, pupil_seq, au_seq, gsr_seq, stim_emo_vec,
       personality, emotion_binned, user_id, eye_features, au_features, shimmer_features, gender)
    
    For numeric tensor inputs (e.g. sequences, personality), we apply:
       lam * sample1 + (1-lam) * sample2.
    For categorical labels (emotion_binned, gender), we convert them to float (one-hot if needed) and mix.
    For user_id, we set to -1 indicating a mixed sample.
    """
    mixed = []
    for i, (a, b) in enumerate(zip(sample1, sample2)):
        # For the user_id (assumed to be at index 7), set mixed value to -1.
        if i == 7:
            mixed.append(torch.tensor(-1, dtype=a.dtype))
        # For gender (assumed to be at index 11) and emotion_binned (assumed index 6), we convert to float.
        elif i in [6, 11]:
            mixed_val = lam * a.float() + (1 - lam) * b.float()
            mixed.append(mixed_val)
        # For other numeric modalities and personality (continuous), mix directly.
        else:
            # Check if it's a tensor.
            if isinstance(a, torch.Tensor):
                mixed.append(lam * a + (1 - lam) * b)
            else:
                mixed.append(lam * a + (1 - lam) * b)
    return tuple(mixed)

# ------------------------------
# Updated MultiModalDataset with Mixup
# ------------------------------
class MultiModalDataset(Dataset):
    def __init__(self, df, selected_emotions, user2idx, modality_scalers=None, device=None):
        """
        Args:
          df: A DataFrame containing the dataset.
          selected_emotions: List of column names for the emotion labels.
          user2idx: Dictionary mapping user IDs to indices.
          modality_scalers: Optional dictionary with keys for modalities (e.g., 'Eye_Data', 'AUs', 'Shimmer')
                            and values as fitted scaler objects (e.g., StandardScaler) to normalize the data.
          device: torch.device object specifying where the tensors should be allocated (e.g., torch.device('cuda'))
        """
        self.df = df.reset_index(drop=True)
        self.selected_emotions = selected_emotions
        self.user2idx = user2idx
        self.modality_scalers = modality_scalers
        self.device = device if device is not None else torch.device('cpu')
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Process each modality.
        eye_seq = process_modal_data(row['Eye_Data'], gaze_cols, common_flag_categories)
        pupil_seq = process_modal_data(row['Eye_Data'], pupil_cols, common_flag_categories)
        au_seq = process_modal_data(row['AUs'], desired_au_cols, common_flag_categories)
        gsr_seq = process_modal_data(row['Shimmer'], desired_gsr_cols, common_flag_categories)
        
        # Optionally normalize each modality if scalers are provided.
        if self.modality_scalers is not None:
            if 'Eye_Data' in self.modality_scalers:
                eye_seq = self.modality_scalers['Eye_Data'].transform(eye_seq)
            if 'AUs' in self.modality_scalers:
                au_seq = self.modality_scalers['AUs'].transform(au_seq)
            if 'Shimmer' in self.modality_scalers:
                gsr_seq = self.modality_scalers['Shimmer'].transform(gsr_seq)
        
        stim_emo_vec = (process_stim_emo(row['stim_emo']) 
                        if pd.notna(row.get('stim_emo', None)) 
                        else np.zeros(6, dtype=np.float32))
        personality = row[['openness','conscientiousness','extraversion','agreeableness','neuroticism']].values.astype(np.float32)
        
        # Convert gender from 'f'/'m' to 0/1.
        gender_val = row['gender']
        if isinstance(gender_val, str):
            if gender_val.lower() == 'f':
                gender_int = 0
            elif gender_val.lower() == 'm':
                gender_int = 1
            else:
                raise ValueError(f"Unexpected gender value: {gender_val}")
        else:
            gender_int = int(gender_val)
        
        # Create tensors directly on the target device.
        gender = torch.tensor(gender_int, dtype=torch.long, device=self.device)
        emotion_raw = row[self.selected_emotions].values.astype(np.int64) - 1
        emotion_binned = emotion_raw // 3
        user_id = self.user2idx[row['user']]
        eye_features = np.array(list(row['Eye_Data_features'].values()), dtype=np.float32)
        au_features = flatten_dict_values(row['AUs_features'])
        gsr_report = row['Shimmer_features']['GSR_report']
        temp_report = row['Shimmer_features']['Temperature_report']
        acc_report = row['Shimmer_features']['Accelerometer_report']
        shimmer_features = np.concatenate([
            np.array(list(gsr_report.values()), dtype=np.float32),
            np.array(list(temp_report.values()), dtype=np.float32),
            np.array(list(acc_report.values()), dtype=np.float32)
        ])
        
        return (torch.tensor(eye_seq, device=self.device),
                torch.tensor(pupil_seq, device=self.device),
                torch.tensor(au_seq, device=self.device),
                torch.tensor(gsr_seq, device=self.device),
                torch.tensor(stim_emo_vec, device=self.device),
                torch.tensor(personality, device=self.device),
                torch.tensor(emotion_binned, dtype=torch.long, device=self.device),
                torch.tensor(user_id, dtype=torch.long, device=self.device),
                torch.tensor(eye_features, device=self.device),
                torch.tensor(au_features, device=self.device),
                torch.tensor(shimmer_features, device=self.device),
                gender)