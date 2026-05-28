import os
import re
import pyxdf
import mne
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, detrend, resample, welch
from sklearn.preprocessing import LabelEncoder
from mne_icalabel import label_components

# (channels, points, trials)

RAW_DIR = r'C:\Users\danic\Desktop\TFM\Code\Experiment\data\silent'
PROCESSED_DIR = r'C:\Users\danic\Desktop\TFM\Code\Experiment\processed'
os.makedirs(PROCESSED_DIR, exist_ok=True)
LABEL_MAP = {
    '1 (ONE)': 0,
    '2 (TWO)': 1,
    '3 (THREE)': 2,
    '4 (FOUR)': 3,
    '5 (FIVE)': 4,
    'REJOIN': 5,
    'ASSET': 6,
    'ABORT': 7,
    'FORMATION': 8,
    'REFUEL': 9
}

# Set log level to 'WARNING' to hide INFO messages like filter parameters
# Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
mne.set_log_level('WARNING')

def load_session(path):
    """
    Loads the .xdf file and returns a multi-stream dictionary  
    classified by each type of stream (EEG, EMG, IMU, Psychopy).
    """
    streams, header = pyxdf.load_xdf(path)

    data = {'eeg': None, 'emg': None, 'imu': None, 'markers': None}
    imu_streams = {}
    for s in streams:
        name = s['info']['name'][0]
        if name == 'e32_speech_eeg' or name == 'eeg_32_speech_eeg':     # Subject 13 changes name of stream
            data["eeg"] = s
        elif name == 'emg-imu_exg':
            data["emg"] = s
        elif 'emg-imu_daux' in name.casefold():
            imu_streams[name] = s
        elif name == 'PsychoPy_Markers':
            data["markers"] = s
    
    data['imu'] = imu_streams
    return data


def get_epochs(eeg, emg, imu, markers, window_pre=0.5, window_post=2.5, offset=None):
    """
    Extracts epoched windows around every 'Cue_Start:' marker.
    Skips any trial where a modality returns an empty slice.
    """
    marker_stream = markers
    all_eeg, all_emg, all_imu, all_labels = [], [], [], []
    skipped = 0

    for i, marker_val in enumerate(marker_stream['time_series']):
        label_text = marker_val[0]

        if not label_text.startswith('Cue_Start:'):
            continue

        t_marker = marker_stream['time_stamps'][i]
        t_zero = t_marker
        t_start = t_zero - window_pre
        t_end = t_zero + window_post

        # --- Slice each modality ---
        eeg_slice = slice_by_time(eeg, t_start, t_end, t0=offset)
        emg_slice = slice_by_time(emg, t_start, t_end)

        imu_epoch = {}
        imu_valid = True
        for key, stream in imu.items():
            imu_slice = slice_by_time(stream, t_start, t_end)
            if imu_slice.shape[0] == 0:
                imu_valid = False
                break
            imu_epoch[key] = imu_slice

        # --- Validate: skip if ANY modality is empty ---
        if eeg_slice.shape[0] == 0 or emg_slice.shape[0] == 0 or not imu_valid:
            skipped += 1
            print(f"  [SKIP] Trial {i}: empty slice "
                  f"(EEG={eeg_slice.shape[0]}, EMG={emg_slice.shape[0]}, "
                  f"t_marker={t_marker:.2f})")
            continue

        all_eeg.append(eeg_slice)
        all_emg.append(emg_slice)
        all_imu.append(imu_epoch)
        all_labels.append(LABEL_MAP[label_text.replace("Cue_Start: ", "")])

    if skipped > 0:
        print(f"  [INFO] Skipped {skipped} trials with empty slices "
              f"({len(all_labels)} trials kept)")

    if len(all_eeg) == 0:
        raise ValueError("All trials were empty — check timestamp alignment!")

    res_eeg, res_emg, res_imu = standardize_epoc_sizes(all_eeg, all_emg, all_imu)

    # Baseline correction (subtract pre-cue mean)
    baseline_samples = int(window_pre / (window_pre + window_post) * res_eeg.shape[1])
    if baseline_samples > 0:
        for j in range(len(res_eeg)):
            baseline = res_eeg[j, :baseline_samples, :].mean(axis=0, keepdims=True)
            res_eeg[j] -= baseline

    return res_eeg, res_emg, res_imu, all_labels


def slice_by_time(stream, t_start, t_end, t0=None):
    """
    Finds the data points between 2 timestamps.
    Handles both XDF dictionaries and MNE Raw objects.
    """
    if isinstance(stream, mne.io.BaseRaw):
        if t0 is None:
            raise ValueError("t0 (offset) is required for MNE Raw")
        
        t_start = t_start - t0
        t_end = t_end - t0

        # MNE crop is more efficient than masking
        t_start = max(t_start, stream.times[0])
        t_end = min(t_end, stream.times[-1])
        
        # .get_data().T returns (Samples, Channels) to match your workflow
        return stream.copy().crop(tmin=t_start, tmax=t_end).get_data().T
    else:
        ts = stream['time_stamps']
        series = stream['time_series']

        mask = (ts >= t_start) & (ts <= t_end)

        return series[mask]

def standardize_epoc_sizes(all_eeg, all_emg, all_imu, target_size=768):
    """
    Ensures every clip has exactly 'target_size' samples.
    3s * 256Hz = 768
    """
    resampled_eeg = []
    for j, x in enumerate(all_eeg):
        if x.shape[0] < 2:  # resample needs at least 2 points
            raise ValueError(f"EEG epoch {j} has only {x.shape[0]} samples")
        resampled_eeg.append(resample(x, target_size, axis=0))
    resampled_eeg = np.array(resampled_eeg)

    resampled_emg = []
    for j, x in enumerate(all_emg):
        if x.shape[0] < 2:
            raise ValueError(f"EMG epoch {j} has only {x.shape[0]} samples")
        resampled_emg.append(resample(x, target_size, axis=0))
    resampled_emg = np.array(resampled_emg)

    resampled_imu = []
    for epoch_dict in all_imu:
        imu_arrays = [epoch_dict[key] for key in sorted(epoch_dict.keys())]
        min_len = min(x.shape[0] for x in imu_arrays)
        imu_arrays = [x[:min_len] for x in imu_arrays]
        combined_imu = np.hstack(imu_arrays)
        if combined_imu.shape[0] < 2:
            raise ValueError(f"IMU epoch has only {combined_imu.shape[0]} samples")
        resampled_imu.append(resample(combined_imu, target_size, axis=0))

    return resampled_eeg, resampled_emg, np.array(resampled_imu)


#####   FILTERS   #####
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-5)
    high = min(highcut / nyq, 0.99)

    if low >= high:
        raise ValueError(f"Invalid band: low={lowcut}Hz, high={highcut}Hz, fs={fs}")

    b, a = butter(order, [low, high], btype='bandpass')
    return filtfilt(b, a, data, axis=0)


def low_pass(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff/nyq, btype='low')
    return filtfilt(b, a, data, axis=0)


def notch_filter(data, fs, freq=50):
    nyq = 0.5 * fs
    w0 = freq / nyq
    b, a = iirnotch(w0, Q=30)
    return filtfilt(b, a, data, axis=0)

def filter_imu(streams):
    """
    Complete preprocessing of multiple IMUs dictionary.
    Steps: Remove, Detrend, Lowpass
    """
    imu_clean = {}
    for key, stream in streams.items():
        raw_data = stream['time_series']
        fs = float(stream['info']['nominal_srate'][0])

        # 1. Remove channel 9 (Counter=useless)
        clean_data = raw_data[:, :9]
        
        # 2. Detrend (Remove gravity baseline)
        clean_data = detrend(clean_data, axis=0)

        # 3. Lowpass (15Hz). Human movement is slow --> Remove high freq. noise
        clean_data = low_pass(clean_data, 15, fs)

        # Save results keeping the original structure
        imu_clean[key] = {
            'time_series': clean_data,
            'time_stamps': stream['time_stamps'],
            'fs': fs
        }
    return imu_clean

def filter_eeg(stream):
    """
    Complete preprocessing of EEG stream.
    Steps: Remove, Notch, Bandpass
    """
    raw_data = stream['time_series']
    fs = float(stream['info']['nominal_srate'][0])
    if fs == 0: fs = 256    # Default if metadata fails

    # 1. Clean EEG channels
    clean_data = raw_data[:, :32]

    # 2. Notch (50Hz), Bandpass (1-45Hz)
    clean_data = notch_filter(clean_data, fs)
    clean_data = butter_bandpass_filter(clean_data, 1, 45, fs)
    
    filtered = {
        'time_series': clean_data,
        'time_stamps': stream['time_stamps'],
        'fs': fs
    }
    return filtered

def filter_emg(stream):
    """
    Complete preprocessing of EMG stream.
    Steps: Remove,
    """
    raw_data = stream['time_series']
    fs = float(stream['info']['nominal_srate'][0])
    if fs == 0: fs = 256

    # 1. Clean EMG channels
    clean_data = raw_data[:, :5]

    # 2. Bandpass (20-100Hz) and Notch (50Hz)
    clean_data = butter_bandpass_filter(clean_data, 20, 100, fs)
    clean_data = notch_filter(clean_data, fs)

    filtered = {
        'time_series': clean_data,
        'time_stamps': stream['time_stamps'],
        'fs': fs
    }
    return filtered
        

def band_power(eeg_epochs, fs=256):
    """
    Extract log band-power per channel
    Input: (Trials, Time, Channels)
    Output: (Trials, Bands * Channels)
    """
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    n_trials, n_time, n_ch = eeg_epochs.shape
    features = []

    for trial in range(n_trials):
        trial_features = []
        for ch in range(n_ch):
            f, psd = welch(eeg_epochs[trial, :, ch], fs=fs, nperseg=min(128, n_time))

            for (low, high) in bands.values():
                band_mask = (f >= low) & (f <= high)
                if band_mask.sum() > 0:
                    trial_features.append(np.log1p(np.trapezoid(psd[band_mask], f[band_mask])))
                else:
                    trial_features.append(0.0)        
        features.append(trial_features)
    return np.array(features)

#####   FEATURES   #####
def features_emg(stream):
    """
    Calculates the temporal characteristics for each EMG trial.
    Input: (Trials, Samples, Channels)
    Output: (Trials, Features)
    """
    raw_data = stream['time_series']
    fs = float(stream['info']['nominal_srate'][0])
    if fs == 0:
        fs = 256

    # 1. Select channels
    emg = raw_data[:, :5]

    # 2. Detrend 
    emg = detrend(emg, axis=0)

    # 3. Notch (powerline)
    emg = notch_filter(emg, fs)

    # 4. Bandpass 
    emg = butter_bandpass_filter(emg, 20, 150, fs)

    # 5. Rectification 
    emg = np.abs(emg)

    # 6. Envelope extraction (smooth muscle activation)
    emg = low_pass(emg, cutoff=5, fs=fs)

    return {
        'time_series': emg,
        'time_stamps': stream['time_stamps'],
        'fs': fs
    }

# def features_imu(imu_epochs):
#     """
#     Reduces 9 axis of each IMU to its total magnitude
#     imu_epochs shape: (Trials, Samples, 36) -> 4 sensors, 9 axis
#     Structure: [AccX, AccY, AccZ, GyroX, GyroY, GyroZ, MagX, MagY, MagZ] x 4
#     """
#     n_trials = imu_epochs.shape
#     smart_imu = []
#     for trial in imu_epochs:
#         magnitudes = []


def features_CSP(eeg_epochs, labels, n_filters=4):
    """
    Find n_filters that explain the most speech variance.
    """
    csp = mne.decoding.CSP(n_components=n_filters, reg=None, log=True)
    eeg_features = csp.fit_transform(eeg_epochs, labels)

    return eeg_features


def features_ICA(stream):      # Subject to human error, laborous in large datasets
    """
    Identifies "Blink", "Heartbeat" and other components.
    n_components: int (25 components); float (cumulative variance, 95%)
    """
    # 1. Convert to MNE Raw object
    ch_names = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F3', 'Fz', 'F4', 'F8',
                'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
                'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8',
                'POz', 'O1', 'Oz', 'O2']
    data = stream['time_series'][:, :32].T * 1e-6
    fs = float(stream['info']['nominal_srate'][0])
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(data, info)

    # SAVE GLOBAL OFFSET 
    t0 = stream['time_stamps'][0]

    # Notch and Pass-Band filter for optimal ICA
    raw.notch_filter(freqs=50.0)
    raw.filter(l_freq=1.0, h_freq=100.0, fir_design='firwin')
    
    # Montage Configuration (electrodes position)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')

    # Common Average Reference (CAR)
    raw.set_eeg_reference('average', projection=False)

    # 2. Train ICA
    ica = mne.preprocessing.ICA(
        n_components=25, 
        method='infomax',
        fit_params=dict(extended=True),
        random_state=42
        #, max_iter=800
    )
    ica.fit(raw)

    # 3. Classify with ICLabel
    ic_labels = label_components(raw, ica, method='iclabel')
    labels = ic_labels['labels']    # ['brain', 'muscle', 'eye', ...]
    print(labels)

    # 4. Select only 'Brain' components (Dimensionality Reduction)
    ica.exclude = [i for i, l in enumerate(labels) if l in ['eye blink', 'heart']]
    print(f"Excluding Components: {ica.exclude}")
    raw_clean = ica.apply(raw)
    raw_clean.filter(0.5, 40)

    return raw_clean, t0


def process_all(folder=RAW_DIR, target_task='silent'):
    all_data = {}
    pattern = re.compile(r"exp_sub_(\d+)_ses_(\d+)_bl_(.*)\.xdf")

    for filename in os.listdir(folder):
        match = pattern.search(filename)
        if not match:
            continue

        sub_id, ses_id, task = match.groups()

        if task != target_task:
            continue

        cached_path = os.path.join(PROCESSED_DIR, f"sub{sub_id}_ses_{ses_id}_{task}_preprocessed.npz")
        if os.path.exists(cached_path):
            print(f"[*] Loading Cache: Sub {sub_id} | Ses {ses_id}")
            cached = np.load(cached_path, allow_pickle=True)
            epochs = {
                'eeg': cached['eeg'],
                'emg': cached['emg'],
                'imu': cached['imu'],
                'labels': cached['labels']
            }
        
        else:
            print(f"[!] Processing Raw: Sub {sub_id} | Ses {ses_id}")
            raw_path = os.path.join(RAW_DIR, filename)
            raw_streams = load_session(raw_path)

            if raw_streams['eeg'] is None:
                print(f"Error: no EEG streams in {filename}")
                continue
            if raw_streams['emg'] is None:
                print(f"Error: no EEG streams in {filename}")
                continue

            # Filtering
            #eeg_ica, eeg_t0 = features_ICA(raw_streams['eeg'])
            eeg_clean = filter_eeg(raw_streams['eeg'])
            #emg_clean = filter_emg(raw_streams['emg'])
            emg_features = features_emg(raw_streams['emg'])
            imu_clean = filter_imu(raw_streams['imu'])

            # Epoching & Standardization
            eeg_ep, emg_ep, imu_ep, labels = get_epochs(eeg_clean, emg_features, imu_clean, raw_streams['markers']) #,offset=eeg_t0)   
            eeg_band = band_power(eeg_ep)
            epochs = {'ses': ses_id, 'eeg': eeg_ep, 'eeg_band': eeg_band, 'emg': emg_ep, 'imu': imu_ep, 'labels': labels}

            np.savez_compressed(cached_path, 
                eeg=eeg_ep.astype(np.float32), 
                emg=emg_ep.astype(np.float32), 
                imu=imu_ep.astype(np.float32), 
                labels=labels
            )
            if os.path.exists(cached_path):
                print(f"[*] Loading Cache: Sub {sub_id} | Ses {ses_id}")
                cached = np.load(cached_path, allow_pickle=True)
                epochs = {
                    'eeg': cached['eeg'],
                    'eeg_band': cached['eeg_band'] if 'eeg_band' in cached else None,
                    'emg': cached['emg'],
                    'imu': cached['imu'],
                    'labels': cached['labels']
                }
                # Recompute band-power if cache is old and doesn't have it
                if epochs['eeg_band'] is None:
                    epochs['eeg_band'] = band_power(epochs['eeg'])
        
        if sub_id not in all_data:
            all_data[sub_id] = []

        all_data[sub_id].append(epochs)

    return all_data
