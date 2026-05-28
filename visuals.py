import mne
import numpy as np
import matplotlib.pyplot as plt

def subplots_streams(stream):
    data_points = stream['time_series']
    timestamps = stream['time_stamps']

    n_channels = int(data_points.shape[1])
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 1.5*n_channels), sharex=True)
    
    for i in range(n_channels):
        axes[i].plot(timestamps, data_points[:, i])
        axes[i].set_ylabel(f'Ch {i}')
        axes[i].grid(True)
    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.show()


def plot_stream(stream):
    data_points = stream['time_series']
    timestamps = stream['time_stamps']

    plt.figure(figsize=(12,4))
    plt.plot(timestamps, data_points)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()


def plot_eeg(eeg_stream):
    data = eeg_stream['time_series']
    fs = eeg_stream['fs']

    # Transpose and multiply by microVolts for MNE
    data_t = data.T * 1e-6
    ch_names = [f'EEG_{i:02d}' for i in range(data_t.shape[0])]

    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(data_t, info)
    raw.plot(scalings='auto', block=True, title='EEG')


def plot_epochs(epochs_array, channel_idx=0, fs=256, window_pre=1.0):
    """
    Grafica la consistencia de un canal a través de todos los trials.
    """
    # Extraer todas las épocas para el canal elegido
    # data shape: (Trials, Tiempo)
    data = epochs_array[:, :, channel_idx]
    n_trials, n_samples = data.shape
    time = np.linspace(0, n_samples / fs, n_samples)-window_pre

    plt.figure(figsize=(10, 6))
    
    # Graficamos cada trial con una transparencia alta (alpha)
    for i in range(n_trials):
        plt.plot(time, data[i, :], color='gray', alpha=0.2, lw=0.5)
    
    plt.plot(time, np.mean(data, axis=0), color='red', lw=2, label='Mean')

    plt.axvline(0, color='black', linestyle='--', label='Start (0s)')
    
    plt.title(f"Channel Consistency {channel_idx} ({n_trials} trials)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# sanity_check.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import preprocessing

def plot_erp_by_class(all_data, channel_idx=16, channel_name="Cz"):
    """
    Plot grand-average ERP per class.
    If there's NO visible difference between classes → signal is absent.
    channel_idx=16 corresponds to Cz in your montage.
    """
    class_epochs = {i: [] for i in range(10)}

    for sub_id, sessions in all_data.items():
        for ses in sessions:
            eeg = ses['eeg']       # (trials, time, channels)
            labels = ses['labels']
            for trial_idx, label in enumerate(labels):
                class_epochs[label].append(eeg[trial_idx, :, channel_idx])

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
    for cls_id, ax in enumerate(axes.flat):
        trials = np.array(class_epochs[cls_id])
        mean = trials.mean(axis=0)
        sem = trials.std(axis=0) / np.sqrt(len(trials))

        t = np.linspace(-0.5, 2.5, len(mean))  
        ax.plot(t, mean, 'b-', linewidth=1.5)
        ax.fill_between(t, mean - sem, mean + sem, alpha=0.3)
        ax.axvline(0, color='r', linestyle='--', label='Cue')
        ax.set_title(f"Class {cls_id}")
        ax.set_xlabel("Time (s)")

    plt.suptitle(f"Grand-Average ERP at {channel_name}")
    plt.tight_layout()
    plt.savefig("erp_sanity_check.png", dpi=150)
    plt.show()


def plot_psd_by_class(all_data, channel_idx=16, fs=85):
    """
    Plot PSD per class. Look for band-power differences.
    """
    class_epochs = {i: [] for i in range(10)}

    for sub_id, sessions in all_data.items():
        for ses in sessions:
            eeg = ses['eeg']
            labels = ses['labels']
            for trial_idx, label in enumerate(labels):
                class_epochs[label].append(eeg[trial_idx, :, channel_idx])

    plt.figure(figsize=(12, 6))
    for cls_id in range(10):
        trials = np.array(class_epochs[cls_id])
        psds = []
        for trial in trials:
            f, psd = welch(trial, fs=fs, nperseg=min(128, len(trial)))
            psds.append(psd)
        mean_psd = np.mean(psds, axis=0)
        plt.semilogy(f, mean_psd, label=f"Class {cls_id}")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.legend()
    plt.title("Power Spectral Density by Class")
    plt.savefig("psd_sanity_check.png", dpi=150)
    plt.show()


def check_imu_confound(all_data):
    """
    Check if IMU has class-discriminative MEAN values.
    If yes → confound (movement during cue, not speech).
    """
    class_means = {i: [] for i in range(10)}

    for sub_id, sessions in all_data.items():
        for ses in sessions:
            imu = ses['imu']       # (trials, time, 36)
            labels = ses['labels']
            for trial_idx, label in enumerate(labels):
                # Mean IMU across time per trial
                class_means[label].append(imu[trial_idx].mean(axis=0))

    print("\n=== IMU CLASS MEANS (first 6 channels) ===")
    for cls_id in range(10):
        arr = np.array(class_means[cls_id])
        print(f"Class {cls_id}: {arr.mean(axis=0)[:6].round(4)}")

    #print("\nIf means differ systematically → movement confound, not speech signal!")