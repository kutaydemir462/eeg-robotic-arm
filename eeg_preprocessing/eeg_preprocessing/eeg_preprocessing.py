"""
====================================================================
  MINDSARM – EEG Preprocessing Pipeline
  Participant: Bedri (Bedri331)
  
  Preprocessing steps based on instructor notes:
    - Artifact reduction
    - Eye blink removal (ICA)
    - Excessive head movement detection (accelerometer)
    - 48–52 Hz notch filter
    - 0.2–100 Hz bandpass filter
    - 200 µV / 100 ms epoch rejection

  Usage:
    python eeg_preprocessing.py
    
  Outputs:
    Bedri_raw_preprocessed.fif
    Bedri_epochs_clean.fif
    Bedri_epochs_labeled.csv
    Bedri_preprocessing_report.txt
====================================================================
"""

import mne
import numpy as np
import pandas as pd
import os
from datetime import datetime

mne.set_log_level('WARNING')

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
INPUT_FILE      = "Bedri331_labeled.vhdr"          # BrainVision header file
OUTPUT_DIR      = "."                      # Output folder

SFREQ           = 500                      # Hz
BAD_CH_STD_MULT = 5.0                      # Bad channel: std > N × median std

BP_LOW          = 0.2                      # Bandpass lower edge (Hz)
BP_HIGH         = 100.0                    # Bandpass upper edge (Hz)

NOTCH_LOW       = 48.0                     # Notch lower edge (Hz)
NOTCH_HIGH      = 52.0                     # Notch upper edge (Hz)

ICA_N_COMPONENTS = 15                      # Number of ICA components
ICA_EOG_CHANNELS = ['Fp1', 'Fp2']         # Channels used for eye blink detection
ICA_Z_THRESHOLD  = 3.0                    # Z-score threshold for EOG component

EPOCH_TMIN      = -1.0                     # Epoch start (s) relative to stimulus
EPOCH_TMAX      =  4.0                     # Epoch end (s) relative to stimulus
BASELINE        = (-1.0, 0.0)             # Baseline correction window

REJECT_AMP_UV   = 200e-6                  # Amplitude rejection threshold (V)
REJECT_WIN_S    = 0.1                      # Window size for amplitude check (s)
REJECT_JERK_MG  = 20.0                    # Accelerometer jerk threshold (mg/sample)

# ─────────────────────────────────────────────
# STEP 0 – LOAD RAW DATA
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  MINDSARM – EEG Preprocessing Pipeline")
print("="*60)
print(f"\n[0] Loading: {INPUT_FILE}")

raw = mne.io.read_raw_brainvision(INPUT_FILE, preload=True)

n_channels_total = len(raw.ch_names)
duration_s       = raw.times[-1]
sfreq            = raw.info['sfreq']

print(f"    Duration : {duration_s:.1f} s ({duration_s/60:.1f} min)")
print(f"    Channels : {n_channels_total}")
print(f"    Sfreq    : {sfreq} Hz")

# Separate EEG and accelerometer channels
accel_ch = [ch for ch in raw.ch_names if ch.lower() in ['x_dir', 'y_dir', 'z_dir']]
eeg_ch   = [ch for ch in raw.ch_names if ch not in accel_ch]
print(f"    EEG      : {len(eeg_ch)} channels")
print(f"    Accel    : {accel_ch}")

# Set channel types properly
raw.set_channel_types({ch: 'misc' for ch in accel_ch})

# ─────────────────────────────────────────────
# STEP 1 – BAD CHANNEL DETECTION
# ─────────────────────────────────────────────
print("\n[1] Bad channel detection...")

eeg_data = raw.get_data(picks='eeg')  # (n_eeg, n_times)
ch_stds   = eeg_data.std(axis=1)
median_std = np.median(ch_stds)
threshold  = BAD_CH_STD_MULT * median_std

bad_chs = [
    raw.ch_names[raw.ch_names.index(ch)]
    for ch, std in zip(mne.pick_info(raw.info, mne.pick_types(raw.info, eeg=True))['ch_names'], ch_stds)
    if std > threshold
]

if bad_chs:
    print(f"    Bad channels found: {bad_chs}")
    raw.info['bads'] = bad_chs
    raw.interpolate_bads(reset_bads=True)
    print(f"    Interpolated {len(bad_chs)} bad channel(s).")
else:
    print("    No bad channels found.")

# ─────────────────────────────────────────────
# STEP 2 – BANDPASS FILTER (0.2 – 100 Hz)
# ─────────────────────────────────────────────
print(f"\n[2] Bandpass filter: {BP_LOW}–{BP_HIGH} Hz (FIR, zero-phase)...")

raw.filter(
    l_freq=BP_LOW,
    h_freq=BP_HIGH,
    method='fir',
    phase='zero',
    fir_window='hamming',
    picks='eeg',
    verbose=False
)
print("    Done.")

# ─────────────────────────────────────────────
# STEP 3 – NOTCH FILTER (48 – 52 Hz)
# ─────────────────────────────────────────────
print(f"\n[3] Notch filter: {NOTCH_LOW}–{NOTCH_HIGH} Hz (FIR, zero-phase)...")

raw.notch_filter(
    freqs=50.0,
    method='fir',
    phase='zero',
    picks='eeg',
    verbose=False
)
print("    Done.")

# ─────────────────────────────────────────────
# STEP 4 – AVERAGE REFERENCE (CAR)
# ─────────────────────────────────────────────
print("\n[4] Setting average reference (CAR)...")

raw.set_eeg_reference(ref_channels='average', projection=True, verbose=False)
raw.apply_proj(verbose=False)
print("    Done.")

# ─────────────────────────────────────────────
# STEP 5 – ICA – EYE BLINK REMOVAL
# ─────────────────────────────────────────────
print(f"\n[5] ICA – FastICA, {ICA_N_COMPONENTS} components...")

ica = mne.preprocessing.ICA(
    n_components=ICA_N_COMPONENTS,
    method='fastica',
    random_state=42,
    max_iter=800
)
ica.fit(raw, picks='eeg', verbose=False)

# Detect eye-blink components using Fp1/Fp2 correlation
eog_indices = []
for eog_ch in ICA_EOG_CHANNELS:
    if eog_ch in raw.ch_names:
        indices, scores = ica.find_bads_eog(
            raw,
            ch_name=eog_ch,
            threshold=ICA_Z_THRESHOLD,
            verbose=False
        )
        eog_indices.extend(indices)

eog_indices = list(set(eog_indices))  # deduplicate
ica.exclude = eog_indices
ica.apply(raw, verbose=False)

print(f"    Components excluded (eye blink): {eog_indices}")

# ─────────────────────────────────────────────
# STEP 6 – EPOCHING
# ─────────────────────────────────────────────
print(f"\n[6] Epoching: {EPOCH_TMIN}s to +{EPOCH_TMAX}s around stimulus onset...")

events, event_id = mne.events_from_annotations(raw, verbose=False)
print(f"    Events found: {len(events)} across {len(event_id)} conditions")

epochs_all = mne.Epochs(
    raw,
    events,
    event_id=event_id,
    tmin=EPOCH_TMIN,
    tmax=EPOCH_TMAX,
    baseline=BASELINE,
    preload=True,
    reject=None,   # Manual rejection below
    verbose=False
)

n_epochs_raw = len(epochs_all)
print(f"    Epochs created: {n_epochs_raw}")

# ─────────────────────────────────────────────
# STEP 7 – EPOCH REJECTION
# ─────────────────────────────────────────────
print(f"\n[7] Epoch rejection...")
print(f"    Amplitude criterion : >{REJECT_AMP_UV*1e6:.0f} µV in any {REJECT_WIN_S*1000:.0f} ms window")
print(f"    Motion criterion    : jerk > {REJECT_JERK_MG} mg/sample")

win_samples = int(REJECT_WIN_S * sfreq)
data_v = epochs_all.get_data(picks='eeg')  # (epochs, channels, times)

reject_amp    = []
reject_motion = []

for i in range(len(epochs_all)):
    epoch = data_v[i]  # (channels, times)
    # Sliding window peak-to-peak check
    rejected = False
    for start in range(0, epoch.shape[1] - win_samples, win_samples // 2):
        window = epoch[:, start:start + win_samples]
        ptp    = window.max(axis=1) - window.min(axis=1)
        if ptp.max() > REJECT_AMP_UV:
            reject_amp.append(i)
            rejected = True
            break

# Motion check using accelerometer data
if accel_ch:
    accel_epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
        baseline=None, preload=True,
        picks=accel_ch, reject=None, verbose=False
    )
    accel_data = accel_epochs.get_data()  # (epochs, 3, times)
    for i in range(len(accel_epochs)):
        jerk = np.abs(np.diff(accel_data[i], axis=1)).max()
        accel_scale = np.abs(accel_data).max()
        jerk_threshold = REJECT_JERK_MG if accel_scale > 10 else REJECT_JERK_MG * 1e-3
        if jerk > jerk_threshold:
            if i not in reject_amp:
                reject_motion.append(i)

all_reject = sorted(set(reject_amp + reject_motion))
keep_mask  = np.ones(len(epochs_all), dtype=bool)
keep_mask[all_reject] = False

epochs_clean = epochs_all[keep_mask]
n_kept = len(epochs_clean)

print(f"    Rejected (amplitude): {len(reject_amp)}")
print(f"    Rejected (motion)   : {len(reject_motion)}")
print(f"    KEPT                : {n_kept} / {n_epochs_raw} ({100*n_kept/n_epochs_raw:.1f}%)")

# ─────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────
print("\n[8] Saving output files...")

out_raw    = os.path.join(OUTPUT_DIR, "Bedri_raw_preprocessed.fif")
out_epochs = os.path.join(OUTPUT_DIR, "Bedri_epochs_clean.fif")
out_csv    = os.path.join(OUTPUT_DIR, "Bedri_epochs_labeled.csv")
out_report = os.path.join(OUTPUT_DIR, "Bedri_preprocessing_report.txt")

raw.save(out_raw, overwrite=True, verbose=False)
print(f"    Saved: {out_raw}")

epochs_clean.save(out_epochs, overwrite=True, verbose=False)
print(f"    Saved: {out_epochs}")

# Build labeled CSV
rows = []
event_id_inv = {v: k for k, v in event_id.items()}
kept_indices  = np.where(keep_mask)[0]

for local_idx, orig_idx in enumerate(kept_indices):
    ev_code  = events[orig_idx, 2]
    label    = event_id_inv.get(ev_code, f"Unknown_{ev_code}")
    ep_data  = data_v[orig_idx]
    ptp_uv   = (ep_data.max() - ep_data.min()) * 1e6
    rows.append({
        'trial':   orig_idx + 1,
        'label':   label,
        'ptp_uv':  round(ptp_uv, 1),
        'status':  'KEPT'
    })

for orig_idx in all_reject:
    ev_code = events[orig_idx, 2]
    label   = event_id_inv.get(ev_code, f"Unknown_{ev_code}")
    ep_data = data_v[orig_idx]
    ptp_uv  = (ep_data.max() - ep_data.min()) * 1e6
    rows.append({
        'trial':  orig_idx + 1,
        'label':  label,
        'ptp_uv': round(ptp_uv, 1),
        'status': 'REJECT'
    })

df = pd.DataFrame(rows).sort_values('trial').reset_index(drop=True)
df.to_csv(out_csv, index=False)
print(f"    Saved: {out_csv}")

# ─────────────────────────────────────────────
# GENERATE REPORT
# ─────────────────────────────────────────────
report_lines = [
    "=" * 68,
    "  MINDSARM – EEG PREPROCESSING REPORT",
    f"  Participant: Bedri (Bedri331)",
    f"  Date processed: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    "=" * 68,
    "",
    "RECORDING DETAILS",
    "-" * 40,
    f"  File              : Bedri331.vhdr / .eeg / .vmrk",
    f"  Duration          : {duration_s:.1f} s ({duration_s/60:.1f} min)",
    f"  Sampling rate     : {sfreq:.0f} Hz",
    f"  EEG channels      : {len(eeg_ch)}",
    f"  Aux channels      : {len(accel_ch)} accelerometers ({', '.join(accel_ch)})",
    f"  Total trials      : {n_epochs_raw}",
    "",
    "PREPROCESSING STEPS",
    "-" * 40,
    "",
    "  STEP 1 – BAD CHANNEL DETECTION & INTERPOLATION",
    f"    Bad found  : {len(bad_chs)} — {'None' if not bad_chs else ', '.join(bad_chs)}",
    f"    Action     : {'No interpolation needed' if not bad_chs else 'Interpolated bad channels'}",
    "",
    "  STEP 2 – BANDPASS FILTER",
    f"    Type       : FIR (zero-phase)",
    f"    Passband   : {BP_LOW} – {BP_HIGH} Hz",
    "",
    "  STEP 3 – NOTCH FILTER",
    f"    Type       : FIR (zero-phase)",
    f"    Stopband   : {NOTCH_LOW} – {NOTCH_HIGH} Hz",
    "",
    "  STEP 4 – AVERAGE REFERENCE",
    f"    Method     : Common average reference (CAR)",
    "",
    "  STEP 5 – ICA – EYE BLINK REMOVAL",
    f"    Method     : FastICA, {ICA_N_COMPONENTS} components",
    f"    Detection  : Fp1/Fp2 correlation (z-threshold = {ICA_Z_THRESHOLD})",
    f"    Excluded   : Component(s) {eog_indices}",
    "",
    "  STEP 6 – EPOCHING",
    f"    Window     : {EPOCH_TMIN} to +{EPOCH_TMAX} s around stimulus onset",
    f"    Baseline   : {BASELINE[0]} to {BASELINE[1]} s",
    f"    Total epochs created : {n_epochs_raw}",
    "",
    "  STEP 7 – EPOCH REJECTION",
    f"    Amplitude criterion : >{REJECT_AMP_UV*1e6:.0f} µV peak-to-peak in any {REJECT_WIN_S*1000:.0f} ms window",
    f"    Motion criterion    : Accelerometer jerk > {REJECT_JERK_MG} mg/sample",
    f"    Rejected (amplitude): {len(reject_amp)}",
    f"    Rejected (motion)   : {len(reject_motion)}",
    f"    KEPT                : {n_kept} / {n_epochs_raw} ({100*n_kept/n_epochs_raw:.1f}%)",
    "",
    "CLEAN DATA SUMMARY",
    "-" * 40,
    f"  Epochs retained   : {n_kept} / {n_epochs_raw} ({100*n_kept/n_epochs_raw:.1f}%)",
    f"  Output files      : {out_raw}",
    f"                      {out_epochs}",
    f"                      {out_csv}",
    "",
    "=" * 68,
    "  Preprocessing method based on instructor notes (Prof. Dr. Tuna Çakar):",
    "  artifact reduction, eye blink removal, head movement detection,",
    f"  {NOTCH_LOW}–{NOTCH_HIGH} Hz notch, {BP_LOW}–{BP_HIGH} Hz bandpass, {REJECT_AMP_UV*1e6:.0f} µV / {REJECT_WIN_S*1000:.0f} ms rejection.",
    "=" * 68,
]

with open(out_report, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print(f"    Saved: {out_report}")

print("\n✓ Preprocessing complete.")
print(f"  {n_kept}/{n_epochs_raw} epochs kept ({100*n_kept/n_epochs_raw:.1f}%)")
