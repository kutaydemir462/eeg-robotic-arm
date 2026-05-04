# 🦾 EEG-Controlled Robotic Arm

**Senior Design Thesis — MEF University, Computer Engineering (2025–2026)**

An end-to-end system for controlling a robotic arm using EEG brain signals. The project integrates EEG signal processing, robotic control, and reinforcement learning to enable brain-driven interaction with a physical robotic system.

> 🚧 **Status:** EEG data collected and preprocessed. Manual control working. RL training in progress.

---

## 📂 Project Files (Google Drive)

Full project files including test videos, EEG data, and training stimuli:

👉 **[EEG Controlled Robotic Arm — Google Drive](https://drive.google.com/drive/folders/1q6Skyh8Njz4_8ewI-p6whcAgoy9CtzqY?usp=sharing)**

| Folder | Contents |
|---|---|
| `Training Stimuli/` | Videos used to elicit motor imagery |
| `test videos time-lapse/` | Accelerated robotic arm tests |
| `test videos raw/` | Raw experimental recordings |
| `EEG Data/` | BrainVision recordings (`.eeg`, `.vhdr`, `.vmrk`) |

---

## 📌 Project Overview

| Component | Description |
|---|---|
| **Hardware** | Robotic arm + Raspberry Pi + PCA9685 |
| **EEG Data** | Motor imagery signals (C3/C4 channels) |
| **Signal Processing** | Filtering, epoching, labeling, feature extraction |
| **Control Modes** | Keyboard, REST API, Motion controller |
| **RL Algorithm** | SAC + HER in PyBullet |
| **Goal** | EEG signals → decoded intention → robotic movement |

---

## 🧠 EEG Signal Processing Pipeline

This project includes a full EEG preprocessing and labeling pipeline implemented in Python.

### Files

- `eeg_preprocessing.py` → preprocessing pipeline
- `marking.py` → event labeling and segmentation
- `.fif files` → cleaned and processed EEG data
- `.vhdr / .vmrk / .eeg` → raw BrainVision recordings

### Processing Steps

- Band-pass filtering (motor imagery bands: mu, beta)
- Noise removal (artifact handling)
- Epoch segmentation
- Event marking and labeling
- Feature preparation for ML/RL pipeline

### Output

- Clean EEG epochs (`.fif`)
- Labeled datasets for training

---

## 🧠 Reinforcement Learning

The system uses **Soft Actor-Critic (SAC)** with **Hindsight Experience Replay (HER)** for robotic arm control.

### Key Features

- Custom Gymnasium environment (`RobotArmPickPlace-v0`)
- 19D observation / 6D action space
- Dense reward shaping
- Curriculum learning
- Parallel training (12 environments)

---

## 🎮 Control Modes

### 1. Keyboard Control
Manual control via keyboard inputs.

### 2. REST API
FastAPI-based control system for sending movement commands.

```bash
uvicorn remote_control_with_api:app --reload
