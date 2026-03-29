# 🦾 EEG-Controlled Robotic Arm

**Senior Design Thesis — MEF University, Computer Engineering (2025–2026)**

An end-to-end system for controlling a commercial robotic arm using EEG brain signals, built on a Raspberry Pi with a PCA9685 servo driver. The project combines inverse kinematics, a REST API control layer, motion-sensor input, and a reinforcement learning pipeline for autonomous pick-and-place tasks.

> 🚧 **Status:** EEG data collected. Keyboard & controller control working. RL training in progress.

---

## 📂 Project Files (Google Drive)

> Full project files including test videos, EEG data, and training stimuli:
> **[🔗 EEG Controlled Robotic Arm — Google Drive](https://drive.google.com/drive/folders/1q6Skyh8Njz4_8ewI-p6whcAgoy9CtzqY?usp=sharing)**

| Folder | Contents |
|---|---|
| `Training Stimuli/` | Videos shown to subjects to elicit motor imagery |
| `test videos time-lapse/` | Speed-up recordings of arm movement tests |
| `test videos raw/` | Raw footage of arm movement sessions |
| `EEG Data/` | BrainVision format recordings (`.eeg`, `.vhdr`, `.vmrk`) from 2 subjects |

---

## 📌 Project Overview

| Component | Description |
|---|---|
| **Hardware** | Commercial robotic arm + Raspberry Pi + PCA9685 |
| **EEG Data** | Motor imagery data collected from 2 subjects (C3/C4 channels, BrainVision format) |
| **Control Modes** | Keyboard, REST API, Gyroscope/motion controller |
| **RL Algorithm** | SAC (Soft Actor-Critic) + HER, trained in PyBullet simulation |
| **Goal** | Map EEG brain signals → arm movement via trained RL policy |

---

## 📁 Repository Structure

```
eeg-robotic-arm/
│
├── keyboard_control_codes/        # Keyboard-based arm control
│
├── ML_Codes/                      # Reinforcement learning pipeline
│   ├── config.py                  # Central config: joints, rewards, SAC hyperparams
│   ├── train.py                   # SAC + HER training with curriculum learning
│   ├── robot_arm_env.py           # Custom Gymnasium environment
│   ├── remote_control_with_api.py         # FastAPI server with async command queue
│   ├── remote_control_with_controller.py  # Gyroscope/motion controller input
│   └── robot_state.json           # Persisted robot position state
│
└── README.md
```

---

## 🧠 Reinforcement Learning

The RL pipeline trains a **SAC (Soft Actor-Critic)** agent with **Hindsight Experience Replay (HER)** to perform pick-and-place tasks in PyBullet simulation.

### Key Design Choices

- **Custom Gymnasium environment** (`RobotArmPickPlace-v0`) with 19-dim observation space and 6-dim action space
- **Dense reward shaping** with contact-based bonuses (single/dual finger contact, grasp, lift, success)
- **Curriculum learning**: grasp threshold and lift height tighten progressively over 2M timesteps
- **Parallel training** across 12 environments using `SubprocVecEnv`
- **Automatic video recording** of training episodes saved as `.mp4`

### Reward Structure (v4)

```python
"distance_to_bottle": -2.0    # approach reward
"distance_to_goal":   -3.0    # goal-directed reward
"single_contact_bonus": 2.0   # finger touches bottle
"dual_contact_bonus":   5.0   # both fingers contact
"grasp_bonus":         10.0   # successful grasp
"lift_bonus":          15.0   # bottle lifted
"success_bonus":      100.0   # bottle placed at goal
```

### Training

```bash
cd ML_Codes
python train.py                    # fresh training (2M timesteps)
python train.py --resume           # continue from checkpoint
python train.py --timesteps 500000 # custom timestep count
```

---

## 🎮 Control Modes

### 1. Keyboard Control
Direct keyboard input for manual arm control. See `keyboard_control_codes/`.

### 2. REST API (`remote_control_with_api.py`)
Async FastAPI server with a FIFO task queue. Sends commands to the physical arm over HTTP.

```bash
uvicorn remote_control_with_api:app --reload
```

**Endpoints:**
- `POST /commands` — enqueue a movement command (`xup`, `xdown`, `yup`, `ydown`, `zup`, `zdown`, `gripopen`, `gripclose`)
- `GET /jobs/{job_id}` — check job status
- `GET /queue` — view queue state

### 3. Motion Controller (`remote_control_with_controller.py`)
Controls the arm using a gyroscope/IMU device. Pitch and roll are mapped to X/Y axis movement; buttons control Z axis and gripper.

```bash
python remote_control_with_controller.py
```

State (position + gripper angle) is automatically saved to `robot_state.json` on exit and restored on next launch.

---

## ⚙️ Hardware Setup

| Component | Details |
|---|---|
| Robotic Arm | Commercial 6-DOF arm |
| Controller | Raspberry Pi |
| Servo Driver | PCA9685 (I2C) |
| EEG Channels | C3, C4 (motor cortex) |
| Input Devices | Keyboard / Gyroscope controller / REST API |

---

## 📦 Dependencies

```bash
pip install stable-baselines3 gymnasium pybullet fastapi uvicorn numpy evdev
```

---

## 🗺️ Roadmap

- [x] Hardware integration (Raspberry Pi + PCA9685)
- [x] Keyboard control
- [x] REST API control layer
- [x] Gyroscope/motion controller input
- [x] EEG data collection (C3/C4, human subjects)
- [x] Custom RL environment + SAC training pipeline
- [ ] EEG signal preprocessing pipeline
- [ ] EEG → action mapping with trained RL policy
- [ ] Real-world sim-to-real transfer

---

## 👥 Team

Developed as part of the MEF University Computer Engineering Senior Design Program (2025–2026).

---

