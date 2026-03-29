"""
ROBOTKOL - Merkezi Konfigürasyon Dosyası (v4 - Contact-Based Rewards)
"""
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(PROJECT_DIR, "robot.urdf")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")
VIDEOS_DIR = os.path.join(PROJECT_DIR, "videos")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model")
CHECKPOINT_DIR = os.path.join(MODELS_DIR, "checkpoints")

for d in [MODELS_DIR, LOGS_DIR, VIDEOS_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

SIM_TIMESTEP = 1.0 / 240.0
SIM_STEPS_PER_ACTION = 10
GRAVITY = -9.81

ARM_JOINT_INDICES = [0, 1, 2, 3]
GRIPPER_JOINT_INDICES = [6, 7]
ALL_CONTROLLABLE_JOINTS = ARM_JOINT_INDICES + GRIPPER_JOINT_INDICES
NUM_JOINTS = len(ALL_CONTROLLABLE_JOINTS)

ARM_JOINT_LIMITS = {
    0: {"lower": -1.5708, "upper": 1.5708, "effort": 30, "velocity": 0.8},
    1: {"lower": 0.0,     "upper": 1.587,  "effort": 30, "velocity": 0.8},
    2: {"lower": 0.0,     "upper": 2.150,  "effort": 30, "velocity": 0.8},
    3: {"lower": -0.780,  "upper": 0.669,  "effort": 20, "velocity": 0.8},
}
GRIPPER_JOINT_LIMITS = {
    6: {"lower": 0.140, "upper": 0.285, "effort": 20, "velocity": 0.8},
    7: {"lower": -0.285, "upper": -0.140, "effort": 20, "velocity": 0.8},
}

END_EFFECTOR_LINK_INDEX = 5

TABLE_HEIGHT = 0.0
TABLE_SIZE = [0.8, 0.8, 0.02]
TABLE_POSITION = [0.0, -0.35, 0.0]

BOTTLE_RADIUS = 0.02
BOTTLE_HEIGHT = 0.10
BOTTLE_MASS = 0.3

BOTTLE_SPAWN_RANGE = {
    "x": [0.10, 0.20],
    "y": [-0.58, -0.35],
    "z": [0.03],
}
FIXED_GOAL_OFFSET_X = -0.30

SUCCESS_THRESHOLD = 0.04
GRASP_THRESHOLD = 0.08
LIFT_HEIGHT = 0.06
LIFT_REQUIRED = True

OBS_DIM = 19
GOAL_DIM = 3
ACTION_DIM = 6

# ============================================================
# REWARD (v4 — fiziksel temas, tek seferlik bonuslar)
# ============================================================
REWARD_TYPE = "dense"
REWARD_WEIGHTS = {
    "distance_to_bottle": -2.0,
    "distance_to_goal": -3.0,         
    "height_bonus": 2.0,              
    
    "single_contact_bonus": 2.0,      
    "dual_contact_bonus": 5.0,        
    "grasp_bonus": 10.0,              
    "lift_bonus": 15.0,               
    "success_bonus": 100.0,           
    
    "gripper_open_approach": 0.3,     
    "gripper_close_contact": 0.5,     
    
    "action_penalty": -0.01,
    "push_penalty": -2.0,    
    "knock_penalty": -2.0,   
}

# ============================================================
# SAC
# ============================================================
SAC_PARAMS = {
    "learning_rate": 3e-4,
    "buffer_size": 300_000,      
    "batch_size": 512,           
    "gamma": 0.98,
    "tau": 0.02,
    "learning_starts": 10_000,
    "train_freq": 1,
    "gradient_steps": 8,
    "ent_coef": "auto",
    "target_entropy": "auto",
    "policy_kwargs": dict(net_arch=[512, 512, 512]), 
}

HER_PARAMS = {
    "n_sampled_goal": 4,
    "goal_selection_strategy": "future",
}

TOTAL_TIMESTEPS = 2_000_000
MAX_EPISODE_STEPS = 100
EVAL_FREQ = 6250
EVAL_EPISODES = 50
CHECKPOINT_FREQ = 100_000
N_ENVS = 12
SEED = 42

EVAL_NUM_EPISODES = 50
VIDEO_NUM_EPISODES = 10
VIDEO_FPS = 30
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720

CAMERA_DISTANCE = 0.9
CAMERA_YAW = 0
CAMERA_PITCH = -35
CAMERA_TARGET = [0.0, -0.35, 0.10]

ACTION_MAX_DELTA = 0.05
GRIPPER_MAX_DELTA = 0.02