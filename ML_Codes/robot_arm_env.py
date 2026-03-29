"""
ROBOTKOL - PyBullet Robot Kol Environment (v4 - Contact-Based Rewards)
======================================================================
Fiziksel temas algılama ile hack-proof kavrama reward sistemi.
p.getContactPoints() kullanarak gerçek parmak-şişe temasını tespit eder.
"""

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
import config as cfg


class RobotArmPickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": cfg.VIDEO_FPS}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.step_counter = 0
        self.robot_id = None
        self.bottle_id = None
        self.table_id = None
        self.goal_visual_id = None
        self.goal_position = np.zeros(3)

        # Durum takibi
        self._bottle_initial_z = 0.0
        self._bottle_was_lifted = False
        self._prev_bottle_pos = np.zeros(3)

        # Tek seferlik reward flag'leri
        self._flag_single_contact = False
        self._flag_dual_contact = False
        self._flag_grasp = False
        self._flag_lift = False

        # PyBullet bağlantısı
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        else:
            self.physics_client = p.connect(p.DIRECT)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(cfg.ACTION_DIM,), dtype=np.float32
        )
        obs_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(cfg.OBS_DIM,), dtype=np.float32
        )
        goal_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(cfg.GOAL_DIM,), dtype=np.float32
        )
        self.observation_space = spaces.Dict({
            "observation": obs_space,
            "achieved_goal": goal_space,
            "desired_goal": goal_space,
        })

    # ==========================================================
    # RESET
    # ==========================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_counter = 0
        self._bottle_was_lifted = False

        # Flag'leri sıfırla
        self._flag_single_contact = False
        self._flag_dual_contact = False
        self._flag_grasp = False
        self._flag_lift = False

        p.resetSimulation(physicsClientId=self.physics_client)
        p.setGravity(0, 0, cfg.GRAVITY, physicsClientId=self.physics_client)
        p.setTimeStep(cfg.SIM_TIMESTEP, physicsClientId=self.physics_client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        self.table_id = self._create_table()

        self.robot_id = p.loadURDF(
            cfg.URDF_PATH,
            basePosition=[0, 0, cfg.TABLE_HEIGHT + cfg.TABLE_SIZE[2]],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
            physicsClientId=self.physics_client
        )

        for joint_idx in cfg.ARM_JOINT_INDICES + cfg.GRIPPER_JOINT_INDICES:
            p.resetJointState(self.robot_id, joint_idx, 0.0,
                              physicsClientId=self.physics_client)

        bottle_pos = self._random_bottle_position()
        self.bottle_id = self._create_bottle(bottle_pos)
        self._bottle_initial_z = bottle_pos[2]
        self._prev_bottle_pos = bottle_pos.copy()

        self.goal_position = self._random_goal_position(bottle_pos)
        self.goal_visual_id = self._create_goal_visual(self.goal_position)

        for _ in range(50):
            p.stepSimulation(physicsClientId=self.physics_client)

        bottle_pos_after, _ = p.getBasePositionAndOrientation(
            self.bottle_id, physicsClientId=self.physics_client
        )
        self._prev_bottle_pos = np.array(bottle_pos_after)
        self._bottle_initial_z = bottle_pos_after[2]

        return self._get_obs(), self._get_info()

    # ==========================================================
    # STEP
    # ==========================================================
    def step(self, action):
        self.step_counter += 1
        action = np.clip(action, -1.0, 1.0)
        self._apply_action(action)

        for _ in range(cfg.SIM_STEPS_PER_ACTION):
            p.stepSimulation(physicsClientId=self.physics_client)

        obs = self._get_obs()
        info = self._get_info()

        # Kaldırma takibi
        if info["bottle_height"] > cfg.LIFT_HEIGHT:
            self._bottle_was_lifted = True
        info["bottle_was_lifted"] = self._bottle_was_lifted

        # Reward hesapla
        reward = self._compute_dense_reward(obs, info, action)

        terminated = info["is_success"]
        truncated = self.step_counter >= cfg.MAX_EPISODE_STEPS
        self._prev_bottle_pos = obs["achieved_goal"].copy()

        return obs, reward, terminated, truncated, info

    # ==========================================================
    # FİZİKSEL TEMAS ALGILAMA
    # ==========================================================
    def _get_contact_info(self):
        """
        PyBullet getContactPoints ile gripper parmaklarının
        şişeye gerçekten dokunup dokunmadığını tespit eder.
        Hack-proof: fiziksel temas yalanlanamaz.
        """
        sol_contacts = p.getContactPoints(
            bodyA=self.robot_id, bodyB=self.bottle_id,
            linkIndexA=cfg.GRIPPER_JOINT_INDICES[0],
            physicsClientId=self.physics_client
        )
        sag_contacts = p.getContactPoints(
            bodyA=self.robot_id, bodyB=self.bottle_id,
            linkIndexA=cfg.GRIPPER_JOINT_INDICES[1],
            physicsClientId=self.physics_client
        )

        has_sol = len(sol_contacts) > 0
        has_sag = len(sag_contacts) > 0

        return {
            "sol_contact": has_sol,
            "sag_contact": has_sag,
            "single_contact": has_sol or has_sag,
            "dual_contact": has_sol and has_sag,
        }

    # ==========================================================
    # DENSE REWARD (v4 — temas bazlı, tek seferlik)
    # ==========================================================
    def _compute_dense_reward(self, obs, info, action):
        w = cfg.REWARD_WEIGHTS
        reward = 0.0

        ee_to_bottle = info["ee_to_bottle"]
        bottle_height = info["bottle_height"]
        contact = info["contact"]

        g_sol = p.getJointState(self.robot_id, cfg.GRIPPER_JOINT_INDICES[0], physicsClientId=self.physics_client)[0]
        g_sag = p.getJointState(self.robot_id, cfg.GRIPPER_JOINT_INDICES[1], physicsClientId=self.physics_client)[0]
        sol_range = cfg.GRIPPER_JOINT_LIMITS[6]["upper"] - cfg.GRIPPER_JOINT_LIMITS[6]["lower"]
        sag_range = abs(cfg.GRIPPER_JOINT_LIMITS[7]["upper"] - cfg.GRIPPER_JOINT_LIMITS[7]["lower"])
        sol_openness = (g_sol - cfg.GRIPPER_JOINT_LIMITS[6]["lower"]) / sol_range
        sag_openness = (cfg.GRIPPER_JOINT_LIMITS[7]["lower"] - g_sag) / sag_range
        gripper_openness = (sol_openness + sag_openness) / 2.0  
        gripper_is_open = gripper_openness > 0.5
        gripper_is_closed = gripper_openness < 0.3

        if not self._flag_grasp:
            reward += w["distance_to_bottle"] * ee_to_bottle

        if ee_to_bottle > 0.08 and gripper_is_open:
            reward += w["gripper_open_approach"]
        
        if contact["single_contact"] and gripper_is_closed:
            reward += w["gripper_close_contact"]

        # --- TEMAS VE KAVRAMA ÖDÜLLERİ ---
        if contact["single_contact"] and not self._flag_single_contact:
            reward += w["single_contact_bonus"]
            self._flag_single_contact = True

        if contact["dual_contact"] and gripper_is_open and not self._flag_dual_contact:
            reward += w["dual_contact_bonus"]
            self._flag_dual_contact = True

        if contact["dual_contact"] and gripper_is_closed and not self._flag_grasp:
            reward += w["grasp_bonus"]
            self._flag_grasp = True

        # Kaldırma ve Yükseklik
        if bottle_height > cfg.LIFT_HEIGHT and not self._flag_lift:
            reward += w["lift_bonus"]
            self._flag_lift = True

        if bottle_height > 0.01:
            reward += w["height_bonus"]  

        # Hedefe taşıma (Kaldırılmışsa)
        if self._bottle_was_lifted:
            reward += w["distance_to_goal"] * info["distance_to_goal"]

        if info["is_success"]:
            reward += w["success_bonus"]

        # --- YENİ: SÜRÜKLEME VE DEVİRME CEZALARI AKTİF ---
        bottle_pos = obs["achieved_goal"]
        horizontal_movement = np.linalg.norm(bottle_pos[:2] - self._prev_bottle_pos[:2])
        z_drop = self._prev_bottle_pos[2] - bottle_pos[2]

        if not self._flag_grasp:
            # Sürüklenirse veya devrilirse anında ceza!
            if z_drop > 0.01 or (horizontal_movement > 0.01 and bottle_height < cfg.LIFT_HEIGHT):
                reward += w["knock_penalty"]

        # Şişe yerdeyken yatayda oynatılırsa (kendine çekerse) ceza
        if horizontal_movement > 0.005 and bottle_height < cfg.LIFT_HEIGHT:
            reward += w["push_penalty"]

        reward += w["action_penalty"] * np.linalg.norm(action)

        return float(reward)

    # ==========================================================
    # OBSERVATION
    # ==========================================================
    def _get_obs(self):
        joint_positions, joint_velocities = [], []
        for idx in cfg.ARM_JOINT_INDICES:
            state = p.getJointState(self.robot_id, idx,
                                    physicsClientId=self.physics_client)
            joint_positions.append(state[0])
            joint_velocities.append(state[1])

        gripper_positions = []
        for idx in cfg.GRIPPER_JOINT_INDICES:
            state = p.getJointState(self.robot_id, idx,
                                    physicsClientId=self.physics_client)
            gripper_positions.append(state[0])

        ee_state = p.getLinkState(self.robot_id, cfg.END_EFFECTOR_LINK_INDEX,
                                  physicsClientId=self.physics_client)
        ee_position = np.array(ee_state[0])

        bottle_pos, _ = p.getBasePositionAndOrientation(
            self.bottle_id, physicsClientId=self.physics_client
        )
        bottle_position = np.array(bottle_pos)

        observation = np.concatenate([
            np.array(joint_positions, dtype=np.float32),
            np.array(joint_velocities, dtype=np.float32),
            np.array(gripper_positions, dtype=np.float32),
            ee_position.astype(np.float32),
            bottle_position.astype(np.float32),
            self.goal_position.astype(np.float32),
        ])

        return {
            "observation": observation,
            "achieved_goal": bottle_position.astype(np.float32),
            "desired_goal": self.goal_position.astype(np.float32),
        }

    def _get_info(self):
        bottle_pos, _ = p.getBasePositionAndOrientation(self.bottle_id, physicsClientId=self.physics_client)
        bottle_pos = np.array(bottle_pos)
        distance = np.linalg.norm(bottle_pos - self.goal_position)

        # --- YENİ: ŞİŞENİN MERKEZİNİ HEDEFLEME ---
        # Şişenin boyu 0.10, tam ortası Z ekseninde +0.05 yukarısıdır.
        bottle_center = bottle_pos.copy()
        bottle_center[2] += cfg.BOTTLE_HEIGHT / 2.0  

        # Gripper parmaklarının orta noktasını al
        state_sol = p.getLinkState(self.robot_id, cfg.GRIPPER_JOINT_INDICES[0], physicsClientId=self.physics_client)
        state_sag = p.getLinkState(self.robot_id, cfg.GRIPPER_JOINT_INDICES[1], physicsClientId=self.physics_client)
        gripper_midpoint = (np.array(state_sol[0]) + np.array(state_sag[0])) / 2.0
        
        # Mesafeyi şişenin dibine değil, ORTASINA göre hesapla
        ee_to_bottle = np.linalg.norm(gripper_midpoint - bottle_center)
        
        # Şişenin ortasına 4 cm yaklaştıysa hizalanmış say
        is_aligned = ee_to_bottle < 0.04 

        # Parmakların kapanma durumu
        g_sol = p.getJointState(self.robot_id, cfg.GRIPPER_JOINT_INDICES[0], physicsClientId=self.physics_client)[0]
        g_sag = p.getJointState(self.robot_id, cfg.GRIPPER_JOINT_INDICES[1], physicsClientId=self.physics_client)[0]
        is_squeezing = (g_sol < 0.20) and (g_sag > -0.20)

        bottle_height = bottle_pos[2] - self._bottle_initial_z
        is_grasping = ee_to_bottle < cfg.GRASP_THRESHOLD
        contact = self._get_contact_info()

        if cfg.LIFT_REQUIRED:
            is_success = (distance < cfg.SUCCESS_THRESHOLD and self._bottle_was_lifted)
        else:
            is_success = distance < cfg.SUCCESS_THRESHOLD

        return {
            "is_success": is_success,
            "distance_to_goal": distance,
            "ee_to_bottle": ee_to_bottle,
            "bottle_height": bottle_height,
            "bottle_was_lifted": self._bottle_was_lifted,
            "contact": contact,
            "is_grasping": is_grasping,
            "is_aligned": is_aligned,
            "is_squeezing": is_squeezing
        }

    # ==========================================================
    # AKSİYON (delta-based)
    # ==========================================================
    def _apply_action(self, action):
        for i, joint_idx in enumerate(cfg.ARM_JOINT_INDICES):
            limits = cfg.ARM_JOINT_LIMITS[joint_idx]
            current_pos = p.getJointState(
                self.robot_id, joint_idx,
                physicsClientId=self.physics_client
            )[0]
            delta = action[i] * cfg.ACTION_MAX_DELTA
            target_pos = np.clip(current_pos + delta,
                                 limits["lower"], limits["upper"])
            p.setJointMotorControl2(
                self.robot_id, joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=limits["effort"],
                maxVelocity=limits["velocity"],
                physicsClientId=self.physics_client
            )

        for i, joint_idx in enumerate(cfg.GRIPPER_JOINT_INDICES):
            limits = cfg.GRIPPER_JOINT_LIMITS[joint_idx]
            current_pos = p.getJointState(
                self.robot_id, joint_idx,
                physicsClientId=self.physics_client
            )[0]
            delta = action[4 + i] * cfg.GRIPPER_MAX_DELTA
            target_pos = np.clip(current_pos + delta,
                                 limits["lower"], limits["upper"])
            p.setJointMotorControl2(
                self.robot_id, joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=limits["effort"],
                maxVelocity=limits["velocity"],
                physicsClientId=self.physics_client
            )

    # ==========================================================
    # NESNE OLUŞTURMA
    # ==========================================================
    def _create_table(self):
        half_extents = [s / 2 for s in cfg.TABLE_SIZE]
        col_id = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half_extents,
            physicsClientId=self.physics_client
        )
        vis_id = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half_extents,
            rgbaColor=[0.6, 0.4, 0.2, 1.0],
            physicsClientId=self.physics_client
        )
        return p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=cfg.TABLE_POSITION,
            physicsClientId=self.physics_client
        )

    def _create_bottle(self, position):
        col_id = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=cfg.BOTTLE_RADIUS,
            height=cfg.BOTTLE_HEIGHT,
            physicsClientId=self.physics_client
        )
        vis_id = p.createVisualShape(
            p.GEOM_CYLINDER, radius=cfg.BOTTLE_RADIUS,
            length=cfg.BOTTLE_HEIGHT,
            rgbaColor=[0.2, 0.6, 0.9, 1.0],
            physicsClientId=self.physics_client
        )
        bottle_id = p.createMultiBody(
            baseMass=cfg.BOTTLE_MASS,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=position.tolist(),
            physicsClientId=self.physics_client
        )
        p.changeDynamics(
            bottle_id, -1,
            lateralFriction=1.5,
            spinningFriction=0.5,
            rollingFriction=0.01,
            physicsClientId=self.physics_client
        )
        return bottle_id

    def _create_goal_visual(self, position):
        vis_id = p.createVisualShape(
            p.GEOM_SPHERE, radius=cfg.SUCCESS_THRESHOLD,
            rgbaColor=[0.0, 1.0, 0.0, 0.4],
            physicsClientId=self.physics_client
        )
        return p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=vis_id,
            basePosition=position.tolist(),
            physicsClientId=self.physics_client
        )

    # ==========================================================
    # SPAWN
    # ==========================================================
    def _random_bottle_position(self):
        x = self.np_random.uniform(*cfg.BOTTLE_SPAWN_RANGE["x"])
        y = self.np_random.uniform(*cfg.BOTTLE_SPAWN_RANGE["y"])
        z = cfg.BOTTLE_SPAWN_RANGE["z"][0] + cfg.TABLE_HEIGHT + cfg.TABLE_SIZE[2]
        return np.array([x, y, z])

    def _random_goal_position(self, bottle_pos):
        goal_x = bottle_pos[0] + cfg.FIXED_GOAL_OFFSET_X
        goal_y = bottle_pos[1]
        goal_z = bottle_pos[2]
        return np.array([goal_x, goal_y, goal_z])

    # ==========================================================
    # RENDER
    # ==========================================================
    def render(self):
        if self.render_mode == "rgb_array":
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=cfg.CAMERA_TARGET,
                distance=cfg.CAMERA_DISTANCE,
                yaw=cfg.CAMERA_YAW,
                pitch=cfg.CAMERA_PITCH,
                roll=0, upAxisIndex=2,
                physicsClientId=self.physics_client
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=cfg.VIDEO_WIDTH / cfg.VIDEO_HEIGHT,
                nearVal=0.01, farVal=10.0,
                physicsClientId=self.physics_client
            )
            _, _, rgb, _, _ = p.getCameraImage(
                width=cfg.VIDEO_WIDTH, height=cfg.VIDEO_HEIGHT,
                viewMatrix=view_matrix, projectionMatrix=proj_matrix,
                renderer=p.ER_TINY_RENDERER,
                physicsClientId=self.physics_client
            )
            return np.array(rgb, dtype=np.uint8).reshape(
                cfg.VIDEO_HEIGHT, cfg.VIDEO_WIDTH, 4
            )[:, :, :3]
        return None

    def close(self):
        if p.isConnected(self.physics_client):
            p.disconnect(self.physics_client)


gym.register(
    id="RobotArmPickPlace-v0",
    entry_point="robot_arm_env:RobotArmPickPlaceEnv",
    max_episode_steps=cfg.MAX_EPISODE_STEPS,
)