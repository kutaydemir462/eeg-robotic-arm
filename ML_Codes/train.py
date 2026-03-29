import argparse
import os
import time
import numpy as np
import multiprocessing

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize, DummyVecEnv

import config as cfg
import robot_arm_env  # noqa
import gymnasium as gym

from evaluate import frames_to_mp4 

def make_env(rank, seed=0):
    def _init():
        env = gym.make("RobotArmPickPlace-v0", render_mode=None)
        env.reset(seed=seed + rank)
        return env
    return _init

def make_vec_env(n_envs):
    return SubprocVecEnv([make_env(i) for i in range(n_envs)])

class SaveBestModelCallback(BaseCallback):
    def __init__(self, eval_freq, eval_episodes, save_path, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.save_path = save_path
        self.best_success_rate = 0.0
        self.eval_env = None

    def _init_callback(self):
        os.makedirs(self.save_path, exist_ok=True)
        self.eval_env = gym.make("RobotArmPickPlace-v0")

    def _on_step(self):
        if self.n_calls % self.eval_freq != 0:
            return True

        successes = 0
        for ep in range(self.eval_episodes):
            obs, info = self.eval_env.reset(seed=ep)
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
            if info.get("is_success", False):
                successes += 1

        success_rate = successes / self.eval_episodes
        self.logger.record("eval/success_rate", success_rate)
        if self.verbose:
            print(f"\n[Eval] Success: {success_rate:.2%} ({successes}/{self.eval_episodes})")

        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
            self.model.save(os.path.join(self.save_path, "best_model"))
            if isinstance(self.training_env, VecNormalize):
                self.training_env.save(os.path.join(self.save_path, "vec_normalize.pkl"))
            if self.verbose:
                print("★ Yeni en iyi model ve normalizasyon verisi kaydedildi")

        return True

class VideoRecorderCallback(BaseCallback):
    def __init__(self, record_freq, video_folder, verbose=1):
        super().__init__(verbose)
        self.record_freq = record_freq
        self.video_folder = video_folder
        os.makedirs(self.video_folder, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.record_freq == 0:
            real_step = self.num_timesteps
            if self.verbose:
                print(f"\n[Otomatik Video] {real_step}. adım için test bölümü kaydediliyor...")
            
            vid_env = DummyVecEnv([lambda: gym.make("RobotArmPickPlace-v0", render_mode="rgb_array")])
            
            if isinstance(self.training_env, VecNormalize):
                vid_env = VecNormalize(vid_env, norm_obs=True, norm_reward=False, clip_obs=10.)
                vid_env.obs_rms = self.training_env.obs_rms 
                vid_env.training = False 
            
            obs = vid_env.reset()
            done = False
            frames = []
            ep_reward = 0.0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, dones, infos = vid_env.step(action)
                ep_reward += reward[0]
                
                try:
                    frame = vid_env.envs[0].render()
                    if frame is not None:
                        frames.append(frame)
                except Exception:
                    pass
                    
                done = dones[0]
            
            vid_env.close()
            
            if frames:
                video_name = f"train_step_{real_step}_rew_{ep_reward:.0f}.mp4"
                video_path = os.path.join(self.video_folder, video_name)
                frames_to_mp4(frames, video_path, fps=cfg.VIDEO_FPS)
                if self.verbose:
                    print(f"🎬 Video başarıyla kaydedildi: {video_path}")
                    
        return True

class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, print_freq=5000):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.print_freq = print_freq
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self):
        if self.n_calls % self.print_freq == 0:
            elapsed = time.time() - self.start_time
            fps = self.n_calls / max(elapsed, 1)
            progress = self.num_timesteps / self.total_timesteps * 100
            print(f"[{progress:5.1f}%] {self.num_timesteps}/{self.total_timesteps} | FPS: {fps:.0f}")
        return True

class CurriculumCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.total_timesteps = total_timesteps

    def _on_step(self):
        progress = self.num_timesteps / self.total_timesteps
        if progress < 0.4:
            cfg.GRASP_THRESHOLD, cfg.LIFT_HEIGHT = 0.10, 0.03
        elif progress < 0.8:
            cfg.GRASP_THRESHOLD, cfg.LIFT_HEIGHT = 0.08, 0.04
        else:
            cfg.GRASP_THRESHOLD, cfg.LIFT_HEIGHT = 0.05, 0.06
        return True

def train(total_timesteps=None, resume=False):
    timesteps = total_timesteps or cfg.TOTAL_TIMESTEPS
    n_envs = min(12, multiprocessing.cpu_count() - 2)
    print(f"\n🚀 Parallel Env: {n_envs}")
    
    env = make_vec_env(n_envs)
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    if resume and os.path.exists(cfg.BEST_MODEL_PATH + ".zip"):
        print("Checkpoint'ten devam ediliyor...")
        if os.path.exists(os.path.join(cfg.MODELS_DIR, "vec_normalize.pkl")):
            env = VecNormalize.load(os.path.join(cfg.MODELS_DIR, "vec_normalize.pkl"), env)
        env.training = True 
        model = SAC.load(cfg.BEST_MODEL_PATH, env=env, tensorboard_log=cfg.LOGS_DIR)
    else:
        print("Yeni model eğitiliyor...")
        model = SAC(
            policy="MultiInputPolicy",
            env=env,
            replay_buffer_class=DictReplayBuffer,
            verbose=0,
            seed=cfg.SEED,
            tensorboard_log=cfg.LOGS_DIR,
            **cfg.SAC_PARAMS,
        )

    save_freq_adjusted = max(cfg.CHECKPOINT_FREQ // n_envs, 1)

    callbacks = CallbackList([
        SaveBestModelCallback(cfg.EVAL_FREQ, cfg.EVAL_EPISODES, cfg.MODELS_DIR),
        CheckpointCallback(save_freq=save_freq_adjusted, save_path=cfg.CHECKPOINT_DIR, name_prefix="sac_dense"),
        VideoRecorderCallback(record_freq=save_freq_adjusted, video_folder=cfg.VIDEOS_DIR), # OTOMATİK VİDEO EKLENDİ
        ProgressCallback(timesteps, print_freq=5000),
        CurriculumCallback(timesteps),
    ])

    print("\nEğitim başlıyor...\n")
    start_time = time.time()

    try:
        model.learn(total_timesteps=timesteps, callback=callbacks, tb_log_name="SAC_DENSE")
    except KeyboardInterrupt:
        print("\nDurduruldu, kaydediliyor...")
        model.save(os.path.join(cfg.MODELS_DIR, "interrupted"))
        env.save(os.path.join(cfg.MODELS_DIR, "vec_normalize_interrupted.pkl"))

    model.save(os.path.join(cfg.MODELS_DIR, "final_model"))
    env.save(os.path.join(cfg.MODELS_DIR, "vec_normalize.pkl"))
    print(f"\nBitti. Süre: {(time.time()-start_time)/60:.1f} dk")
    env.close()

if __name__ == "__main__":
    multiprocessing.freeze_support() 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args.timesteps, args.resume)