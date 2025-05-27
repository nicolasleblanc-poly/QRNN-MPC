import gymnasium as gym
import numpy as np
from sb3_contrib import TQC
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
import os
import gc

def save_data(prob, method_name, episodic_rep_returns, mean_episodic_returns, std_episodic_returns):
    
    # Get the folder where this script is located
    origin_folder = os.path.dirname(os.path.abspath(__file__))
    # Construct full path to save
    save_path = os.path.join(origin_folder, f"{prob}_{method_name}_results.npz")

    np.savez(
        save_path,
        f"{prob}_{method_name}_results.npz",
        episode_rewards=episodic_rep_returns,
        mean_rewards=mean_episodic_returns,
        std_rewards=std_episodic_returns
        )
    
    # np.savez(
    # f"{prob}_{method_name}_results.npz",
    # episode_rewards=episodic_rep_returns,
    # mean_rewards=mean_episodic_returns,
    # std_rewards=std_episodic_returns
    # )
    

class EpisodicReturnLogger(BaseCallback):
    def __init__(self, max_episodes):
        super().__init__()
        self.max_episodes = max_episodes
        self.episode_count = 0
        self.episodic_returns = []  # Stores episodic returns
        self.timesteps = []         # Stores corresponding timesteps

    def _on_step(self) -> bool:
        # Log at the end of each episode (SB3 updates `ep_info_buffer` after an episode ends)
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_count += 1
                episodic_return = info['episode']['r']  # Cumulative reward of the episode
                self.episodic_returns.append(episodic_return)
                self.timesteps.append(self.num_timesteps)  # Current timestep
                
                if self.episode_count >= self.max_episodes:
                    return False
                
        return True


# def run(env_seeds, prob, method_name, steps_per_episode, max_episodes):
#     episodic_return_seeds = []

#     for seed in env_seeds:
#         # === Create environment ===
#         if prob == "CartPole":
#             env = gym.make("CartPole-v0")
#         elif prob == "Acrobot":
#             env = gym.make("Acrobot-v1")
#         elif prob == "MountainCar":
#             env = gym.make("MountainCar-v0")
#         elif prob == "LunarLander":
#             env = gym.make("LunarLander-v3")
#         elif prob == "Pendulum":
#             env = gym.make("Pendulum-v1")
#         elif prob == "InvertedPendulum":
#             env = gym.make("InvertedPendulum-v5")
#         elif prob == "MountainCarContinuous":
#             env = gym.make("MountainCarContinuous-v0")
#         elif prob == "LunarLanderContinuous":
#             env = gym.make("LunarLanderContinuous-v3")
#         elif prob == "Reacher":
#             env = gym.make("Reacher-v5")

#         # === Seed the environment ===
#         env = DummyVecEnv([lambda: env])
#         env = VecMonitor(env, "logs/")
#         env.seed(seed)

#         # === Initialize model ===
#         if method_name == "A2C":
#             model = A2C("MlpPolicy", env)
#         elif method_name == "DDPG":
#             model = DDPG("MlpPolicy", env)
#         elif method_name == "PPO":
#             model = PPO("MlpPolicy", env)
#         elif method_name == "SAC":
#             model = SAC("MlpPolicy", env)
#         elif method_name == "TD3":
#             model = TD3("MlpPolicy", env)
#         elif method_name == "TQC":
#             model = TQC("MlpPolicy", env)

#         # === Manual training loop ===
#         episodic_returns = []

#         for ep in range(max_episodes):
#             obs = env.reset()
#             done = False
#             total_reward = 0
#             step = 0

#             while not done and step < steps_per_episode:
#                 action, _ = model.predict(obs, deterministic=False)
#                 obs, reward, done, info = env.step(action)
#                 total_reward += reward[0]  # reward is a vector due to VecEnv
#                 step += 1

#                 # Train on this step (rollout buffer will collect transitions)
#                 model.train()

#             episodic_returns.append(total_reward)

#         episodic_return_seeds.append(episodic_returns)
#         print(f"Finished seed {seed} with {len(episodic_returns)} episodes")

#     # === Save results ===
#     episodic_return_seeds = np.array(episodic_return_seeds)
#     mean_episodic_return = np.mean(episodic_return_seeds, axis=0)
#     std_episodic_return = np.std(episodic_return_seeds, axis=0)

#     save_data(prob, method_name, episodic_return_seeds, mean_episodic_return, std_episodic_return)

class ObservationOnlyWrapper(gym.Wrapper):
    """
    Wrapper that modifies Panda Gym environments to return only the observation component.
    This is useful for SB3 algorithms that don't need the full HER-style state dict.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Override observation_space to match the extracted observation
        self.observation_space = env.observation_space["observation"]
        
    def reset(self, **kwargs):
        """
        Reset the environment and return only the observation
        """
        obs_dict, info = self.env.reset(**kwargs)
        # print("obs_dict['observation']: ", obs_dict['observation'], "\n")
        return obs_dict['observation'], info #['observation']
    
    def step(self, action):
        """
        Take a step in the environment and return (obs, reward, done, info)
        with obs being just the observation component
        """
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        obs = obs_dict['observation']
        # print("obs_dict: ", obs_dict, "\n")
        return obs, reward, terminated, truncated, info


def run(env_seeds, prob, method_name, steps_per_episode, max_episodes):
    episodic_return_seeds = []
    
    for seed in env_seeds:
        if prob == "CartPole":
            env = gym.make("CartPole-v0")
        elif prob == "Acrobot":
            env = gym.make("Acrobot-v1")
        elif prob == "MountainCar":
            env = gym.make("MountainCar-v0")
        elif prob == "LunarLander":
            env = gym.make("LunarLander-v3")
        elif prob == "Pendulum":
            env = gym.make("Pendulum-v1")
        elif prob == "InvertedPendulum":
            env = gym.make("InvertedPendulum-v5")
        elif prob == "MountainCarContinuous":
            env = gym.make("MountainCarContinuous-v0")
        elif prob == "LunarLanderContinuous":
            env = gym.make("LunarLanderContinuous-v3")
        elif prob == "Reacher":
            env = gym.make("Reacher-v5")
        elif prob == "PandaReach":
            env = gym.make("PandaReach-v3")
            env = ObservationOnlyWrapper(env)  # Wrap the environment to only return the observation
        elif prob == "PandaReacherDense":
            env = gym.make("PandaReachDense-v3")
            env = ObservationOnlyWrapper(env)  # Wrap the environment to only return the observation
        
        
        if method_name == "A2C":
            model = A2C("MlpPolicy", env)
        elif method_name == "DDPG":
            model = DDPG("MlpPolicy", env)
        elif method_name == "PPO":
            model = PPO("MlpPolicy", env)
        elif method_name == "SAC":
            model = SAC("MlpPolicy", env)
        elif method_name == "TD3":
            model = TD3("MlpPolicy", env)
        elif method_name == "TQC":
            model = TQC("MlpPolicy", env)
        # env = gym.make("LunarLander-v3", render_mode="human")
        # env = Monitor(env)  # Wraps env to log episode stats
        env = DummyVecEnv([lambda: env])  # Required for SB3
        env = VecMonitor(env, "logs/")  # Saves logs to "logs/"
        env.seed(seed)  # Set the seed for reproducibility

        # model = A2C("MlpPolicy", env) # , verbose=1
        # model.learn(total_timesteps=1000)
        # model.save("a2c_cartpole")

        # Initialize callback
        return_logger = EpisodicReturnLogger(max_episodes)

        # if prob == "Acrobot":
        #     model.learn(total_timesteps=max_episodes, callback=return_logger)
        # else:
        model.learn(total_timesteps=steps_per_episode*max_episodes, callback=return_logger)
        
        episodic_return_seeds.append(return_logger.episodic_returns)
        print("len(return_logger.episodic_returns): ", len(return_logger.episodic_returns))
        
        env.close()
        del env
        del model
        gc.collect()
        
    episodic_return_seeds = np.array(episodic_return_seeds)

    mean_episodic_return = np.mean(episodic_return_seeds, axis=0)
    std_episodic_return = np.std(episodic_return_seeds, axis=0)
    
    save_data(prob, method_name, episodic_return_seeds, mean_episodic_return, std_episodic_return)
    
    
    
    
