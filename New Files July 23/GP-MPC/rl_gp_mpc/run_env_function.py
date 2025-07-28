import time
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.core import Env

from rl_gp_mpc import GpMpcController
from rl_gp_mpc import ControlVisualizations
from rl_gp_mpc.config_classes.visu_config import VisuConfig
from rl_gp_mpc.config_classes.total_config import Config
# import os

NUM_DECIMALS_REPR = 3
np.set_printoptions(precision=NUM_DECIMALS_REPR, suppress=True)

def run_env(env: Env, env_name, seed, control_config:Config, visu_config: VisuConfig, random_actions_init=10, num_steps=150, verbose=True):
    visu_obj = ControlVisualizations(env=env, env_name=env_name, num_steps=num_steps, control_config=control_config, visu_config=visu_config)

    obs = env.reset(seed=seed)[0]
    
    # print("obs ", obs, "\n")
    
    if env_name == "PandaReach-v3" or env_name == "PandaReachDense-v3" or env_name == "PandaPush-v3" or env_name == "PandaPushDense-v3":
        obs = obs['observation']
        ctrl_obj = GpMpcController(observation_low=env.observation_space['observation'].low,
                                observation_high=env.observation_space['observation'].high, 
                                action_low=env.action_space.low,
                                action_high=env.action_space.high, 
                                config=control_config)
        
    elif env_name == "Reacher-v5" or env_name == "Pusher-v5":
        # obs = np.array([obs[0], obs[1], obs[2], obs[3], obs[6], obs[7], obs[8], obs[9]])
        # goal_state = np.array([obs[4], obs[5]])
        ctrl_obj = GpMpcController(observation_low=env.observation_space.low,
                                observation_high=env.observation_space.high,
                                # observation_low=env.observation_space.low[:8],
                                # observation_high=env.observation_space.high[:8], 
                                action_low=env.action_space.low,
                                action_high=env.action_space.high, 
                                config=control_config)
    else:
        ctrl_obj = GpMpcController(observation_low=env.observation_space.low,
                                observation_high=env.observation_space.high, 
                                action_low=env.action_space.low,
                                action_high=env.action_space.high, 
                                config=control_config)
    
    sum_reward = 0
    
    for idx_ctrl in range(num_steps):
        action_is_random = (idx_ctrl < random_actions_init)
        action = ctrl_obj.get_action(obs_mu=obs, random=action_is_random)

        iter_info = ctrl_obj.get_iter_info()

        cost, cost_var = ctrl_obj.compute_cost_unnormalized(obs, action)
        # visu_obj.update(obs=obs, reward=-cost, action=action, env=env, iter_info=iter_info)

        obs_new, reward, done, truncated, info = env.step(action)
        
        if env_name == "PandaReach-v3" or env_name == "PandaReachDense-v3" or env_name == "PandaPush-v3" or env_name == "PandaPushDense-v3":
            obs_new = obs_new['observation']
        # if env_name == "Reacher-v5":
        #     obs_new = np.array([obs_new[0], obs_new[1], obs_new[2], obs_new[3], obs_new[6], obs_new[7], obs_new[8], obs_new[9]])

        ctrl_obj.add_memory(obs=obs, action=action, obs_new=obs_new,
                            reward=-cost,
                            predicted_state=iter_info.predicted_states[1],
                            predicted_state_std=iter_info.predicted_states_std[1])
        obs = obs_new
        # if verbose:
        #     print(str(iter_info))
            
        # if obs[0] > 0.45:
        #     break
            
        sum_reward += reward
        
        if done or truncated:
            break
    # print("sum_reward ", sum_reward, "\n")


    # visu_obj.save(ctrl_obj)
    ctrl_obj.check_and_close_processes()
    # env.__exit__()
    visu_obj.close()
    return sum_reward
    # return visu_obj.get_costs(), sum_reward

def save_data(prob, method_name, episodic_rep_returns, mean_episodic_returns, std_episodic_returns):

        # # Get the folder where this script is located
        # origin_folder = os.path.dirname(os.path.abspath(__file__))
        # # Construct full path to save
        # save_path = os.path.join(origin_folder, f"{prob}_{method_name}_results.npz")

        np.savez(
        # save_path,
        f"{prob}_{method_name}_results.npz",
        episode_rewards=episodic_rep_returns,
        mean_rewards=mean_episodic_returns,
        std_rewards=std_episodic_returns
        )

def run_env_multiple(env, env_name, control_config:Config, visu_config: VisuConfig, seeds, num_episodes, random_actions_init=10, num_steps=150, verbose=True):
    return_reps = []
    method_name = "GP-MPC"
    num_reps = len(seeds)
    
    if env_name == "PandaReach-v3":
        prob = "PandaReach"
    elif env_name == "PandaReachDense-v3":
        prob = "PandaReachDense"
    elif env_name == "PandaPush-v3":
        prob = "PandaPush"
    elif env_name == "PandaPushDense-v3":
        prob = "PandaPushDense"
    elif env_name == "Reacher-v5":
        prob = "Reacher"
    elif env_name == "Pusher-v5":
        prob = "Pusher"
    elif env_name == "Pendulum-v1":
        prob = "Pendulum"
    elif env_name == "LunarLanderContinuous-v3":
        prob = "LunarLander"
    elif env_name == "MountainCarContinuous-v0":
        prob = "MountainCar"
    elif env_name == "InvertedPendulum-v5":
        prob = "InvertedPendulum"
    
    for rep in range(num_reps): # repetition over the different seeds
        # costs_runs = []
        returns = []
        seed = seeds[rep]
        for episode in range(num_episodes):
            if prob == "PandaReach" or prob == "PandaReachDense" or prob == "PandaPush" or prob == "PandaPushDense": #  or prob == "Reacher" 
                if seed == 0:
                    control_config_seed = control_config[0]
                elif seed == 8:
                    control_config_seed = control_config[1]
                elif seed == 15:
                    control_config_seed = control_config[2]
                return_iter = run_env(env, env_name, seed, control_config_seed, visu_config, random_actions_init, num_steps, verbose=verbose)
            
            else:
                return_iter = run_env(env, env_name, seed, control_config, visu_config, random_actions_init, num_steps, verbose=verbose)
            # costs_iter, return_iter = run_env(env, control_config, visu_config, random_actions_init, num_steps, verbose=verbose)
            # costs_runs.append(costs_iter)
            returns.append(np.float64(return_iter))
            time.sleep(1)
            print("episode ", episode, "return_iter ", return_iter, "\n")
        return_reps.append(returns)
        
    # print("costs_runs ", costs_runs, "\n")
    # print("returns ", returns, "\n")
    # costs_runs = np.array(costs_runs)

    return_reps = np.array(return_reps)

    #costs_runs_mean = np.mean(costs_runs, axis=0)
    # costs_runs_std = np.std(costs_runs, axis=0)
    
    return_reps_mean = np.mean(return_reps, axis=0)
    return_reps_std = np.std(return_reps, axis=0)
    
    save_data(prob, method_name, return_reps, return_reps_mean, return_reps_std)
    
    print("return_reps ", list(return_reps), "\n")
    print("return_reps_mean ", list(return_reps_mean), "\n")
    print("return_reps_std ", list(return_reps_std), "\n")

    # x_axis = np.arange(len(costs_runs_mean))
    # x_axis = np.arange(len(returns))
    # x_axis = np.arange(len(return_reps_mean))
    # fig, ax = plt.subplots(figsize=(10, 5))
    # ax.plot(x_axis, costs_runs_mean)
    # ax.fill_between(x_axis, costs_runs_mean - costs_runs_std, costs_runs_mean + costs_runs_std, alpha=0.4)
    # plt.figure(1)
    # plt.plot(x_axis, costs_runs_mean)
    # plt.fill_between(x_axis, costs_runs_mean - costs_runs_std, costs_runs_mean + costs_runs_std, alpha=0.4)
    # plt.title(f"Costs of multiples {env_name} runs")
    # plt.ylabel("Cost")
    # plt.xlabel("Env iteration")
    # plt.savefig(f'multiple_runs_costs_{env_name}.png')
    
    env.close()
    
    # plt.figure(2)
    # plt.plot(returns_runs_mean) # x_axis, 
    # plt.plot(x_axis, returns)
    # plt.plot(x_axis, return_reps_mean)
    plt.plot(return_reps_mean)
    plt.fill_between(return_reps_mean - return_reps_std, return_reps_mean + return_reps_std, alpha=0.4) # x_axis, 
    plt.title(f"Returns of multiples {env_name} runs")
    plt.ylabel("Episode Return")
    plt.xlabel("Env iteration")
    plt.savefig(f'multiple_runs_returns_{env_name}.png')
    
    plt.show()

    