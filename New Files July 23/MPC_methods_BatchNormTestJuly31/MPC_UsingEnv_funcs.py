import numpy as np
import torch
import random
from ASNN import train_ActionSequenceNN
from choose_action_UsingEnv import choose_action_func_UsingEnv
from RndUniformGeneratedActionSequences import generate_random_action_sequences
from ShiftParticlesReplaceWithRandom import ShiftParticlesReplaceWithRandom_func
from GenerateParticlesUsingASNN import GenerateParticlesUsingASNN_func_UsingEnv

def start_UsingEnv_MPC_wASNN(prob_vars, env, seed, model_ASNN, replay_buffer_ASNN, optimizer_ASNN, do_RS, use_ASNN):
    
    episode_reward_list_withASNN = []
    episode_success_rate_withASNN = [] # For Panda Gym envs

    nb_episode_success = 0 # For Panda Gym envs

    for episode in range(prob_vars.max_episodes):
        state, _ = env.reset(seed=seed)
        episode_reward = 0
        done = False
        actions_list = []
        if prob_vars.prob == "Pendulum":
            state = env.state.copy()
        if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "PandaReacherDense" or prob_vars.prob == "PandaPusherDense":
            prob_vars.goal_state = state['desired_goal'] # 3 components for Reach and for Push
            state = state['observation']
        if prob_vars.prob == "MuJoCoReacher":
            prob_vars.goal_state = np.array([state[4], state[5]])
            state = np.array([state[0], state[1], state[2], state[3], state[6], state[7], state[8], state[9]])
        if prob_vars.prob == "MuJoCoPusher":
            prob_vars.goal_state = np.array([state[20], state[21], state[22]])
        
        costs = []

        # if episode == 0:
        if episode <= 0: #10: # Generate new uniformly random action sequences
            particles = generate_random_action_sequences(prob_vars)
        
        else: # New episode action sequences using neural network
            particles = GenerateParticlesUsingASNN_func_UsingEnv(prob_vars, state, particles, model_ASNN)

        # for step in tqdm(range(max_steps)): # If you want to use tqdm, uncomment this line
        for step in range(prob_vars.max_steps):
            if prob_vars.prob == "Pendulum":
                state = env.state.copy()
            
            particles = np.clip(particles, prob_vars.action_low, prob_vars.action_high)

            best_particle, action, best_cost, particles = choose_action_func_UsingEnv(prob_vars, state, particles, do_RS, use_ASNN, model_ASNN, episode=episode, step=step, goal_state=prob_vars.goal_state)
            # best_particle, particles, cost = particle_filtering_cheating(particles, env, state, horizon, nb_reps=5, using_Env=usingEnv, episode=episode, step=step)
            
            costs.append(best_cost)
            
            if prob_vars.prob == "Pendulum" or prob_vars.prob == "MountainCarContinuous" or prob_vars.prob == "Pendulum_xyomega" or prob_vars.prob == "InvertedPendulum" or prob_vars.prob == "CartPoleContinuous":
                action = [best_particle[0]]
                actions_list.append(list(action))
            
            elif prob_vars.prob == "PanadaReacher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "LunarLanderContinuous" or prob_vars.prob == "PandaReacherDense" or prob_vars.prob == "PandaPusherDense":
                action = best_particle[:prob_vars.action_dim]
            
            elif prob_vars.prob == "CartPole" or prob_vars.prob == "Acrobot" or prob_vars.prob == "MountainCar" or prob_vars.prob == "LunarLander":
                action = int(action)
                actions_list.append(action)
            
            # Apply the first action from the optimized sequence
            next_state, reward, done, terminated, info = env.step(action)
            
            episode_reward += reward

            if prob_vars.prob == "Pendulum":
                next_state = env.state.copy()

            if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "PandaReacherDense" or prob_vars.prob == "PandaPusherDense":
                prob_vars.goal_state = next_state['desired_goal'] # 3 components
                next_state = next_state['observation']
                
            if prob_vars.prob == "MuJoCoReacher":
                next_state = np.array([next_state[0], next_state[1], next_state[2], next_state[3], next_state[6], next_state[7], next_state[8], next_state[9]])

            if prob_vars.prob == "CartPole" or prob_vars.prob == "Acrobot" or prob_vars.prob == "MountainCar" or prob_vars.prob == "LunarLander":
                replay_buffer_ASNN.push(state, prob_vars.goal_state, np.array([action]))
            else:
                replay_buffer_ASNN.push(state, prob_vars.goal_state, action)
            
            if len (replay_buffer_ASNN) < prob_vars.batch_size:
                pass
            else:
                train_ActionSequenceNN(model_ASNN, replay_buffer_ASNN, prob_vars.batch_size, optimizer_ASNN, num_epochs=1)
            
            if prob_vars.prob == "MuJoCoReacher":
                if np.sqrt(next_state[-2]**2+next_state[-1]**2) < 0.05:
                    # print("Reached target position \n")
                    done = True
            
            done = done or terminated
            if done:
                nb_episode_success += 1
                break
            
            state = next_state
            
            # Shift all particles to the left by removing the first element
            particles[:, :-prob_vars.action_dim] = particles[:, prob_vars.action_dim:]
            
        episode_reward_list_withASNN.append(episode_reward)

        episode_success_rate_withASNN.append(nb_episode_success/(episode+1)) # Episodic success rate for Panda Gym envs

    if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "PandaReacherDense" or prob_vars.prob == "PandaPusherDense":
        return episode_reward_list_withASNN, episode_success_rate_withASNN
    else:
        return episode_reward_list_withASNN



def start_UsingEnvrand_RS(prob_vars, env, seed, model_ASNN, replay_buffer_ASNN, optimizer_ASNN, do_RS, do_QRNN_step_rnd, use_ASNN):

    episode_reward_list = []
    episode_success_rate = [] # For Panda Gym envs

    nb_episode_success = 0 # For Panda Gym envs

    # for episode in range(tqdm(max_episodes)): # If you want to use tqdm, uncomment this lines
    for episode in range(prob_vars.max_episodes):
        state, _ = env.reset(seed=seed)
        episode_reward = 0
        done = False
        actions_list = []
        if prob_vars.prob == "Pendulum":
            state = env.state.copy()
        if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "PandaReacherDense" or prob_vars.prob == "PandaPusherDense":
            prob_vars.goal_state = state['desired_goal'] # 3 components
            state = state['observation']#[:3] # 6 components for Reacher, 18 components for Pusher
        if prob_vars.prob == "MuJoCoReacher":
            prob_vars.goal_state = np.array([state[4], state[5]])
            state = np.array([state[0], state[1], state[2], state[3], state[6], state[7], state[8], state[9]])
        if prob_vars.prob == "MuJoCoPusher":
            prob_vars.goal_state = np.array([state[20], state[21], state[22]])
            
        costs = []
        
        # Generate new uniformly random action sequences
        particles = generate_random_action_sequences(prob_vars)
        
        # for step in range(tqdm(max_steps)): # If you want to use tqdm, uncomment this lines
        for step in range(prob_vars.max_steps):
            
            if prob_vars.prob == "Pendulum":
                state = env.state.copy()
            
            if do_RS or do_QRNN_step_rnd: # Generate new uniformly random action sequences at each step in the env
   
                particles = generate_random_action_sequences(prob_vars)

            particles = np.clip(particles, prob_vars.action_low, prob_vars.action_high)

            best_particle, action, best_cost, particles = choose_action_func_UsingEnv(prob_vars, state, particles, do_RS, use_ASNN, model_ASNN, episode=episode, step=step, goal_state=prob_vars.goal_state)
            
            actions_list.append(action)
            
            costs.append(best_cost)
            
            if prob_vars.prob == "Pendulum" or prob_vars.prob == "MountainCarContinuous" or prob_vars.prob == "Pendulum_xyomega" or prob_vars.prob == "InvertedPendulum" or prob_vars.prob == "CartPoleContinuous":
                action = [best_particle[0]]
            
            elif prob_vars.prob == "PanadaReacher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "LunarLanderContinuous" or prob_vars.prob == "PandaReacherDense" or prob_vars.prob == "PandaPusherDense":
                action = best_particle[:prob_vars.action_dim]
            
            elif prob_vars.prob == "CartPole" or prob_vars.prob == "Acrobot" or prob_vars.prob == "MountainCar" or prob_vars.prob == "LunarLander":
                action = int(action)
            
            # Apply the first action from the optimized sequence
            next_state, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            actions_list.append(action)
            
            if prob_vars.prob == "Pendulum":
                # state = env.state.copy()
                next_state = env.state.copy()
            if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "PandaReacherDense" or prob_vars.prob == "PandaPusherDense":
                prob_vars.goal_state = next_state['desired_goal'] # 3 components
                next_state = next_state['observation']#[:3] # 6 components
            if prob_vars.prob == "MuJoCoReacher":
                next_state = np.array([next_state[0], next_state[1], next_state[2], next_state[3], next_state[6], next_state[7], next_state[8], next_state[9]])
            
            done = done or truncated
            if done:
                nb_episode_success += 1
                break
            
            if not do_RS or not do_QRNN_step_rnd: # basic method: Shift particles to the left and add new actions sampled for a uniform distribution
                
                particles = ShiftParticlesReplaceWithRandom_func(prob_vars, particles)
                
            particles = np.clip(particles, prob_vars.action_low, prob_vars.action_high)
            
            state = next_state
        
        episode_reward_list.append(episode_reward)

        episode_success_rate.append(nb_episode_success/(episode+1)) # Episodic success rate for Panda Gym envs

    if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "PandaReacherDense" or prob_vars.prob == "PandaPusherDense":
        return episode_reward_list, episode_success_rate
    else:
        return episode_reward_list
    
