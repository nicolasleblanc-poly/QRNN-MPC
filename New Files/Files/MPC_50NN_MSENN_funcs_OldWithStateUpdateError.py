import numpy as np
import torch
import random
from scipy.optimize import minimize
from state_pred_models import quantile_loss
from ASNN import train_ActionSequenceNN
from choose_action_50NN_MSENN import choose_action_func_50NN_MSENN, choose_action_func_50NN_MSENN_LBFGSB

def start_50NN_MSENN_MPC_wASGNN(prob_vars, env, seed, model_state, replay_buffer_state, optimizer_state, loss_state, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, use_sampling, use_mid, use_ASGNN):
    
    episode_reward_list_withASGNN = []
    episode_success_rate_withASGNN = [] # For Panda Gym envs

    # goal_state = prob_vars.goal_state

    nb_episode_success = 0 # For Panda Gym envs

    # if prob == "Pendulum_xyomega":
    #     states_low = torch.tensor([-1, -1, -8])
    #     states_high = torch.tensor([1, 1, 8])

    # if prob == "PandaReacher" or prob == "PandaPusher":
    #     states_low = torch.tensor(-10)
    #     states_high = torch.tensor(10)

    # if prob == "MuJoCoReacher":
    #     states_low = torch.tensor([-1, -1, -1, -1, -100, -100, -torch.inf, -torch.inf])
    #     states_high = torch.tensor([1, 1, 1, 1, 100, 100, torch.inf, torch.inf])

    for episode in range(prob_vars.max_episodes):
        state, _ = env.reset(seed=seed)
        episode_reward = 0
        # is_success_bool = False # For Panda Gym envs
        done = False
        actions_list = []
        if prob_vars.prob == "Pendulum":
            state = env.state.copy()
        if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "PandaReacherDense":
            prob_vars.goal_state = state['desired_goal'] # 3 components for Reach and for Push
            state = state['observation'] #[:3] # 6 components for Reach, 18 components for Push
            # print("goal_state ", goal_state, "\n")
        if prob_vars.prob == "MuJoCoReacher":
            prob_vars.goal_state = np.array([state[4], state[5]])
            state = np.array([state[0], state[1], state[2], state[3], state[6], state[7], state[8], state[9]])
        if prob_vars.prob == "MuJoCoPusher":
            prob_vars.goal_state = np.array([state[20], state[21], state[22]])
        
        costs = []
        episodic_step_rewards = []
        episodic_step_reward_values = []
        
        # if episode == 0:
        #     # Random initial action sequence
        #     # initial_action_sequence = np.random.randint(0, 2, horizon)
        #     init_particles = [np.random.uniform(-2, 2, horizon) for _ in range(num_particles)] # 2*horizon_list[0]
        # else:
        #     # Add small random noise to encourage exploration (for now this can stay the same)
        #     # particles = np.clip(best_action_sequence + np.random.randint(0, 2, horizon), 0, 1)
        #     for i in range(len(particles)):
        #         # print("best_particle ", best_particle, "\n")
        #         # print("np.random.uniform(-0.5, 0.5, horizon) ", np.random.uniform(-0.5, 0.5, horizon), "\n")
        #         particles[i] = np.clip(best_particle + np.random.uniform(-0.5, 0.5, horizon), -2, 2)
        #         # particles = np.clip(init_particles + np.random.randint(0, 2, len(init_particles)), 0, 1)
        
        # particles = [np.random.uniform(-2, 2, horizon) for _ in range(num_particles)] # 2*horizon_list[0]
        
        
        # if episode == 0:
        if episode <= 0: #10:
            if prob_vars.prob == "CartPole":
                particles = np.random.randint(0, 2, (prob_vars.num_particles, prob_vars.horizon))
            elif prob_vars.prob == "Acrobot" or prob_vars.prob == "MountainCar":
                particles = np.random.randint(0, 3, (prob_vars.num_particles, prob_vars.horizon))
            elif prob_vars.prob == "LunarLander":
                particles = np.random.randint(0, 4, (prob_vars.num_particles, prob_vars.horizon))
            elif prob_vars.prob == "PandaReacher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "LunarLanderContinuous" or prob_vars.prob == "PandaReacherDense":
                particles = np.random.uniform(prob_vars.action_low, prob_vars.action_high, (prob_vars.num_particles, prob_vars.action_dim*prob_vars.horizon))
            else: # Pendulum, MountainCarContinuous, Pendulum_xyomega
                particles = np.random.uniform(prob_vars.action_low, prob_vars.action_high, (prob_vars.num_particles, prob_vars.horizon))
        
        else: # New episode action sequences using neural network
            if prob_vars.prob == "CartPole" or prob_vars.prob == "Acrobot" or prob_vars.prob == "MountainCar" or prob_vars.prob == "LunarLander": # Discrete actions
                action_probs = model_ASN(torch.tensor(state, dtype=torch.float32), torch.tensor(prob_vars.goal_state, dtype=torch.float32))[0]
                
                # print("actions_probs.detach().numpy( ", action_probs.detach().numpy(), "\n")
                
                # print("np.random.multinomial(num_particles, action_probs.detach().numpy(), size=num_particles) ", np.random.multinomial(num_particles, action_probs.detach().numpy(), size=num_particles), "\n")
                # particles[:, 0] = np.random.multinomial(num_particles, action_probs.detach().numpy(), size=num_particles)
                
                particles[:, 0] = np.random.choice(prob_vars.nb_actions, prob_vars.num_particles, p=action_probs.detach().numpy())
                
                # for loop_iter in range(num_particles):
                #     particles[loop_iter, 0] = np.random.normal(action_mu.detach().numpy(), action_sigma.detach().numpy())
                
                # for j in range(horizon):
                for j in range(1, prob_vars.horizon):
                    sim_states = torch.tensor(state, dtype=torch.float32).repeat(len(particles), 1)
                    actions = torch.tensor(particles[:, j], dtype=torch.float32).reshape(len(particles),1)
                    
                    sim_states = torch.clip(sim_states, prob_vars.states_low, prob_vars.states_high)
                    # sim_states = 2 * ((sim_states - prob_vars.states_low) / (prob_vars.states_high - prob_vars.states_low)) - 1
                    actions = torch.clip(actions, prob_vars.action_low, prob_vars.action_high)

                    """
                    # Predict next states using the quantile model_QRNN
                    # predicted_quantiles = model_QRNN(sim_states, actions)
                    # # mid_quantile = predicted_quantiles[:, predicted_quantiles.size(1) // 2, :]
                    # # next_states = mid_quantile

                    # if use_sampling:
                        # '''
                    #     ############# Need to add the sorta quantile ponderated sum here #############
                    #     # chosen_quantiles = np.random.uniform(0, 1, (num_particles, state_dim))
                    #     chosen_quantiles = torch.rand(prob_vars.num_particles, prob_vars.state_dim)
                    #     # https://pytorch.org/docs/main/generated/torch.rand.html
                    #     bottom_int_indices = torch.floor(chosen_quantiles*10).to(torch.int) # .astype(int)
                    #     top_int_indices = bottom_int_indices+1
                    #     top_int_indices[top_int_indices == 11] = 10
                        
                    #     # Expand batch dimension to match indexing (assuming batch_size = 1)
                    #     batch_indices = torch.zeros((prob_vars.num_particles, prob_vars.state_dim), dtype=torch.long)  # Shape (num_particles, state_dim)
                        
                    #     # pred_bottom_quantiles = predicted_quantiles[:, bottom_int_indices, :]
                    #     # pred_top_quantiles = predicted_quantiles[:, top_int_indices, :]
                    #     pred_bottom_quantiles = predicted_quantiles[batch_indices, bottom_int_indices, torch.arange(prob_vars.state_dim)]
                    #     pred_top_quantiles = predicted_quantiles[batch_indices, top_int_indices, torch.arange(prob_vars.state_dim)]
                        
                    #     # print("chosen_quantiles.shape ", chosen_quantiles.shape, "\n")
                    #     # print("bottom_int_indices.shape ", bottom_int_indices.shape, "\n")
                    #     # print("top_int_indices.shape ", top_int_indices.shape, "\n")
                    #     # print("pred_bottom_quantiles.shape ", pred_bottom_quantiles.shape, "\n")
                    #     # print("pred_top_quantiles.shape ", pred_top_quantiles.shape, "\n")
                        
                    #     # print("chosen_quantiles ", chosen_quantiles, "\n")
                    #     # print("bottom_int_indices ", bottom_int_indices, "\n")
                    #     # print("top_int_indices ", top_int_indices, "\n")
                    #     # print("pred_bottom_quantiles ", pred_bottom_quantiles, "\n")
                    #     # print("pred_top_quantiles ", pred_top_quantiles, "\n")
                        
                    #     next_states = (top_int_indices-10*chosen_quantiles)*pred_bottom_quantiles+(10*chosen_quantiles-bottom_int_indices)*(pred_top_quantiles) 
                    #     # '''

                        
                    #     # print("next_states ", next_states, "\n")

                
                    # if use_mid:
                    #     mid_quantile = predicted_quantiles[:, predicted_quantiles.size(1) // 2, :]
                    #     next_states = mid_quantile
                    """
                    
                    next_states = model_state(sim_states, actions)
                    
                    # action_mu, action_sigma = model_ASN(torch.tensor(state, dtype=torch.float32), torch.tensor(goal_state, dtype=torch.float32))
                    action_probs = model_ASN(next_states, torch.tensor(prob_vars.goal_state, dtype=torch.float32))
                    
                    # print("action_probs ", action_probs, "\n")
                    # print("action_probs[loop_iter].detach().numpy() ", action_probs[0].detach().numpy(), "\n")
                    
                    # particles[loop_iter, j] = np.random.choice(nb_actions, num_particles, p=action_probs[loop_iter].detach().numpy())

                    for loop_iter in range(prob_vars.num_particles):
                        # print("np.random.choice(nb_actions, num_particles, p=action_probs[loop_iter].detach().numpy()) ", np.random.choice(nb_actions, 1, p=action_probs[loop_iter].detach().numpy()), "\n")
                        particles[loop_iter, j] = np.random.choice(prob_vars.nb_actions, 1, p=action_probs[loop_iter].detach().numpy())
        
                
                # particles = np.random.randint(0, 2, (num_particles, horizon))

            # elif prob == "Acrobot": # Discrete actions
                # particles = np.random.randint(0, 3, (num_particles, horizon))
            
            
            else: # Continuous actions # Pendulum, MountainCarContinuous, PandaReacher
                action_mu, action_sigma = model_ASN(torch.tensor(state, dtype=torch.float32), torch.tensor(prob_vars.goal_state, dtype=torch.float32))
                # print("num_particles ", num_particles, "\n")
                # print("action_mu ", action_mu, "\n")
                # print("action_sigma ", action_sigma, "\n")
                # print("len(particles) ", len(particles), "\n")
                # print("action_mu.shape ", action_mu.shape, "\n")
                # print("action_sigma.shape ", action_sigma.shape, "\n")
                
                # print("action_mu.detach().numpy()[0] ", action_mu.detach().numpy()[0], "\n")
                
                if prob_vars.prob == "PandaReacher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "LunarLanderContinuous" or prob_vars.prob == "PandaReacherDense":
                    for loop_iter in range(prob_vars.num_particles):
                        # print("np.random.normal(action_mu.detach().numpy()[0], action_sigma.detach().numpy()[0]) ", np.random.normal(action_mu.detach().numpy()[0], action_sigma.detach().numpy()[0]), "\n")
                        particles[loop_iter, :prob_vars.action_dim] = np.random.normal(action_mu.detach().numpy()[0], action_sigma.detach().numpy()[0])
                
                    for j in range(1, prob_vars.horizon):
                    # for j in range(horizon):
                        sim_states = torch.tensor(state, dtype=torch.float32).repeat(len(particles), 1)
                        # actions = torch.tensor(particles[:, j], dtype=torch.float32).reshape(len(particles),1)
                        actions_array = particles[:, j * prob_vars.action_dim : (j + 1) * prob_vars.action_dim]
                        actions = torch.tensor([actions_array], dtype=torch.float32).reshape(len(particles), prob_vars.action_dim)
                        
                        # # Predict next states using the quantile model_QRNN
                        # predicted_quantiles = model_QRNN(sim_states, actions)
                        # mid_quantile = predicted_quantiles[:, predicted_quantiles.size(1) // 2, :]
                        # next_states = mid_quantile

                        # if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher":
                        #     # Clip states to ensure they are within the valid range
                        #     # before inputting them to the model (sorta like normalization)
                        sim_states = torch.clip(sim_states, prob_vars.states_low, prob_vars.states_high)
                        # sim_states = 2 * ((sim_states - prob_vars.states_low) / (prob_vars.states_high - prob_vars.states_low)) - 1
                        actions = torch.clip(actions, prob_vars.action_low, prob_vars.action_high)

                        # # Predict next states using the quantile model
                        # predicted_quantiles = model_QRNN(sim_states, actions)
                        # # print("predicted_quantiles ", predicted_quantiles, "\n")

                        # if use_sampling:
                        #     # '''
                        #     ############# Need to add the sorta quantile ponderated sum here #############
                        #     # chosen_quantiles = np.random.uniform(0, 1, (num_particles, state_dim))
                        #     chosen_quantiles = torch.rand(prob_vars.num_particles, prob_vars.state_dim)
                        #     # https://pytorch.org/docs/main/generated/torch.rand.html
                        #     bottom_int_indices = torch.floor(chosen_quantiles*10).to(torch.int) # .astype(int)
                        #     top_int_indices = bottom_int_indices+1
                        #     top_int_indices[top_int_indices == 11] = 10
                            
                        #     # Expand batch dimension to match indexing (assuming batch_size = 1)
                        #     batch_indices = torch.zeros((prob_vars.num_particles, prob_vars.state_dim), dtype=torch.long)  # Shape (num_particles, state_dim)
                            
                        #     # pred_bottom_quantiles = predicted_quantiles[:, bottom_int_indices, :]
                        #     # pred_top_quantiles = predicted_quantiles[:, top_int_indices, :]
                        #     pred_bottom_quantiles = predicted_quantiles[batch_indices, bottom_int_indices, torch.arange(prob_vars.state_dim)]
                        #     pred_top_quantiles = predicted_quantiles[batch_indices, top_int_indices, torch.arange(prob_vars.state_dim)]
                            
                        #     # print("chosen_quantiles.shape ", chosen_quantiles.shape, "\n")
                        #     # print("bottom_int_indices.shape ", bottom_int_indices.shape, "\n")
                        #     # print("top_int_indices.shape ", top_int_indices.shape, "\n")
                        #     # print("pred_bottom_quantiles.shape ", pred_bottom_quantiles.shape, "\n")
                        #     # print("pred_top_quantiles.shape ", pred_top_quantiles.shape, "\n")
                            
                        #     # print("chosen_quantiles ", chosen_quantiles, "\n")
                        #     # print("bottom_int_indices ", bottom_int_indices, "\n")
                        #     # print("top_int_indices ", top_int_indices, "\n")
                        #     # print("pred_bottom_quantiles ", pred_bottom_quantiles, "\n")
                        #     # print("pred_top_quantiles ", pred_top_quantiles, "\n")
                            
                        #     next_states = (top_int_indices-10*chosen_quantiles)*pred_bottom_quantiles+(10*chosen_quantiles-bottom_int_indices)*(pred_top_quantiles) 
                        #     # '''

                            
                        #     # print("next_states ", next_states, "\n")

                        # if use_mid:
                        #     mid_quantile = predicted_quantiles[:, predicted_quantiles.size(1) // 2, :]
                        #     next_states = mid_quantile
                        
                        next_states = model_state(sim_states, actions)
                        
                        # action_mu, action_sigma = model_ASN(torch.tensor(state, dtype=torch.float32), torch.tensor(goal_state, dtype=torch.float32))
                        action_mu, action_sigma = model_ASN(next_states, torch.tensor(prob_vars.goal_state, dtype=torch.float32))
                        
                        # print("action_mu.shape ", action_mu.shape, "\n")
                        # print("action_sigma.shape ", action_sigma.shape, "\n")
                        # print("action_mu[0] ", action_mu[0], "\n")
                        # print("action_sigma[0] ", action_sigma[0], "\n")

                        for loop_iter in range(prob_vars.num_particles):
                            particles[loop_iter, j * prob_vars.action_dim : (j + 1) * prob_vars.action_dim] = np.random.normal(action_mu.detach().numpy()[loop_iter], action_sigma.detach().numpy()[loop_iter])
                
                else: # Pendulum, MountainCarContinuous, Pendulum_xyomega
                    for loop_iter in range(prob_vars.num_particles):
                        # print("np.random.normal(action_mu.detach().numpy()[0], action_sigma.detach().numpy()[0]) ", np.random.normal(action_mu.detach().numpy()[0], action_sigma.detach().numpy()[0]), "\n")
                        particles[loop_iter, 0] = np.random.normal(action_mu.detach().numpy()[0], action_sigma.detach().numpy()[0])
                
                    # for j in range(1, horizon):
                    for j in range(prob_vars.horizon):
                        sim_states = torch.tensor(state, dtype=torch.float32).repeat(len(particles), 1)
                        actions = torch.tensor(particles[:, j], dtype=torch.float32).reshape(len(particles),1)

                        # if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher":
                        #     # Clip states to ensure they are within the valid range
                        #     # before inputting them to the model (sorta like normalization)
                        sim_states = torch.clip(sim_states, prob_vars.states_low, prob_vars.states_high)
                        # sim_states = 2 * ((sim_states - prob_vars.states_low) / (prob_vars.states_high - prob_vars.states_low)) - 1
                        actions = torch.clip(actions, prob_vars.action_low, prob_vars.action_high)
                        
                        # # Predict next states using the quantile model_QRNN
                        # predicted_quantiles = model_QRNN(sim_states, actions)
                        # # mid_quantile = predicted_quantiles[:, predicted_quantiles.size(1) // 2, :]
                        # # next_states = mid_quantile

                        # if use_sampling:
                        #     # '''
                        #     ############# Need to add the sorta quantile ponderated sum here #############
                        #     # chosen_quantiles = np.random.uniform(0, 1, (num_particles, state_dim))
                        #     chosen_quantiles = torch.rand(prob_vars.num_particles, prob_vars.state_dim)
                        #     # https://pytorch.org/docs/main/generated/torch.rand.html
                        #     bottom_int_indices = torch.floor(chosen_quantiles*10).to(torch.int) # .astype(int)
                        #     top_int_indices = bottom_int_indices+1
                        #     top_int_indices[top_int_indices == 11] = 10
                            
                        #     # Expand batch dimension to match indexing (assuming batch_size = 1)
                        #     batch_indices = torch.zeros((prob_vars.num_particles, prob_vars.state_dim), dtype=torch.long)  # Shape (num_particles, state_dim)
                            
                        #     # pred_bottom_quantiles = predicted_quantiles[:, bottom_int_indices, :]
                        #     # pred_top_quantiles = predicted_quantiles[:, top_int_indices, :]
                        #     pred_bottom_quantiles = predicted_quantiles[batch_indices, bottom_int_indices, torch.arange(prob_vars.state_dim)]
                        #     pred_top_quantiles = predicted_quantiles[batch_indices, top_int_indices, torch.arange(prob_vars.state_dim)]
                            
                        #     # print("chosen_quantiles.shape ", chosen_quantiles.shape, "\n")
                        #     # print("bottom_int_indices.shape ", bottom_int_indices.shape, "\n")
                        #     # print("top_int_indices.shape ", top_int_indices.shape, "\n")
                        #     # print("pred_bottom_quantiles.shape ", pred_bottom_quantiles.shape, "\n")
                        #     # print("pred_top_quantiles.shape ", pred_top_quantiles.shape, "\n")
                            
                        #     # print("chosen_quantiles ", chosen_quantiles, "\n")
                        #     # print("bottom_int_indices ", bottom_int_indices, "\n")
                        #     # print("top_int_indices ", top_int_indices, "\n")
                        #     # print("pred_bottom_quantiles ", pred_bottom_quantiles, "\n")
                        #     # print("pred_top_quantiles ", pred_top_quantiles, "\n")
                            
                        #     next_states = (top_int_indices-10*chosen_quantiles)*pred_bottom_quantiles+(10*chosen_quantiles-bottom_int_indices)*(pred_top_quantiles) 
                        #     # '''

                            
                        #     # print("next_states ", next_states, "\n")

                    
                        # if use_mid:
                        #     mid_quantile = predicted_quantiles[:, predicted_quantiles.size(1) // 2, :]
                        #     next_states = mid_quantile
                        
                        next_states = model_state(sim_states, actions)

                        # action_mu, action_sigma = model_ASN(torch.tensor(state, dtype=torch.float32), torch.tensor(goal_state, dtype=torch.float32))
                        action_mu, action_sigma = model_ASN(next_states, torch.tensor(prob_vars.goal_state, dtype=torch.float32))
                        
                        # print("action_mu.shape ", action_mu.shape, "\n")
                        # print("action_sigma.shape ", action_sigma.shape, "\n")
                        # print("action_mu[0] ", action_mu[0], "\n")
                        # print("action_sigma[0] ", action_sigma[0], "\n")

                        for loop_iter in range(prob_vars.num_particles):
                            particles[loop_iter, j] = np.random.normal(action_mu.detach().numpy()[loop_iter], action_sigma.detach().numpy()[loop_iter])
        
                # particles = np.random.uniform(action_low, action_high, (num_quantiles, horizon))
        
        # particles = np.zeros((num_particles, horizon))
        # particles[0] = np.array([-1.982811505902002, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, -1.0812550657808897, 2.0, 2.0, 2.0, 0.6938272212396055, 2.0])
        
        # for step in tqdm(range(max_steps)):
        for step in range(prob_vars.max_steps):
            # Get the current state
            # """ Need to change this!!!!!!!!!!!!!!!!! """
            # state = env.state
            # print("step ", step, "\n")
            if prob_vars.prob == "Pendulum":
                state = env.state.copy()
            
            # Choose the best action sequence
            # if step == 0:
            #     best_particle, particles, cost = particle_filtering_cheating(init_particles, env, state, horizon, nb_reps=5, using_Env=usingEnv, episode=episode, step=step)
            # elif step >= 1:
            #     best_particle, particles, cost = particle_filtering_cheating(particles, env, state, horizon, nb_reps=5, using_Env=usingEnv, episode=episode, step=step)
            
            # particles = np.random.uniform(-2, 2, (num_particles, horizon))
            
            # print("state ", state, "\n")
            
            particles = np.clip(particles, prob_vars.action_low, prob_vars.action_high)

            best_particle, action, best_cost, particles = choose_action_func_50NN_MSENN(prob_vars, state, particles, do_RS, use_sampling, use_mid, use_ASGNN, model_state, model_ASN, episode=episode, step=step, goal_state=prob_vars.goal_state)
            # best_particle, particles, cost = particle_filtering_cheating(particles, env, state, horizon, nb_reps=5, using_Env=usingEnv, episode=episode, step=step)
            
            costs.append(best_cost)
            
            # print("best_particle ", best_particle, "\n")
            if prob_vars.prob == "Pendulum" or prob_vars.prob == "MountainCarContinuous" or prob_vars.prob == "Pendulum_xyomega" or prob_vars.prob == "InvertedPendulum" or prob_vars.prob == "CartPoleContinuous":
                action = [best_particle[0]]
                actions_list.append(list(action))
            
            elif prob_vars.prob == "PandaReacher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "LunarLanderContinuous" or prob_vars.prob == "PandaReacherDense":
                action = best_particle[:prob_vars.action_dim]
                # actions_list.append(list(action))
            
            elif prob_vars.prob == "CartPole" or prob_vars.prob == "Acrobot" or prob_vars.prob == "MountainCar" or prob_vars.prob == "LunarLander":
                action = int(action)
                actions_list.append(action)
            
            # actions_list.append(action)
            # actions_list.append(list(action))
            
            # print("env.step(env.action_space.sample()) ", env.step(env.action_space.sample()), "\n")
            # print("action.shape ", action.shape, "\n")
            # print("env.action_space.sample().shape ", env.action_space.sample().shape, "\n")
            
            # Apply the first action from the optimized sequence
            next_state, reward, done, terminated, info = env.step(action)
            
            episode_reward += reward
            # actions_list.append(action)
            # actions_list.append(list(action))
            
            # Apply the first action from the optimized sequence
            # next_state, reward, done, terminated, info = env.step(action)
            # episode_reward += reward
            if prob_vars.prob == "Pendulum":
                # state = env.state.copy()
                next_state = env.state.copy()

            if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "PandaReacherDense":
                prob_vars.goal_state = next_state['desired_goal'] # 3 components
                next_state = next_state['observation']#[:3] # 6 components for Reacher, 18 components for Pusher
                # is_success_bool = info['is_success']
                
            if prob_vars.prob == "MuJoCoReacher":
                next_state = np.array([next_state[0], next_state[1], next_state[2], next_state[3], next_state[6], next_state[7], next_state[8], next_state[9]])
            
            # print("state ", state, "next_state ", next_state, "\n")
            # print("states[0] ", state[0], "states[1] ", state[1], "\n")
            
            # episodic_step_rewards.append(episode_reward)
            # episodic_step_reward_values.append(reward)
            
            # next_state = env.state.copy()
            # Store experience in replay buffer
            # print("state ", state, "\n")
            
            # if prob != "CartPole" and prob != "Acrobot":
            #     replay_buffer_QRNN.append((state, action, reward, next_state, done))
            # else:
            #     replay_buffer_QRNN.append((state, np.array([action]), reward, next_state, terminated))
            if prob_vars.prob == "CartPole" or prob_vars.prob == "Acrobot" or prob_vars.prob == "MountainCar" or prob_vars.prob == "LunarLander":
                replay_buffer_state.append((state, np.array([action]), reward, next_state, terminated))
                replay_buffer_ASN.push(state, prob_vars.goal_state, np.array([action]))
            else:
                replay_buffer_state.append((state, action, reward, next_state, terminated))
                replay_buffer_ASN.push(state, prob_vars.goal_state, action)
            
            if len (replay_buffer_ASN) < 32:
                pass
            else:
                loss = train_ActionSequenceNN(model_ASN, replay_buffer_ASN, prob_vars.batch_size, optimizer_ASN, num_epochs=1)
            
            if len(replay_buffer_state) < prob_vars.batch_size:
                pass
            else:
                batch = random.sample(replay_buffer_state, prob_vars.batch_size)
                states, actions_train, rewards, next_states, dones = zip(*batch)
                # print("batch states ", states, "\n")
                states = torch.tensor(states, dtype=torch.float32)
                actions_tensor = torch.tensor(actions_train, dtype=torch.float32)
                # print("actions.shape ", actions_tensor, "\n")
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)

                # if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher":
                #     # Clip states to ensure they are within the valid range
                #     # before inputting them to the model (sorta like normalization)
                states = torch.clip(states, prob_vars.states_low, prob_vars.states_high)
                # states = 2 * ((states - prob_vars.states_low) / (prob_vars.states_high - prob_vars.states_low)) - 1
                actions_tensor = torch.clip(actions_tensor, prob_vars.action_low, prob_vars.action_high)
                
                # Predict next state quantiles
                # predicted_quantiles = model_QRNN(states, actions_tensor)  # Shape: (batch_size, num_quantiles, state_dim)
                predicted_next_states = model_state(states, actions_tensor)
                
                # Use next state as target (can be improved with target policy)
                # target_quantiles = next_states
                
                # Compute the target quantiles (e.g., replicate next state across the quantile dimension)
                # target_quantiles = next_states.unsqueeze(-1).repeat(1, 1, num_quantiles)

                # Compute Quantile Huber Loss
                # loss = quantile_loss(predicted_quantiles, target_quantiles, prob_vars.quantiles)
                loss = loss_state(predicted_next_states, next_states)
                
                # # Compute Quantile Huber Loss
                # loss = quantile_loss(predicted_quantiles, target_quantiles, prob_vars.quantiles)
                
                # Optimize the model_QRNN
                optimizer_state.zero_grad()
                loss.backward()
                optimizer_state.step()
            
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
            # The last column will be determined in MPC
            
            # if prob == "CartPole" or prob == "Acrobot":
            #     action_probs = model_ASN(torch.tensor(state, dtype=torch.float32), torch.tensor(goal_state, dtype=torch.float32))[0]
            #     particles[:, -1] = np.random.choice(nb_actions, num_particles, p=action_probs.detach().numpy())
            
            # else: # Pendulum, MountainCarContinuous, PandaReacher, MuJoCoReacher
            #     action_mu, action_sigma = model_ASN(torch.tensor(state, dtype=torch.float32), torch.tensor(goal_state, dtype=torch.float32))
            #     for loop_iter in range(num_particles):
            #         # print("np.random.normal(action_mu.detach().numpy()[0], action_sigma.detach().numpy()[0]) ", np.random.normal(action_mu.detach().numpy()[0], action_sigma.detach().numpy()[0]), "\n")
            #         particles[loop_iter, :action_dim] = np.random.normal(action_mu.detach().numpy()[0], action_sigma.detach().numpy()[0])
            
            # particles = np.clip(particles, action_low, action_high)
            
        # print("best_particle ", best_particle, "\n")
        # print("actions ", actions, "\n")
        # print('horizon: %d, episode: %d, reward: %d' % (horizon, episode, episode_reward))
        # episode_reward_list.append(episode_reward)
        # episode_reward_list_MPC_PF_QRNN_WithActionSequenceGeneratorNN.append(episode_reward)
        
        # episode_reward_list_withASGNN[episode] = episode_reward
        episode_reward_list_withASGNN.append(episode_reward)

        episode_success_rate_withASGNN.append(nb_episode_success/(episode+1)) # Episodic success rate for Panda Gym envs
        # episode_success_rate_withASGNN.append(nb_episode_success) # /max_steps # Episodic success rate for Panda Gym envs
        
        # print(f'episode: {episode}, reward: {episode_reward}')
        # # episode_reward_list.append(episode_reward)
        # print("actions_list ", actions_list, "\n")
        
        """ Print stuff """
        # if prob == 'PandaReacher':
        #     print(f'episode: {episode}, reward: {episode_reward}')
        #     print("nb_episode_success ", nb_episode_success, "\n")
        #     print("np.linalg.norm(goal_state-state) ", np.linalg.norm(goal_state-next_state[:3]), "\n")
        #     print("episode", episode, "actions_list ", actions_list, "\n")
            
        # if prob == "MuJoCoReacher":
        #     print("np.linalg.norm(goal_state-state)=np.sqrt(next_state[-2]**2+next_state[-1]**2) ", np.sqrt(next_state[-2]**2+next_state[-1]**2), "\n")

    # if use_sampling:
    #     # Assuming `agent` is your RL model and `optimizer` is the optimizer
    #     torch.save({
    #         'model_state_dict': model_QRNN.state_dict(),
    #         'optimizer_state_dict': optimizer_QRNN.state_dict(),
    #     }, f"model_QRNN_{prob}_sampling_{change_prob}.pth")
    #     torch.save({
    #         'model_state_dict': model_ASN.state_dict(),
    #         'optimizer_state_dict': optimizer_ASN.state_dict(),
    #     }, "model_ASN_{prob}_mid_{change_prob}.pth")
    # if use_mid:
    #     torch.save({
    #         'model_state_dict': model_QRNN.state_dict(),
    #         'optimizer_state_dict': optimizer_QRNN.state_dict(),
    #     }, f"model_QRNN_{prob}_mid_{change_prob}.pth")
    #     torch.save({
    #         'model_state_dict': model_ASN.state_dict(),
    #         'optimizer_state_dict': optimizer_ASN.state_dict(),
    #     }, "model_ASN_{prob}_mid_{change_prob}.pth")

    if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "PandaReacherDense":
        return episode_reward_list_withASGNN, episode_success_rate_withASGNN
    else:
        return episode_reward_list_withASGNN



def start_50NN_MSENNrand_RS(prob_vars, env, seed, model_state, replay_buffer_state, optimizer_state, loss_state, model_ASN, replay_buffer_ASN, optimizer_ASN, do_RS, do_QRNN_step_rnd, use_sampling, use_mid, use_ASGNN):

    episode_reward_list = []
    episode_success_rate = [] # For Panda Gym envs

    # goal_state = prob_vars.goal_state

    nb_episode_success = 0 # For Panda Gym envs

    # if prob == "PandaReacher" or prob == "PandaPusher":
    #     states_low = torch.tensor(-10)
    #     states_high = torch.tensor(10)

    # if prob == "MuJoCoReacher":
    #     states_low = torch.tensor([-1, -1, -1, -1, -100, -100, -torch.inf, -torch.inf])
    #     states_high = torch.tensor([1, 1, 1, 1, 100, 100, torch.inf, torch.inf])

    # print("seed ", seed, "\n")
    # print("env ", env, "\n")

    # for episode in range(tqdm(max_episodes)):
    for episode in range(prob_vars.max_episodes):
        state, _ = env.reset(seed=seed)
        episode_reward = 0
        done = False
        actions_list = []
        if prob_vars.prob == "Pendulum":
            state = env.state.copy()
        if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "PandaReacherDense":
            prob_vars.goal_state = state['desired_goal'] # 3 components
            state = state['observation']#[:3] # 6 components for Reacher, 18 components for Pusher
        if prob_vars.prob == "MuJoCoReacher":
            prob_vars.goal_state = np.array([state[4], state[5]])
            state = np.array([state[0], state[1], state[2], state[3], state[6], state[7], state[8], state[9]])
        if prob_vars.prob == "MuJoCoPusher":
            prob_vars.goal_state = np.array([state[20], state[21], state[22]])
            
        
        costs = []
        episodic_step_rewards = []
        episodic_step_reward_values = []
        
        # if episode == 0:
        #     # Random initial action sequence
        #     # initial_action_sequence = np.random.randint(0, 2, horizon)
        #     init_particles = [np.random.uniform(-2, 2, horizon) for _ in range(num_particles)] # 2*horizon_list[0]
        # else:
        #     # Add small random noise to encourage exploration (for now this can stay the same)
        #     # particles = np.clip(best_action_sequence + np.random.randint(0, 2, horizon), 0, 1)
        #     for i in range(len(particles)):
        #         # print("best_particle ", best_particle, "\n")
        #         # print("np.random.uniform(-0.5, 0.5, horizon) ", np.random.uniform(-0.5, 0.5, horizon), "\n")
        #         particles[i] = np.clip(best_particle + np.random.uniform(-0.5, 0.5, horizon), -2, 2)
        #         # particles = np.clip(init_particles + np.random.randint(0, 2, len(init_particles)), 0, 1)
        
        # particles = [np.random.uniform(-2, 2, horizon) for _ in range(num_particles)] # 2*horizon_list[0]
        
        if prob_vars.prob == "CartPole":
            particles = np.random.randint(0, 2, (prob_vars.num_particles, prob_vars.horizon))
        elif prob_vars.prob == "Acrobot" or prob_vars.prob == "MountainCar": 
            particles = np.random.randint(0, 3, (prob_vars.num_particles, prob_vars.horizon))
        elif prob_vars.prob == "LunarLander":
            particles = np.random.randint(0, 4, (prob_vars.num_particles, prob_vars.horizon))
        elif prob_vars.prob == "PandaReacher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "LunarLanderContinuous" or prob_vars.prob == "PandaReacherDense":
            particles = np.random.uniform(prob_vars.action_low, prob_vars.action_high, (prob_vars.num_particles, prob_vars.action_dim*prob_vars.horizon))
        else: # Pendulum, MountainCarContinuous
            particles = np.random.uniform(prob_vars.action_low, prob_vars.action_high, (prob_vars.num_particles, prob_vars.horizon))
        
        # particles = np.zeros((num_particles, horizon))
        # particles[0] = np.array([-1.982811505902002, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, -1.0812550657808897, 2.0, 2.0, 2.0, 0.6938272212396055, 2.0])
        
        # for step in range(tqdm(max_steps)):
        for step in range(prob_vars.max_steps):
            # Get the current state
            # """ Need to change this!!!!!!!!!!!!!!!!! """
            # state = env.state
            # print("step ", step, "\n")
            if prob_vars.prob == "Pendulum":
                state = env.state.copy()
            
            # Choose the best action sequence
            # if step == 0:
            #     best_particle, particles, cost = particle_filtering_cheating(init_particles, env, state, horizon, nb_reps=5, using_Env=usingEnv, episode=episode, step=step)
            # elif step >= 1:
            #     best_particle, particles, cost = particle_filtering_cheating(particles, env, state, horizon, nb_reps=5, using_Env=usingEnv, episode=episode, step=step)
            
            # particles = np.random.uniform(-2, 2, (num_particles, horizon))
            
            # print("state ", state, "\n")
            
            # print("len(particles) ", len(particles), "\n")
            
            if do_RS or do_QRNN_step_rnd:
                if prob_vars.prob == "CartPole":
                    particles = np.random.randint(0, 2, (prob_vars.num_particles, prob_vars.horizon))
                elif prob_vars.prob == "Acrobot" or prob_vars.prob == "MountainCar":
                    particles = np.random.randint(0, 3, (prob_vars.num_particles, prob_vars.horizon))
                elif prob_vars.prob == "LunarLander":
                    particles = np.random.randint(0, 4, (prob_vars.num_particles, prob_vars.horizon))
                elif prob_vars.prob == "PandaReacher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "LunarLanderContinuous" or prob_vars.prob == "PandaReacherDense":
                    particles = np.random.uniform(prob_vars.action_low, prob_vars.action_high, (prob_vars.num_particles, prob_vars.action_dim*prob_vars.horizon))
                else: # Pendulum, MountainCarContinuous
                    particles = np.random.uniform(prob_vars.action_low, prob_vars.action_high, (prob_vars.num_particles, prob_vars.horizon))
            # best_particle, action, best_cost, particles = choose_action(prob, state, do_RS, use_sampling, use_mid, use_ASGNN, horizon, particles, model_QRNN, action_low, action_high, nb_reps_MPC, std, change_prob, nb_top_particles, nb_random, episode=episode, step=step, goal_state=goal_state)
            ##

            particles = np.clip(particles, prob_vars.action_low, prob_vars.action_high)

            best_particle, action, best_cost, particles = choose_action_func_50NN_MSENN(prob_vars, state, particles, do_RS, use_sampling, use_mid, use_ASGNN, model_state, model_ASN, episode=episode, step=step, goal_state=prob_vars.goal_state)
            # best_particle, action, best_cost, particles = choose_action(prob_vars.prob, state, horizon, particles, do_RS, use_sampling, use_mid, use_ASGNN, model_QRNN, model_ASN, action_dim, action_low, action_high, states_low, states_high, nb_reps_MPC, std, change_prob, nb_top_particles, nb_random, episode=episode, step=step, goal_state=goal_state)
            
            # best_particle, particles, cost = particle_filtering_cheating(particles, env, state, horizon, nb_reps=5, using_Env=usingEnv, episode=episode, step=step)
            
            # print("action ", action, "\n")
            
            # print("best_particle ", best_particle, "\n")
            # if prob != "CartPole" and prob != "Acrobot" and prob != "PandaReacher" and prob != "MuJoCoReacher":
            #     action = [best_particle[0]]
            
            actions_list.append(action)
            
            costs.append(best_cost)
            
            if prob_vars.prob == "Pendulum" or prob_vars.prob == "MountainCarContinuous" or prob_vars.prob == "Pendulum_xyomega" or prob_vars.prob == "InvertedPendulum" or prob_vars.prob == "CartPoleContinuous":
                action = [best_particle[0]]
                # print("action ", action, "\n")
            
            elif prob_vars.prob == "PandaReacher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "LunarLanderContinuous" or prob_vars.prob == "PandaReacherDense":
                action = best_particle[:prob_vars.action_dim]
            
            elif prob_vars.prob == "CartPole" or prob_vars.prob == "Acrobot" or prob_vars.prob == "MountainCar" or prob_vars.prob == "LunarLander":
                action = int(action)
            
            # if prob == "CartPole" or prob == "Acrobot":
            #     action = int(action)
            
            # Apply the first action from the optimized sequence
            next_state, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            actions_list.append(action)
            
            # Apply the first action from the optimized sequence
            # next_state, reward, done, terminated, info = env.step(action)
            # episode_reward += reward
            if prob_vars.prob == "Pendulum":
                # state = env.state.copy()
                next_state = env.state.copy()
            if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "PandaReacherDense":
                prob_vars.goal_state = next_state['desired_goal'] # 3 components
                next_state = next_state['observation']#[:3] # 6 components
            if prob_vars.prob == "MuJoCoReacher":
                next_state = np.array([next_state[0], next_state[1], next_state[2], next_state[3], next_state[6], next_state[7], next_state[8], next_state[9]])
                
            # print("state ", state, "next_state ", next_state, "\n")
            # print("states[0] ", state[0], "states[1] ", state[1], "\n")
            
            # episodic_step_rewards.append(episode_reward)
            # episodic_step_reward_values.append(reward)
            
            # next_state = env.state.copy()
            # Store experience in replay buffer
            # print("state ", state, "\n")
            
            # if prob != "CartPole" and prob != "Acrobot":
            #     replay_buffer.append((state, action, reward, next_state, done))
            # else:
            #     replay_buffer.append((state, np.array([action]), reward, next_state, terminated))
            if prob_vars.prob == "CartPole" or prob_vars.prob == "Acrobot" or prob_vars.prob == "MountainCar" or prob_vars.prob == "LunarLander":
                replay_buffer_state.append((state, np.array([action]), reward, next_state, truncated))
            else:
                replay_buffer_state.append((state, action, reward, next_state, truncated))
            
                
            if len(replay_buffer_state) < prob_vars.batch_size:
                pass
            else:
                batch = random.sample(replay_buffer_state, prob_vars.batch_size)
                states, actions_train, rewards, next_states, dones = zip(*batch)
                # print("batch states ", states, "\n")
                states = torch.tensor(states, dtype=torch.float32)
                actions_tensor = torch.tensor(actions_train, dtype=torch.float32)
                # print("actions.shape ", actions_tensor, "\n")
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)

                # if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher":
                #     # Clip states to ensure they are within the valid range
                #     # before inputting them to the model (sorta like normalization)
                states = torch.clip(states, prob_vars.states_low, prob_vars.states_high)
                # states = 2 * ((states - prob_vars.states_low) / (prob_vars.states_high - prob_vars.states_low)) - 1
                actions_tensor = torch.clip(actions_tensor, prob_vars.action_low, prob_vars.action_high)
                
                # Predict next state quantiles
                # predicted_quantiles = model_QRNN(states, actions_tensor)  # Shape: (batch_size, num_quantiles, state_dim)
                
                # Use next state as target (can be improved with target policy)
                # target_quantiles = next_states
                
                predicted_next_states = model_state(states, actions_tensor)
                
                # Compute the target quantiles (e.g., replicate next state across the quantile dimension)
                # target_quantiles = next_states.unsqueeze(-1).repeat(1, 1, num_quantiles)

                # Compute Quantile Huber Loss
                # loss = quantile_loss(predicted_quantiles, target_quantiles, prob_vars.quantiles)
                loss = loss_state(predicted_next_states, next_states)
                
                # # Compute Quantile Huber Loss
                # loss = quantile_loss(predicted_quantiles, target_quantiles, quantiles)
                
                # Optimize the model
                optimizer_state.zero_grad()
                loss.backward()
                optimizer_state.step()
            
            # if prob == "MuJoCoReacher":
            #     if np.sqrt(next_state[-2]**2+next_state[-1]**2) < 0.05:
            #         print("Reached target position \n")
            #         done = True
            
            done = done or truncated
            if done:
                nb_episode_success += 1
                break
            
            if not do_RS or not do_QRNN_step_rnd:
                if prob_vars.prob == "CartPole":
                    # Shift all particles to the left by removing the first element
                    particles[:, :-1] = particles[:, 1:]
                
                    # Generate new random values (0 or 1) for the last column
                    new_values = np.random.randint(0, 2, size=(particles.shape[0], 1))
                    
                    # Add the new values to the last position
                    particles[:, -1:] = new_values
                    
                elif prob_vars.prob == "Acrobot" or prob_vars.prob == "MountainCar":
                    # Shift all particles to the left by removing the first element
                    particles[:, :-1] = particles[:, 1:]
                    
                    # Generate new random values (0 or 1) for the last column
                    new_values = np.random.randint(0, 3, size=(particles.shape[0], 1))
                    
                    # Add the new values to the last position
                    particles[:, -1:] = new_values

                elif prob_vars.prob == "LunarLander":
                    # Shift all particles to the left by removing the first element
                    particles[:, :-1] = particles[:, 1:]
                    
                    # Generate new random values (0 or 1) for the last column
                    new_values = np.random.randint(0, 4, size=(particles.shape[0], 1))
                    
                    # Add the new values to the last position
                    particles[:, -1:] = new_values
                
                elif prob_vars.prob == "PandaReacher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "LunarLanderContinuous" or prob_vars.prob == "PandaReacherDense":
                    # Shift all particles to the left by removing the first element
                    particles[:, :-prob_vars.action_dim] = particles[:, prob_vars.action_dim:]
                    
                    # Generate new random values for the last column
                    new_values = np.random.uniform(prob_vars.action_low, prob_vars.action_high, size=(particles.shape[0], prob_vars.action_dim))
                    
                    # Add the new values to the last position
                    particles[:, -prob_vars.action_dim:] = new_values  
                
                else: # Pendulum, MountainCarContinuous, Pendulum_xyomega
                    # Shift all particles to the left by removing the first element
                    particles[:, :-1] = particles[:, 1:]
                    
                    # Generate new random values for the last column
                    new_values = np.random.uniform(prob_vars.action_low, prob_vars.action_high, size=(particles.shape[0], 1))
                    
                    # Add the new values to the last position
                    particles[:, -1:] = new_values
            particles = np.clip(particles, prob_vars.action_low, prob_vars.action_high)
            
            # if step == 0:
            #     # print("len(init_particles) ", len(init_particles), "\n")
            #     # particles = np.copy(init_particles)
            #     for i in range(len(particles)):
            #         # particles[i] = np.clip(top_particles[0][1:] + [np.random.randint(0, 2)], 0, 1)
            #         particles[i] = np.clip(np.append(particles[i][1:],[np.random.uniform(-2, 2)]), -2, 2)
            #         # print("particles[i] ", particles[i].shape, "\n")    
            # else:
            #     # print("len(particles) ", len(particles), "\n")
            #     particles[0] = best_particle
            #     for i in range(1, len(particles)):
            #         # particles[i] = np.clip(top_particles[0][1:] + [np.random.randint(0, 2)], 0, 1)
            #         particles[i] = np.clip(np.append(particles[i][1:],[np.random.uniform(-2, 2)]), -2, 2)
            #         # print("particles[i] ", particles[i].shape, "\n")
            
            # state = env.state.copy() # next_state
            
            state = next_state
        
        # print("best_particle ", best_particle, "\n")
        # print("actions ", actions, "\n")
        # print('horizon: %d, episode: %d, reward: %d' % (horizon, episode, episode_reward))
        episode_reward_list.append(episode_reward)

        episode_success_rate.append(nb_episode_success/(episode+1)) # Episodic success rate for Panda Gym envs
        # episode_success_rate.append(nb_episode_success) # /max_steps # Episodic success rate for Panda Gym envs     
        
        # print("actions_list ", actions_list, "\n")
        
        # print(f'episode: {episode}, reward: {episode_reward}')
        # # episode_reward_list.append(episode_reward)
        # print("actions_list ", actions_list, "\n")
        
        ''' Print stuff '''
        # if prob == 'PandaReacher':
        #     print("np.linalg.norm(goal_state-state) ", np.linalg.norm(goal_state-next_state[:3]), "\n")
        #     print("actions_list ", actions_list, "\n")
        # if prob == "MuJoCoReacher":
        #     print("np.linalg.norm(goal_state-state)=np.sqrt(next_state[-2]**2+next_state[-1]**2) ", np.sqrt(next_state[-2]**2+next_state[-1]**2), "\n")

    # if use_sampling:
    #     if do_RS:
    #         # Assuming `agent` is your RL model and `optimizer` is the optimizer
    #         torch.save({
    #             'model_state_dict': model_QRNN.state_dict(),
    #             'optimizer_state_dict': optimizer_QRNN.state_dict(),
    #         }, f"RS_{prob_vars.prob}_sampling_{prob_vars.change_prob}.pth")
    #     elif do_QRNN_step_rnd:
    #         # Assuming `agent` is your RL model and `optimizer` is the optimizer
    #         torch.save({
    #             'model_state_dict': model_QRNN.state_dict(),
    #             'optimizer_state_dict': optimizer_QRNN.state_dict(),
    #         }, f"QRNN_step_rnd_{prob_vars.prob}_sampling_{prob_vars.change_prob}.pth")
    #     else:
    #         # Assuming `agent` is your RL model and `optimizer` is the optimizer
    #         torch.save({
    #             'model_state_dict': model_QRNN.state_dict(),
    #             'optimizer_state_dict': optimizer_QRNN.state_dict(),
    #         }, f"QRNN_basic_{prob_vars.prob}_sampling_{prob_vars.change_prob}.pth")
    # if use_mid:
    #     if do_RS:
    #         # Assuming `agent` is your RL model and `optimizer` is the optimizer
    #         torch.save({
    #             'model_state_dict': model_QRNN.state_dict(),
    #             'optimizer_state_dict': optimizer_QRNN.state_dict(),
    #         }, f"RS_{prob_vars.prob}_mid_{prob_vars.change_prob}.pth")
    #     elif do_QRNN_step_rnd:
    #         # Assuming `agent` is your RL model and `optimizer` is the optimizer
    #         torch.save({
    #             'model_state_dict': model_QRNN.state_dict(),
    #             'optimizer_state_dict': optimizer_QRNN.state_dict(),
    #         }, f"QRNN_step_rnd_{prob_vars.prob}_mid_{prob_vars.change_prob}.pth")
    #     else:
    #         # Assuming `agent` is your RL model and `optimizer` is the optimizer
    #         torch.save({
    #             'model_state_dict': model_QRNN.state_dict(),
    #             'optimizer_state_dict': optimizer_QRNN.state_dict(),
    #         }, f"QRNN_basic_{prob_vars.prob}_mid_{prob_vars.change_prob}.pth")

    # return episode_reward_list
    if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "PandaReacherDense":
        return episode_reward_list, episode_success_rate
    else:
        return episode_reward_list
    

def start_50NN_MSENN_MPC_LBFGSB(prob_vars, env, seed, model_state, replay_buffer_state, optimizer_state, loss_state, use_sampling, use_mid):
    
    episode_reward_list = []
    episode_success_rate = [] # For Panda Gym envs

    # goal_state = prob_vars.goal_state

    nb_episode_success = 0 # For Panda Gym envs

    # if prob == "PandaReacher" or prob == "PandaPusher":
    #     states_low = torch.tensor(-10)
    #     states_high = torch.tensor(10)

    # if prob == "MuJoCoReacher":
    #     states_low = torch.tensor([-1, -1, -1, -1, -100, -100, -torch.inf, -torch.inf])
    #     states_high = torch.tensor([1, 1, 1, 1, 100, 100, torch.inf, torch.inf])

    # print("seed ", seed, "\n")
    # print("env ", env, "\n")

    # for episode in range(tqdm(max_episodes)):
    for episode in range(prob_vars.max_episodes):
        state, _ = env.reset(seed=seed)
        episode_reward = 0
        done = False
        actions_list = []
        if prob_vars.prob == "Pendulum":
            state = env.state.copy()
        if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher":
            prob_vars.goal_state = state['desired_goal'] # 3 components
            state = state['observation']#[:3] # 6 components for Reacher, 18 components for Pusher
        if prob_vars.prob == "MuJoCoReacher":
            prob_vars.goal_state = np.array([state[4], state[5]])
            state = np.array([state[0], state[1], state[2], state[3], state[6], state[7], state[8], state[9]])
        if prob_vars.prob == "MuJoCoPusher":
            prob_vars.goal_state = np.array([state[20], state[21], state[22]])
            
        if episode <= 0:
            if prob_vars.prob == "CartPole":
                actions = np.random.randint(0, 2, (prob_vars.num_particles, prob_vars.horizon))
            elif prob_vars.prob == "Acrobot" or prob_vars.prob == "MountainCar": 
                actions = np.random.randint(0, 3, (prob_vars.num_particles, prob_vars.horizon))
            elif prob_vars.prob == "LunarLander":
                actions = np.random.randint(0, 4, (prob_vars.num_particles, prob_vars.horizon))
            elif prob_vars.prob == "PandaReacher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "LunarLanderContinuous" or prob_vars.prob == "PandaReacherDense":
                actions = np.random.uniform(prob_vars.action_low, prob_vars.action_high, (prob_vars.num_particles, prob_vars.action_dim*prob_vars.horizon))
            else: # Pendulum, MountainCarContinuous
                actions = np.random.uniform(prob_vars.action_low, prob_vars.action_high, (prob_vars.num_particles, prob_vars.horizon))
        
   
        
        # particles = np.zeros((num_particles, horizon))
        # particles[0] = np.array([-1.982811505902002, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, -1.0812550657808897, 2.0, 2.0, 2.0, 0.6938272212396055, 2.0])
        
        # for step in range(tqdm(max_steps)):
        for step in range(prob_vars.max_steps):
            # Get the current state
            # """ Need to change this!!!!!!!!!!!!!!!!! """
            # state = env.state
            # print("step ", step, "\n")
            if prob_vars.prob == "Pendulum":
                state = env.state.copy()
            
            best_cost = np.inf
            best_action_sequence = None
            for rep in range(prob_vars.nb_reps_MPC):
                iter_rep = minimize(fun=choose_action_func_50NN_MSENN_LBFGSB,
                        x0 = actions,
                        jac=True,
                        args=(prob_vars, state, use_sampling, use_mid, model_state),
                        method='L-BFGS-B',
                        bounds= [(prob_vars.action_low, prob_vars.action_high)],
                        options= {
                            "disp": None, "maxcor": 8, "ftol": 1e-18, "gtol": 1e-18, "eps": 1e-2, "maxfun": 8,
                            "maxiter": 8, "iprint": -1, "maxls": 8, "finite_diff_rel_step": None
                            },
                )
                actions_rep = iter_rep.x
                cost_iter = iter_rep.fun
                if cost_iter < best_cost:
                    best_cost = cost_iter
                    best_action_sequence = actions_rep

                # actions, grads = choose_action_LBFGSB(actions, prob, state, horizon, do_RS, use_sampling, use_mid, use_ASGNN, model_QRNN, model_ASN, action_dim, action_low, action_high, states_low, states_high, nb_reps, std, change_prob, nb_top_particles, nb_random)
                        
            # action = actions[0] #.detach().numpy()
            action = best_action_sequence[0]
            # print("action ", action, "\n")
            
            if action is None:
                print("actions_rep ", actions_rep, '\n')
                print("cost_iter ", cost_iter, "\n")
            
            if prob_vars.prob == "Pendulum" or prob_vars.prob == "MountainCarContinuous" or prob_vars.prob == "Pendulum_xyomega" or prob_vars.prob == "CartPoleContinuous":
                # action = [best_particle[0]]
                action = [action]
            #     # print("action ", action, "\n")
            
            # elif prob_vars.prob == "PandaReacher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "LunarLanderContinuous":
            #     action = best_particle[:prob_vars.action_dim]
            
            elif prob_vars.prob == "CartPole" or prob_vars.prob == "Acrobot" or prob_vars.prob == "MountainCar" or prob_vars.prob == "LunarLander":
                action = int(action)
            
            # if prob == "CartPole" or prob == "Acrobot":
            #     action = int(action)
            
            # Apply the first action from the optimized sequence
            next_state, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            actions_list.append(action)
            
            # Apply the first action from the optimized sequence
            # next_state, reward, done, terminated, info = env.step(action)
            # episode_reward += reward
            if prob_vars.prob == "Pendulum":
                # state = env.state.copy()
                next_state = env.state.copy()
            if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher":
                prob_vars.goal_state = next_state['desired_goal'] # 3 components
                next_state = next_state['observation']#[:3] # 6 components
            if prob_vars.prob == "MuJoCoReacher":
                next_state = np.array([next_state[0], next_state[1], next_state[2], next_state[3], next_state[6], next_state[7], next_state[8], next_state[9]])
                
            # print("state ", state, "next_state ", next_state, "\n")
            # print("states[0] ", state[0], "states[1] ", state[1], "\n")
            
            # episodic_step_rewards.append(episode_reward)
            # episodic_step_reward_values.append(reward)
            
            # next_state = env.state.copy()
            # Store experience in replay buffer
            # print("state ", state, "\n")
            
            # if prob != "CartPole" and prob != "Acrobot":
            #     replay_buffer.append((state, action, reward, next_state, done))
            # else:
            #     replay_buffer.append((state, np.array([action]), reward, next_state, terminated))
            if prob_vars.prob == "CartPole" or prob_vars.prob == "Acrobot" or prob_vars.prob == "MountainCar" or prob_vars.prob == "LunarLander":
                replay_buffer_state.append((state, np.array([action]), reward, next_state, truncated))
            else:
                replay_buffer_state.append((state, action, reward, next_state, truncated))
            
                
            if len(replay_buffer_state) < prob_vars.batch_size:
                pass
            else:
                batch = random.sample(replay_buffer_state, prob_vars.batch_size)
                states, actions_train, rewards, next_states, dones = zip(*batch)
                # print("batch states ", states, "\n")
                states = torch.tensor(states, dtype=torch.float32)
                actions_tensor = torch.tensor(actions_train, dtype=torch.float32)
                # print("actions.shape ", actions_tensor, "\n")
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)

                # if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher":
                #     # Clip states to ensure they are within the valid range
                #     # before inputting them to the model (sorta like normalization)
                states = torch.clip(states, prob_vars.states_low, prob_vars.states_high)
                # states = 2 * ((states - prob_vars.states_low) / (prob_vars.states_high - prob_vars.states_low)) - 1
                actions_tensor = torch.clip(actions_tensor, prob_vars.action_low, prob_vars.action_high)
                
                # Predict next state quantiles
                # predicted_quantiles = model_QRNN(states, actions_tensor)  # Shape: (batch_size, num_quantiles, state_dim)
                
                # Use next state as target (can be improved with target policy)
                # target_quantiles = next_states
                
                predicted_next_states = model_state(states, actions_tensor)
                
                # Compute the target quantiles (e.g., replicate next state across the quantile dimension)
                # target_quantiles = next_states.unsqueeze(-1).repeat(1, 1, num_quantiles)

                # Compute Quantile Huber Loss
                # loss = quantile_loss(predicted_quantiles, target_quantiles, prob_vars.quantiles)
                loss = loss_state(predicted_next_states, next_states)
                
                # # Compute Quantile Huber Loss
                # loss = quantile_loss(predicted_quantiles, target_quantiles, quantiles)
                
                # Optimize the model
                optimizer_state.zero_grad()
                loss.backward()
                optimizer_state.step()
            
            # if prob == "MuJoCoReacher":
            #     if np.sqrt(next_state[-2]**2+next_state[-1]**2) < 0.05:
            #         print("Reached target position \n")
            #         done = True
            
            done = done or truncated
            if done:
                nb_episode_success += 1
                break
            
            state = next_state
        
        # print("best_particle ", best_particle, "\n")
        # print("actions ", actions, "\n")
        # print('horizon: %d, episode: %d, reward: %d' % (horizon, episode, episode_reward))
        episode_reward_list.append(episode_reward)

        episode_success_rate.append(nb_episode_success/(episode+1)) # Episodic success rate for Panda Gym envs
        # episode_success_rate.append(nb_episode_success) # /max_steps # Episodic success rate for Panda Gym envs     
        
        # print("actions_list ", actions_list, "\n")

    if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "PandaReacherDense":
        return episode_reward_list, episode_success_rate
    else:
        return episode_reward_list
    


