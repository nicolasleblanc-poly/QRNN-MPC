import numpy as np
import torch
from state_pred_models import get_mid_quantile
import copy

def GenerateParticlesUsingASNN_func_QRNN(prob_vars, state, particles, model_QRNN, model_ASNN, use_sampling, use_mid):
    if prob_vars.prob == "CartPole" or prob_vars.prob == "Acrobot" or prob_vars.prob == "MountainCar" or prob_vars.prob == "LunarLander": # Discrete actions
        action_probs = model_ASNN(torch.tensor(state, dtype=torch.float32), torch.tensor(prob_vars.goal_state, dtype=torch.float32))[0]
        
        particles[:, 0] = np.random.choice(prob_vars.nb_actions, prob_vars.num_particles, p=action_probs.detach().numpy())
        
        sim_states = torch.tensor(state, dtype=torch.float32).repeat(len(particles), 1)
        for j in range(prob_vars.horizon-1):
            actions = torch.tensor(particles[:, j], dtype=torch.float32).reshape(len(particles),1)
            
            sim_states = torch.clip(sim_states, prob_vars.states_low, prob_vars.states_high)

            actions = torch.clip(actions, prob_vars.action_low, prob_vars.action_high)

            # Predict next states using the quantile model_QRNN
            predicted_quantiles = model_QRNN(sim_states, actions)

            if use_sampling:
                # '''
                ############# Need to add the sorta quantile ponderated sum here #############
                # chosen_quantiles = np.random.uniform(0, 1, (num_particles, state_dim))
                chosen_quantiles = torch.rand(prob_vars.num_particles, prob_vars.state_dim)
                # https://pytorch.org/docs/main/generated/torch.rand.html
                bottom_int_indices = torch.floor(chosen_quantiles*10).to(torch.int) # .astype(int)
                top_int_indices = bottom_int_indices+1
                top_int_indices[top_int_indices == 11] = 10
                
                # Expand batch dimension to match indexing (assuming batch_size = 1)
                batch_indices = torch.zeros((prob_vars.num_particles, prob_vars.state_dim), dtype=torch.long)  # Shape (num_particles, state_dim)

                pred_bottom_quantiles = predicted_quantiles[batch_indices, bottom_int_indices, torch.arange(prob_vars.state_dim)]
                pred_top_quantiles = predicted_quantiles[batch_indices, top_int_indices, torch.arange(prob_vars.state_dim)]
                
                next_states = (top_int_indices-10*chosen_quantiles)*pred_bottom_quantiles+(10*chosen_quantiles-bottom_int_indices)*(pred_top_quantiles) 

            if use_mid:
                mid_quantile = predicted_quantiles[:, predicted_quantiles.size(1) // 2, :]
                next_states = mid_quantile
                
            
            action_probs = model_ASNN(next_states, torch.tensor(prob_vars.goal_state, dtype=torch.float32))


            for loop_iter in range(prob_vars.num_particles):
                particles[loop_iter, j+1] = np.random.choice(prob_vars.nb_actions, 1, p=action_probs[loop_iter].detach().numpy())

            sim_states = next_states

    else: # Continuous actions # Pendulum, MountainCarContinuous, PandaReacher
        action_mu, action_sigma = model_ASNN(torch.tensor(state, dtype=torch.float32), torch.tensor(prob_vars.goal_state, dtype=torch.float32))
        
        if prob_vars.prob == "PandaReacher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "LunarLanderContinuous" or prob_vars.prob == "PandaReacherDense" or prob_vars.prob == "PandaPusherDense":
            for loop_iter in range(prob_vars.num_particles):
                particles[loop_iter, :prob_vars.action_dim] = np.random.normal(action_mu.detach().numpy()[0], action_sigma.detach().numpy()[0])
        
            
            sim_states = torch.tensor(state, dtype=torch.float32).repeat(len(particles), 1)
            for j in range(prob_vars.horizon-1):
                actions_array = particles[:, j * prob_vars.action_dim : (j + 1) * prob_vars.action_dim]
                actions = torch.tensor([actions_array], dtype=torch.float32).reshape(len(particles), prob_vars.action_dim)

                sim_states = torch.clip(sim_states, prob_vars.states_low, prob_vars.states_high)

                actions = torch.clip(actions, prob_vars.action_low, prob_vars.action_high)

                # Predict next states using the quantile model
                predicted_quantiles = model_QRNN(sim_states, actions)

                if use_sampling:
                    # '''
                    ############# Need to add the sorta quantile ponderated sum here #############
                    # chosen_quantiles = np.random.uniform(0, 1, (num_particles, state_dim))
                    chosen_quantiles = torch.rand(prob_vars.num_particles, prob_vars.state_dim)
                    # https://pytorch.org/docs/main/generated/torch.rand.html
                    bottom_int_indices = torch.floor(chosen_quantiles*10).to(torch.int) # .astype(int)
                    top_int_indices = bottom_int_indices+1
                    top_int_indices[top_int_indices == 11] = 10
                    
                    # Expand batch dimension to match indexing (assuming batch_size = 1)
                    batch_indices = torch.zeros((prob_vars.num_particles, prob_vars.state_dim), dtype=torch.long)  # Shape (num_particles, state_dim)
                    
                    # pred_bottom_quantiles = predicted_quantiles[:, bottom_int_indices, :]
                    # pred_top_quantiles = predicted_quantiles[:, top_int_indices, :]
                    pred_bottom_quantiles = predicted_quantiles[batch_indices, bottom_int_indices, torch.arange(prob_vars.state_dim)]
                    pred_top_quantiles = predicted_quantiles[batch_indices, top_int_indices, torch.arange(prob_vars.state_dim)]
                    
                    next_states = (top_int_indices-10*chosen_quantiles)*pred_bottom_quantiles+(10*chosen_quantiles-bottom_int_indices)*(pred_top_quantiles) 

                if use_mid:
                    mid_quantile = predicted_quantiles[:, predicted_quantiles.size(1) // 2, :]
                    next_states = mid_quantile
                
                # action_mu, action_sigma = model_ASNN(torch.tensor(state, dtype=torch.float32), torch.tensor(goal_state, dtype=torch.float32))
                action_mu, action_sigma = model_ASNN(next_states, torch.tensor(prob_vars.goal_state, dtype=torch.float32))

                for loop_iter in range(prob_vars.num_particles):
                    particles[loop_iter, j * prob_vars.action_dim : (j + 1) * prob_vars.action_dim] = np.random.normal(action_mu.detach().numpy()[loop_iter], action_sigma.detach().numpy()[loop_iter])
        
                sim_states = next_states
        
        else: # Pendulum, MountainCarContinuous, Pendulum_xyomega
            for loop_iter in range(prob_vars.num_particles):
                particles[loop_iter, 0] = np.random.normal(action_mu.detach().numpy()[0], action_sigma.detach().numpy()[0])
        
            sim_states = torch.tensor(state, dtype=torch.float32).repeat(len(particles), 1)
            for j in range(prob_vars.horizon-1):
                actions = torch.tensor(particles[:, j], dtype=torch.float32).reshape(len(particles),1)

                sim_states = torch.clip(sim_states, prob_vars.states_low, prob_vars.states_high)

                actions = torch.clip(actions, prob_vars.action_low, prob_vars.action_high)
                
                # Predict next states using the quantile model_QRNN
                predicted_quantiles = model_QRNN(sim_states, actions)

                if use_sampling:
                    # '''
                    ############# Need to add the sorta quantile ponderated sum here #############
                    # chosen_quantiles = np.random.uniform(0, 1, (num_particles, state_dim))
                    chosen_quantiles = torch.rand(prob_vars.num_particles, prob_vars.state_dim)
                    # https://pytorch.org/docs/main/generated/torch.rand.html
                    bottom_int_indices = torch.floor(chosen_quantiles*10).to(torch.int) # .astype(int)
                    top_int_indices = bottom_int_indices+1
                    top_int_indices[top_int_indices == 11] = 10
                    
                    # Expand batch dimension to match indexing (assuming batch_size = 1)
                    batch_indices = torch.zeros((prob_vars.num_particles, prob_vars.state_dim), dtype=torch.long)  # Shape (num_particles, state_dim)

                    pred_bottom_quantiles = predicted_quantiles[batch_indices, bottom_int_indices, torch.arange(prob_vars.state_dim)]
                    pred_top_quantiles = predicted_quantiles[batch_indices, top_int_indices, torch.arange(prob_vars.state_dim)]

                    next_states = (top_int_indices-10*chosen_quantiles)*pred_bottom_quantiles+(10*chosen_quantiles-bottom_int_indices)*(pred_top_quantiles) 

            
                if use_mid:
                    mid_quantile = predicted_quantiles[:, predicted_quantiles.size(1) // 2, :]
                    next_states = mid_quantile
                
                
                # action_mu, action_sigma = model_ASNN(torch.tensor(state, dtype=torch.float32), torch.tensor(goal_state, dtype=torch.float32))
                action_mu, action_sigma = model_ASNN(next_states, torch.tensor(prob_vars.goal_state, dtype=torch.float32))
                

                for loop_iter in range(prob_vars.num_particles):
                    particles[loop_iter, j] = np.random.normal(action_mu.detach().numpy()[loop_iter], action_sigma.detach().numpy()[loop_iter])

                sim_states = next_states
    
    
    
    
    return particles


def GenerateParticlesUsingASNN_func_50NN_MSENN(prob_vars, state, particles, model_state, model_ASNN):
    
    if prob_vars.prob == "CartPole" or prob_vars.prob == "Acrobot" or prob_vars.prob == "MountainCar" or prob_vars.prob == "LunarLander": # Discrete actions
        action_probs = model_ASNN(torch.tensor(state, dtype=torch.float32), torch.tensor(prob_vars.goal_state, dtype=torch.float32))[0]
        
        particles[:, 0] = np.random.choice(prob_vars.nb_actions, prob_vars.num_particles, p=action_probs.detach().numpy())
        
        sim_states = torch.tensor(state, dtype=torch.float32).repeat(len(particles), 1)
        for j in range(prob_vars.horizon-1):
            actions = torch.tensor(particles[:, j], dtype=torch.float32).reshape(len(particles),1)
            
            sim_states = torch.clip(sim_states, prob_vars.states_low, prob_vars.states_high)
            actions = torch.clip(actions, prob_vars.action_low, prob_vars.action_high)

            next_states = model_state(sim_states, actions)
            
            action_probs = model_ASNN(next_states, torch.tensor(prob_vars.goal_state, dtype=torch.float32))

            for loop_iter in range(prob_vars.num_particles):
                particles[loop_iter, j+1] = np.random.choice(prob_vars.nb_actions, 1, p=action_probs[loop_iter].detach().numpy())

            sim_states = next_states
    
    else: # Continuous actions # Pendulum, MountainCarContinuous, PandaReacher
        action_mu, action_sigma = model_ASNN(torch.tensor(state, dtype=torch.float32), torch.tensor(prob_vars.goal_state, dtype=torch.float32))
        
        if prob_vars.prob == "PandaReacher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "LunarLanderContinuous" or prob_vars.prob == "PandaReacherDense" or prob_vars.prob == "PandaPusherDense":
            for loop_iter in range(prob_vars.num_particles):
                particles[loop_iter, :prob_vars.action_dim] = np.random.normal(action_mu.detach().numpy()[0], action_sigma.detach().numpy()[0])
        
            sim_states = torch.tensor(state, dtype=torch.float32).repeat(len(particles), 1)
            for j in range(prob_vars.horizon-1):
                
                actions_array = particles[:, j * prob_vars.action_dim : (j + 1) * prob_vars.action_dim]
                actions = torch.tensor([actions_array], dtype=torch.float32).reshape(len(particles), prob_vars.action_dim)
                
                sim_states = torch.clip(sim_states, prob_vars.states_low, prob_vars.states_high)
                actions = torch.clip(actions, prob_vars.action_low, prob_vars.action_high)
                
                next_states = model_state(sim_states, actions)
                
                action_mu, action_sigma = model_ASNN(next_states, torch.tensor(prob_vars.goal_state, dtype=torch.float32))

                for loop_iter in range(prob_vars.num_particles):
                    particles[loop_iter, (j+1) * prob_vars.action_dim : (j + 2) * prob_vars.action_dim] = np.random.normal(action_mu.detach().numpy()[loop_iter], action_sigma.detach().numpy()[loop_iter])
        
                sim_states = next_states#.cpu().numpy()
        
        else: # Pendulum, MountainCarContinuous, Pendulum_xyomega
            for loop_iter in range(prob_vars.num_particles):
                particles[loop_iter, 0] = np.random.normal(action_mu.detach().numpy()[0], action_sigma.detach().numpy()[0])
        
            sim_states = torch.tensor(state, dtype=torch.float32).repeat(len(particles), 1)
            for j in range(prob_vars.horizon-1):
                actions = torch.tensor(particles[:, j], dtype=torch.float32).reshape(len(particles),1)

                sim_states = torch.clip(sim_states, prob_vars.states_low, prob_vars.states_high)
                actions = torch.clip(actions, prob_vars.action_low, prob_vars.action_high)
                
                next_states = model_state(sim_states, actions)

                action_mu, action_sigma = model_ASNN(next_states, torch.tensor(prob_vars.goal_state, dtype=torch.float32))
            
                for loop_iter in range(prob_vars.num_particles):
                    particles[loop_iter, j+1] = np.random.normal(action_mu.detach().numpy()[loop_iter], action_sigma.detach().numpy()[loop_iter])

                sim_states = next_states

    return particles

def GenerateParticlesUsingASNN_func_UsingEnv(prob_vars, state, particles, model_ASNN):
    
    if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "PandaReacherDense" or prob_vars.prob == "PandaPusherDense":
        og_state_id = prob_vars.env.save_state()
        
        state_ids = np.array([prob_vars.env.save_state() for _ in range(prob_vars.num_particles)], dtype=object)
        state_ids[:] = prob_vars.env.save_state()  # Save the initial state of the environment
    
    
    if prob_vars.prob == "CartPole" or prob_vars.prob == "Acrobot" or prob_vars.prob == "MountainCar" or prob_vars.prob == "LunarLander": # Discrete actions
        action_probs = model_ASNN(torch.tensor(state, dtype=torch.float32), torch.tensor(prob_vars.goal_state, dtype=torch.float32))[0]
        
        particles[:, 0] = np.random.choice(prob_vars.nb_actions, prob_vars.num_particles, p=action_probs.detach().numpy())
        
        sim_states = torch.tensor(state, dtype=torch.float32).repeat(len(particles), 1)
        for j in range(prob_vars.horizon-1):
            # actions = torch.tensor(particles[:, j], dtype=torch.float32).reshape(len(particles),1)
            
            sim_states = torch.clip(sim_states, prob_vars.states_low, prob_vars.states_high)
            # actions = torch.clip(actions, prob_vars.action_low, prob_vars.action_high)
            actions = particles[:, j].reshape(len(particles), prob_vars.action_dim)
            actions = np.clip(actions, prob_vars.action_low, prob_vars.action_high)
        
            next_states = []
            for i in range(prob_vars.num_particles):
            
                if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "PandaReacherDense" or prob_vars.prob == "PandaPusherDense":
                    prob_vars.env.restore_state(state_ids[i])
                    prob_vars.env.remove_state(state_ids[i])
                    
                    next_state_step, reward, terminated, truncated, info = prob_vars.env.step(actions[i].numpy())
                    
                    state_ids[i] = prob_vars.env.save_state()
                    next_states.append(next_state_step['observation'])
                    
                elif prob_vars.prob == "MountainCar" or prob_vars.prob == "MountainCarContinuous" or prob_vars.prob == "CartPole" or prob_vars.prob == "CartPoleContinuous":
                    # if j >0:
                    #     env_i = copy.deepcopy(prob_vars.env)
                    #     env_i.reset() # seed=prob_vars.seed
                    #     env_i.state = sim_states[i].numpy()  # Set the initial state of the environment
                    # else: # if h == 0:
                    #     env_i = copy.deepcopy(prob_vars.env)
                    #     env_i.reset() # seed=prob_vars.seed
                    #     if isinstance(sim_states, torch.Tensor):
                    #         sim_states.detach().cpu().numpy()
                    #     env_i.state = sim_states # .numpy() # Set the initial state of the environment
                        
                    # next_state_step, reward, terminated, truncated, info = env_i.step(actions[i]) # .numpy()
                    
                    # next_state = env_i.state
                    # next_states.append(next_state)
                    
                    env_i = copy.deepcopy(prob_vars.env)
                    env_i.reset() # seed=prob_vars.seed
                    if isinstance(sim_states, torch.Tensor):
                        sim_states.detach().cpu().numpy()
                    env_i.state = sim_states[i] # .numpy() # Set the initial state of the environment
                    
                    # print("env_i.state ", env_i.state, "\n")
                    
                    if prob_vars.discrete == True:
                        next_state_step, reward, terminated, truncated, info = env_i.step(int(actions[i]))
                    else:
                        next_state_step, reward, terminated, truncated, info = env_i.step(actions[i]) # .numpy() 
                    
                    next_state = env_i.state
                    next_states.append(next_state)
                    
                else:
                    print("Env isn't supported use the env as a model of the env for MPC")
                
            next_states = torch.tensor(next_states, dtype=torch.float32)

            # next_states = model_state(sim_states, actions)
            
            action_probs = model_ASNN(next_states, torch.tensor(prob_vars.goal_state, dtype=torch.float32))

            for loop_iter in range(prob_vars.num_particles):
                particles[loop_iter, j+1] = np.random.choice(prob_vars.nb_actions, 1, p=action_probs[loop_iter].detach().numpy())

            sim_states = next_states
    
    else: # Continuous actions # Pendulum, MountainCarContinuous, PandaReacher
        action_mu, action_sigma = model_ASNN(torch.tensor(state, dtype=torch.float32), torch.tensor(prob_vars.goal_state, dtype=torch.float32))
        
        if prob_vars.prob == "PandaReacher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "LunarLanderContinuous" or prob_vars.prob == "PandaReacherDense" or prob_vars.prob == "PandaPusherDense":
            for loop_iter in range(prob_vars.num_particles):
                particles[loop_iter, :prob_vars.action_dim] = np.random.normal(action_mu.detach().numpy()[0], action_sigma.detach().numpy()[0])
        
            sim_states = torch.tensor(state, dtype=torch.float32).repeat(len(particles), 1)
            for j in range(prob_vars.horizon-1):
                
                actions_array = particles[:, j * prob_vars.action_dim : (j + 1) * prob_vars.action_dim]
                actions = torch.tensor([actions_array], dtype=torch.float32).reshape(len(particles), prob_vars.action_dim)
                
                # sim_states = torch.clip(sim_states, prob_vars.states_low, prob_vars.states_high)
                # actions = torch.clip(actions, prob_vars.action_low, prob_vars.action_high)
                
                # next_states = model_state(sim_states, actions)
                
                next_states = []
                for i in range(prob_vars.num_particles):
                
                    if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "PandaReacherDense" or prob_vars.prob == "PandaPusherDense":
                        prob_vars.env.restore_state(state_ids[i])
                        prob_vars.env.remove_state(state_ids[i])
                        
                        next_state_step, reward, terminated, truncated, info = prob_vars.env.step(actions[i].numpy())
                        
                        state_ids[i] = prob_vars.env.save_state()
                        next_states.append(next_state_step['observation'])
                        
                    elif prob_vars.prob == "MountainCar" or prob_vars.prob == "MountainCarContinuous" or prob_vars.prob == "CartPole" or prob_vars.prob == "CartPoleContinuous":
                        if j >0:
                            env_i = copy.deepcopy(prob_vars.env)
                            env_i.reset() # seed=prob_vars.seed
                            env_i.state = sim_states[i].numpy()  # Set the initial state of the environment
                        else: # if h == 0:
                            env_i = copy.deepcopy(prob_vars.env)
                            env_i.reset() # seed=prob_vars.seed
                            if isinstance(sim_states, torch.Tensor):
                                sim_states.detach().cpu().numpy()
                            env_i.state = sim_states # .numpy() # Set the initial state of the environment
                            
                        next_state_step, reward, terminated, truncated, info = env_i.step(actions[i].numpy())
                        
                        next_state = env_i.state
                        next_states.append(next_state)
                        
                    else:
                        print("Env isn't supported use the env as a model of the env for MPC")
                    
                next_states = torch.tensor(next_states, dtype=torch.float32)
                
                action_mu, action_sigma = model_ASNN(next_states, torch.tensor(prob_vars.goal_state, dtype=torch.float32))

                for loop_iter in range(prob_vars.num_particles):
                    particles[loop_iter, (j+1) * prob_vars.action_dim : (j + 2) * prob_vars.action_dim] = np.random.normal(action_mu.detach().numpy()[loop_iter], action_sigma.detach().numpy()[loop_iter])
        
                sim_states = next_states#.cpu().numpy()
        
        else: # Pendulum, MountainCarContinuous, Pendulum_xyomega
            for loop_iter in range(prob_vars.num_particles):
                particles[loop_iter, 0] = np.random.normal(action_mu.detach().numpy()[0], action_sigma.detach().numpy()[0])
        
            sim_states = torch.tensor(state, dtype=torch.float32).repeat(len(particles), 1)
            for j in range(prob_vars.horizon-1):
                actions = torch.tensor(particles[:, j], dtype=torch.float32).reshape(len(particles),1)

                sim_states = torch.clip(sim_states, prob_vars.states_low, prob_vars.states_high)
                actions = torch.clip(actions, prob_vars.action_low, prob_vars.action_high)
                
                # next_states = model_state(sim_states, actions)
                
                next_states = []
                for i in range(prob_vars.num_particles):
                
                    if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "PandaReacherDense" or prob_vars.prob == "PandaPusherDense":
                        prob_vars.env.restore_state(state_ids[i])
                        prob_vars.env.remove_state(state_ids[i])
                        
                        next_state_step, reward, terminated, truncated, info = prob_vars.env.step(actions[i].numpy())
                        
                        state_ids[i] = prob_vars.env.save_state()
                        next_states.append(next_state_step['observation'])
                        
                    elif prob_vars.prob == "MountainCar" or prob_vars.prob == "MountainCarContinuous" or prob_vars.prob == "CartPole" or prob_vars.prob == "CartPoleContinuous":
                        if j >0:
                            env_i = copy.deepcopy(prob_vars.env)
                            env_i.reset() # seed=prob_vars.seed
                            env_i.state = sim_states[i].numpy()  # Set the initial state of the environment
                        else: # if h == 0:
                            env_i = copy.deepcopy(prob_vars.env)
                            env_i.reset() # seed=prob_vars.seed
                            if isinstance(sim_states, torch.Tensor):
                                sim_states.detach().cpu().numpy()
                            env_i.state = sim_states # .numpy() # Set the initial state of the environment
                            
                        next_state_step, reward, terminated, truncated, info = env_i.step(actions[i].numpy())
                        
                        next_state = env_i.state
                        next_states.append(next_state)
                        
                    else:
                        print("Env isn't supported use the env as a model of the env for MPC")
                    
                next_states = torch.tensor(next_states, dtype=torch.float32)

                action_mu, action_sigma = model_ASNN(next_states, torch.tensor(prob_vars.goal_state, dtype=torch.float32))
            
                for loop_iter in range(prob_vars.num_particles):
                    particles[loop_iter, j+1] = np.random.normal(action_mu.detach().numpy()[loop_iter], action_sigma.detach().numpy()[loop_iter])

                sim_states = next_states

    return particles
