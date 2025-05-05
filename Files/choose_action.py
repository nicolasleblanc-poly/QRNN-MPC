import numpy as np
import random
import torch

from mpc import mpc_func
from particle_filtering import particle_filtering_func

def choose_action_func(prob_vars, state, particles, do_RS, use_sampling, use_mid, use_ASGNN, model_QRNN, model_ASN, episode=0, step=1, goal_state=None):

    best_cost = float('inf')
    best_action_sequence = None

    nb_reps = prob_vars.nb_reps_MPC
    # horizon = prob_vars.horizon
    # action_dim = prob_vars.action_dim
    # action_low = prob_vars.action_low
    # action_high = prob_vars.action_high
    # states_low = prob_vars.states_low
    # states_high = prob_vars.states_high
    # std = prob_vars.std
    # change_prob = prob_vars.change_prob
    # nb_top_particles = prob_vars.nb_top_particles
    # nb_random = prob_vars.nb_random

    # if prob == "PandaReacher" or prob == "PandaPusher":
    #     states_low = torch.tensor(-10)
    #     states_high = torch.tensor(10)

    # if prob == "MuJoCoReacher":
    #     states_low = torch.tensor([-1, -1, -1, -1, -100, -100, -torch.inf, -torch.inf])
    #     states_high = torch.tensor([1, 1, 1, 1, 100, 100, torch.inf, torch.inf])

    if do_RS:
        nb_reps = 1

    for rep in range(nb_reps):
    
        # costs, best_action_sequence = objective_function(particles, sim_states, horizon, model)
        # num_particles = particles.shape[0]
        
        
        sim_states = torch.tensor(state, dtype=torch.float32).repeat(prob_vars.num_particles, 1)

        # best_cost = float('inf')
        # best_action_sequence = None

        costs = mpc_func(prob_vars, sim_states, particles, use_ASGNN, model_QRNN, use_sampling, use_mid, model_ASN)

        min_idx = torch.argmin(costs)
        
        # print("costs[min_idx] ", costs[min_idx], "\n")
        # print("best_cost ", best_cost, "\n")
        
        if costs[min_idx] < best_cost:
            best_cost = costs[min_idx]
            best_action_sequence = particles[min_idx].copy()

        if best_cost == None or best_action_sequence is None:
            best_cost = costs[0]
            best_action_sequence = particles[0].copy()
            print("costs ", costs, "\n")
            print("best_action_sequence ", best_action_sequence, "\n")
            print("sim_states ", sim_states, "\n")
            print("state ", state, "\n")
            print("particles ", particles, "\n")
        
        # if prob == "Acrobot" : # or prob == "CartPole"
        #     min_idx = torch.argmin(costs)
        #     if costs[min_idx] < best_cost:
        #         best_cost = costs[min_idx]
        #         best_action_sequence = particles[min_idx].copy()

        # print("particles[0] ", particles[0], "\n")
        # print("best_action_sequence ", best_action_sequence, "\n")

        particles[0] = best_action_sequence

        best_first_action, particles = particle_filtering_func(prob_vars, particles, costs, best_action_sequence)   

    # best_first_action = int(best_action_sequence[0].item())
    return best_action_sequence, best_first_action, best_cost, particles

def choose_action_continuous_cem_func(prob_vars, state, particles, do_RS, use_sampling, use_mid, use_ASGNN, model_QRNN, model_ASN, episode=0, step=1, goal_state=None):

    best_cost = float('inf')
    best_action_sequence = None

    nb_reps = prob_vars.nb_reps_MPC
    # horizon = prob_vars.horizon
    # action_dim = prob_vars.action_dim
    # action_low = prob_vars.action_low
    # action_high = prob_vars.action_high
    # states_low = prob_vars.states_low
    # states_high = prob_vars.states_high
    # std = prob_vars.std
    # change_prob = prob_vars.change_prob
    # nb_top_particles = prob_vars.nb_top_particles
    # nb_random = prob_vars.nb_random

    # if prob == "PandaReacher" or prob == "PandaPusher":
    #     states_low = torch.tensor(-10)
    #     states_high = torch.tensor(10)

    # if prob == "MuJoCoReacher":
    #     states_low = torch.tensor([-1, -1, -1, -1, -100, -100, -torch.inf, -torch.inf])
    #     states_high = torch.tensor([1, 1, 1, 1, 100, 100, torch.inf, torch.inf])

    if do_RS:
        nb_reps = 1

    for rep in range(nb_reps):
    
        # costs, best_action_sequence = objective_function(particles, sim_states, horizon, model)
        # num_particles = particles.shape[0]
        
        
        sim_states = torch.tensor(state, dtype=torch.float32).repeat(prob_vars.num_particles, 1)

        # best_cost = float('inf')
        # best_action_sequence = None

        costs = mpc_func(prob_vars, sim_states, particles, use_ASGNN, model_QRNN, use_sampling, use_mid, model_ASN)

        min_idx = torch.argmin(costs)
        
        # print("costs[min_idx] ", costs[min_idx], "\n")
        # print("best_cost ", best_cost, "\n")
        
        if costs[min_idx] < best_cost:
            best_cost = costs[min_idx]
            best_action_sequence = particles[min_idx].copy()

        if best_cost == None or best_action_sequence is None:
            best_cost = costs[0]
            best_action_sequence = particles[0].copy()
            print("costs ", costs, "\n")
            print("best_action_sequence ", best_action_sequence, "\n")
            print("sim_states ", sim_states, "\n")
            print("state ", state, "\n")
            print("particles ", particles, "\n")
        
        # if prob == "Acrobot" : # or prob == "CartPole"
        #     min_idx = torch.argmin(costs)
        #     if costs[min_idx] < best_cost:
        #         best_cost = costs[min_idx]
        #         best_action_sequence = particles[min_idx].copy()

        # print("particles[0] ", particles[0], "\n")
        # print("best_action_sequence ", best_action_sequence, "\n")

        particles[0] = best_action_sequence

        best_first_action, particles = particle_filtering_func(prob_vars, particles, costs, best_action_sequence)   

    # best_first_action = int(best_action_sequence[0].item())
    return best_action_sequence, best_first_action, best_cost, particles

def choose_action_func_LBFGSB(actions, prob_vars, state, use_sampling, use_mid, model_QRNN, episode=0, step=1, goal_state=None):

    sim_state = torch.tensor(state, dtype=torch.float32)
    cost = 0
    num_particles = 1

    actions_tensor = torch.tensor(actions, dtype=torch.float32)
    actions_tensor.requires_grad = True

    # cost = mpc_func(prob_vars, sim_state, actions, use_ASGNN, model_QRNN, use_sampling, use_mid, use_LBFGSB=True)
    for h in range(prob_vars.horizon):

        if prob_vars.prob == "PandaReacher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "LunarLanderContinuous" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoPusher":
            # action_array = actions[:, h * action_dim : (h + 1) * action_dim]
            action_tensor = actions_tensor[h * prob_vars.action_dim : (h + 1) * prob_vars.action_dim]
        else:
            # action_array = actions[h]
            action_tensor = actions_tensor[h:h+1]
            
        # action_tensor = torch.tensor(action, dtype=torch.float32)
        # # action = torch.tensor([action_array], dtype=torch.float32) # .reshape(1,action_dim)
        # # # action = torch.tensor([actions[]])
        
        # print("actions.shape ", actions.shape, "\n")
        # print("sim_states.shape ", sim_states.shape, "\n")
        
        # if model_QRNN is None:
        #     # Simulate the environment (replace with batch environment step if available)
        #     next_states = torch.tensor(
        #         [env.step(int(a))[0] for a in actions.cpu().numpy()]
        #     )
        # else:

        # if prob == "PandaReacher" or prob == "PandaPusher" or prob == "MuJoCoReacher":
        #     # Clip states to ensure they are within the valid range
        #     # before inputting them to the model (sorta like normalization)
        sim_state = torch.clip(sim_state, prob_vars.states_low, prob_vars.states_high)
        # sim_state = 2 * ((sim_state - prob_vars.states_low) / (prob_vars.states_high - prob_vars.states_low)) - 1
        action_tensor = action_tensor.clip(prob_vars.action_low, prob_vars.action_high)

        # print("sim_state.shape ", sim_state.shape, '\n')
        # print("action_tensor.shape ", action_tensor.shape, '\n')
        # print("sim_state ", sim_state, '\n')
        # print("action_tensor ", action_tensor, '\n')

        # Predict next states using the quantile model
        predicted_quantiles = model_QRNN(sim_state, action_tensor)
        # print("predicted_quantiles ", predicted_quantiles, "\n")

        if use_sampling:
            # '''
            ############# Need to add the sorta quantile ponderated sum here #############
            # chosen_quantiles = np.random.uniform(0, 1, (num_particles, state_dim))
            chosen_quantiles = torch.rand(num_particles, prob_vars.state_dim)
            # https://pytorch.org/docs/main/generated/torch.rand.html
            bottom_int_indices = torch.floor(chosen_quantiles*10).to(torch.int) # .astype(int)
            top_int_indices = bottom_int_indices+1
            top_int_indices[top_int_indices == 11] = 10
            
            # Expand batch dimension to match indexing (assuming batch_size = 1)
            batch_indices = torch.zeros((num_particles, prob_vars.state_dim), dtype=torch.long)  # Shape (num_particles, state_dim)
            
            # pred_bottom_quantiles = predicted_quantiles[:, bottom_int_indices, :]
            # pred_top_quantiles = predicted_quantiles[:, top_int_indices, :]
            pred_bottom_quantiles = predicted_quantiles[batch_indices, bottom_int_indices, torch.arange(prob_vars.state_dim)]
            pred_top_quantiles = predicted_quantiles[batch_indices, top_int_indices, torch.arange(prob_vars.state_dim)]
            
            # print("chosen_quantiles.shape ", chosen_quantiles.shape, "\n")
            # print("bottom_int_indices.shape ", bottom_int_indices.shape, "\n")
            # print("top_int_indices.shape ", top_int_indices.shape, "\n")
            # print("pred_bottom_quantiles.shape ", pred_bottom_quantiles.shape, "\n")
            # print("pred_top_quantiles.shape ", pred_top_quantiles.shape, "\n")
            
            # print("chosen_quantiles ", chosen_quantiles, "\n")
            # print("bottom_int_indices ", bottom_int_indices, "\n")
            # print("top_int_indices ", top_int_indices, "\n")
            # print("pred_bottom_quantiles ", pred_bottom_quantiles, "\n")
            # print("pred_top_quantiles ", pred_top_quantiles, "\n")
            
            next_states = (top_int_indices-10*chosen_quantiles)*pred_bottom_quantiles+(10*chosen_quantiles-bottom_int_indices)*(pred_top_quantiles) 
            # '''

            
            # print("next_states ", next_states, "\n")

    
        if use_mid:
                mid_quantile = predicted_quantiles[:, predicted_quantiles.size(1) // 2, :]
                next_state = mid_quantile

        # Update state and accumulate cost
        # print("next_state.shape ", next_state.shape, "\n")
        # print("next_state ", next_state, "\n")
        sim_state = next_state.reshape(next_state.shape[1])
        if prob_vars.prob == "PandaReacher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoPusher":
            cost += prob_vars.compute_cost(prob_vars.prob, next_state, h, prob_vars.horizon, action_tensor, prob_vars.goal_state)
            # print("costs ", costs, "\n")
            # print("next_states ", next_states, "\n")
            
        else:
            # print("type(next_state)", type(next_state), "\n")
            # print("type(action)", type(action_tensor), "\n")
            cost += prob_vars.compute_cost(prob_vars.prob, next_state, h, prob_vars.horizon, action_tensor)

    grad = torch.autograd.grad(cost, actions_tensor, retain_graph=False)[0]
    print("grad ", grad, "\n")
    

    return cost.item(), grad.detach().numpy()
