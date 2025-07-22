import numpy as np
import torch
from state_pred_models import get_mid_quantile

def mpc_QRNN_func(prob_vars, sim_states, particles, use_ASGNN, model_QRNN, use_sampling, use_mid, model_ASN=None): # , use_LBFGSB=False
    horizon = prob_vars.horizon
    num_particles = prob_vars.num_particles
    action_dim = prob_vars.action_dim
    nb_actions = prob_vars.nb_actions

    costs = torch.zeros(prob_vars.num_particles)
    
    for h in range(horizon):
        
        if use_ASGNN and h == horizon-1: # Replace last action column with ASGNN predictions
            # Use ASGNN to generate last action/actions based of last non-terminal state
            if prob_vars.prob == "CartPole" or prob_vars.prob == "Acrobot" or prob_vars.prob == "LunarLander" or prob_vars.prob == "MountainCar": # Discrete actions
                action_probs = model_ASN(sim_states, torch.tensor(prob_vars.goal_state, dtype=torch.float32))

                for loop_iter in range(num_particles):
                    particles[loop_iter, -1] = np.random.choice(nb_actions, 1, p=action_probs[loop_iter].detach().numpy())

            else: # Continuous actions
                action_mus, action_sigmas = model_ASN(torch.tensor(sim_states, dtype=torch.float32), torch.tensor(prob_vars.goal_state, dtype=torch.float32))

                for loop_iter in range(num_particles):
                    particles[loop_iter, -action_dim:] = np.random.normal(action_mus.detach().numpy()[loop_iter], action_sigmas.detach().numpy()[loop_iter])

        if prob_vars.prob == "PandaReacher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "LunarLanderContinuous" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "PandaReacherDense" or prob_vars.prob == "PandaPusherDense":
            particles_t_array = particles[:, h * action_dim : (h + 1) * action_dim]
        else:
            particles_t_array = particles[:, h]
        
        actions = torch.tensor([particles_t_array], dtype=torch.float32).reshape(len(particles),action_dim)

        sim_states = torch.clip(sim_states, prob_vars.states_low, prob_vars.states_high)

        actions = actions.clip(prob_vars.action_low, prob_vars.action_high)

        # Predict next states using the quantile model
        predicted_quantiles = model_QRNN(sim_states, actions)

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
            mid_quantile = get_mid_quantile(prob_vars.num_quantiles, predicted_quantiles)
            next_states = mid_quantile

        # Update state and accumulate cost
        sim_states = next_states
        sim_states = torch.clip(sim_states, prob_vars.states_low, prob_vars.states_high)
        
        if prob_vars.prob == "PandaReacher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "PandaReacherDense" or prob_vars.prob == "PandaPusherDense":
            costs += prob_vars.compute_cost(prob_vars.prob, next_states, h, horizon, actions, prob_vars.goal_state)
            
        else:
            costs += prob_vars.compute_cost(prob_vars.prob, next_states, h, horizon, actions)

    return costs

