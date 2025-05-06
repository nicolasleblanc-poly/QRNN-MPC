import numpy as np
import random
import torch

# from run import compute_cost

def mpc_50NN_MSENN_func(prob_vars, sim_states, particles, use_ASGNN, model_state, use_sampling, use_mid, model_ASN): # , use_LBFGSB=False
    horizon = prob_vars.horizon
    num_particles = prob_vars.num_particles
    action_dim = prob_vars.action_dim
    nb_actions = prob_vars.nb_actions

    costs = torch.zeros(prob_vars.num_particles)
    
    for h in range(horizon):
        if use_ASGNN and h == horizon-1:
            # Use ASGNN to generate last action/actions based of last non-terminal state
            if prob_vars.prob == "CartPole" or prob_vars.prob == "Acrobot" or prob_vars.prob == "LunarLander" or prob_vars.prob == "MountainCar": # Discrete actions

                # print("sim_states ", sim_states, "\n")

                action_probs = model_ASN(sim_states, torch.tensor(prob_vars.goal_state, dtype=torch.float32))
                
                # print("action_probs ", action_probs, "\n")
                # print("action_probs[loop_iter].detach().numpy() ", action_probs[0].detach().numpy(), "\n")
                
                # particles[loop_iter, j] = np.random.choice(nb_actions, num_particles, p=action_probs[loop_iter].detach().numpy())

                for loop_iter in range(num_particles):
                    # print("np.random.choice(nb_actions, num_particles, p=action_probs[loop_iter].detach().numpy()) ", np.random.choice(nb_actions, 1, p=action_probs[loop_iter].detach().numpy()), "\n")
                    particles[loop_iter, -1] = np.random.choice(nb_actions, 1, p=action_probs[loop_iter].detach().numpy())

                # pass

            else: # Continuous actions
                action_mus, action_sigmas = model_ASN(torch.tensor(sim_states, dtype=torch.float32), torch.tensor(prob_vars.goal_state, dtype=torch.float32))

                # print("action_mus ", action_mus, "\n")
                # print("action_sigmas ", action_sigmas, "\n")
                # print("action_mus.shape ", action_mus.shape, "\n")
                # print("action_sigmas.shape ", action_sigmas.shape, "\n")

                for loop_iter in range(num_particles):
                    particles[loop_iter, -action_dim:] = np.random.normal(action_mus.detach().numpy()[loop_iter], action_sigmas.detach().numpy()[loop_iter])

                    # if particles[loop_iter, -action_dim:-(action_dim)+1] < action_low or particles[loop_iter, -(action_dim)+1:] < action_low or particles[loop_iter, -action_dim:-(action_dim)+1] > action_high or particles[loop_iter, -(action_dim)+1:] > action_high:
                    #     print("actions ", actions, "\n")
                    #     print("sim_states[loop_iter] ", sim_states[loop_iter], "\n")
                    #     print("action_mus[loop_iter] ", action_mus[loop_iter], "\n")
                    #     print("action_sigmas[loop_iter] ", action_sigmas[loop_iter], "\n")

        # if use_LBFGSB: # particles is a 1D array
        #     if prob_vars.prob == "PandaReacher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "LunarLanderContinuous" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoPusher":
        #         particles_t_array = particles[h * action_dim : (h + 1) * action_dim]
        #     else:
        #         particles_t_array = particles[h:h+1]

        # else:
        if prob_vars.prob == "PandaReacher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "LunarLanderContinuous" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoPusher":
            particles_t_array = particles[:, h * action_dim : (h + 1) * action_dim]
        else:
            # print("particles.shape ", particles.shape, "\n")
            particles_t_array = particles[:, h]
        
        actions = torch.tensor([particles_t_array], dtype=torch.float32).reshape(len(particles),action_dim)
        
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
        sim_states = torch.clip(sim_states, prob_vars.states_low, prob_vars.states_high)
        # sim_states = 2 * ((sim_states - prob_vars.states_low) / (prob_vars.states_high - prob_vars.states_low)) - 1
        actions = actions.clip(prob_vars.action_low, prob_vars.action_high)

        # Predict next states using the quantile model
        # predicted_quantiles = model_QRNN(sim_states, actions)
        # # print("predicted_quantiles ", predicted_quantiles, "\n")

        # if use_sampling:
        #     # '''
        #     ############# Need to add the sorta quantile ponderated sum here #############
        #     # chosen_quantiles = np.random.uniform(0, 1, (num_particles, state_dim))
        #     chosen_quantiles = torch.rand(num_particles, prob_vars.state_dim)
        #     # https://pytorch.org/docs/main/generated/torch.rand.html
        #     bottom_int_indices = torch.floor(chosen_quantiles*10).to(torch.int) # .astype(int)
        #     top_int_indices = bottom_int_indices+1
        #     top_int_indices[top_int_indices == 11] = 10
            
        #     # Expand batch dimension to match indexing (assuming batch_size = 1)
        #     batch_indices = torch.zeros((num_particles, prob_vars.state_dim), dtype=torch.long)  # Shape (num_particles, state_dim)
            
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
        # print("next_states ", next_states, "\n")

        # Update state and accumulate cost
        sim_states = next_states
        sim_states = torch.clip(sim_states, prob_vars.states_low, prob_vars.states_high)
        # sim_states = 2 * ((sim_states - prob_vars.states_low) / (prob_vars.states_high - prob_vars.states_low)) - 1
        
        if prob_vars.prob == "PandaReacher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoPusher":
            costs += prob_vars.compute_cost(prob_vars.prob, next_states, h, horizon, actions, prob_vars.goal_state)
            # print("costs ", costs, "\n")
            # print("next_states ", next_states, "\n")
            
        else:
            costs += prob_vars.compute_cost(prob_vars.prob, next_states, h, horizon, actions) # h, horizon,
        # trajectory_costs = compute_cost(next_states, h, horizon, actions)

    return costs

