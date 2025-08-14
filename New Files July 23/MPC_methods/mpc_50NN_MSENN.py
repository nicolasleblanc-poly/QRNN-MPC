import numpy as np
import torch

def mpc_50NN_MSENN_func(prob_vars, sim_states, particles, use_ASGNN, model_state, use_sampling, use_mid, model_ASN): # , use_LBFGSB=False
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

        next_states = model_state(sim_states, actions)

        # Update state and accumulate cost
        sim_states = next_states
        sim_states = torch.clip(sim_states, prob_vars.states_low, prob_vars.states_high)
        
        if prob_vars.prob == "PandaReacher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "PandaReacherDense" or prob_vars.prob == "PandaPusherDense":
            costs += prob_vars.compute_cost(prob_vars.prob, next_states, h, horizon, actions, prob_vars.goal_state)
            
        else:
            costs += prob_vars.compute_cost(prob_vars.prob, next_states, h, horizon, actions) # h, horizon,

    return costs

