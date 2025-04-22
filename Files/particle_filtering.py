import numpy as np
import random
import torch

from mpc import mpc_func

def particle_filtering_func(prob_vars, particles, costs, best_action_sequence):

    # Get the indices of the top 15 particles
    top_indices = torch.argsort(costs)[:prob_vars.nb_top_particles]
    top_particles = particles[top_indices]

    # Randomly select len(particles)-nb_random-1 particles from the top nb_top_particles with replacement
    sampled_indices = np.random.choice(prob_vars.nb_top_particles, prob_vars.num_particles-prob_vars.nb_random-1, replace=True)
    sampled_particles = top_particles[sampled_indices]

    # # Add Gaussian noise to the sampled particles
    # noise = np.random.normal(0, std, sampled_particles.shape)
    # # particles[1:len(particles)-nb_random] = np.clip(sampled_particles + noise, action_low, action_high)

    if prob_vars.prob == "CartPole":
        # Randomly change some of the sampled particles
        change_indices = np.random.choice([True, False], len(sampled_particles), p=[prob_vars.change_prob, 1-prob_vars.change_prob])
        sampled_particles[change_indices] = np.random.randint(0, 2, (len(sampled_particles[change_indices]), prob_vars.horizon))
        
        particles[1:prob_vars.num_particles-prob_vars.nb_random] = sampled_particles
        # particles[1:len(particles)-nb_random] = np.clip(sampled_particles + noise, action_low, action_high)
        
        # Randomly initialize the remaining particles
        particles[prob_vars.num_particles-prob_vars.nb_random:] = np.random.randint(0, 2, (prob_vars.nb_random, prob_vars.horizon))
        best_first_action = int(best_action_sequence[0].item())
        
    elif prob_vars.prob == "Acrobot" or prob_vars.prob == "MountainCar":
        # Randomly change some of the sampled particles
        change_indices = np.random.choice([True, False], len(sampled_particles), p=[prob_vars.change_prob, 1-prob_vars.change_prob])
        sampled_particles[change_indices] = np.random.randint(0, 3, (len(sampled_particles[change_indices]), prob_vars.horizon))
        
        particles[1:prob_vars.num_particles-prob_vars.nb_random] = sampled_particles
        # particles[1:len(particles)-nb_random] = np.clip(best_action_sequence + noise, action_low, action_high)
        
        # Randomly initialize the remaining particles
        particles[prob_vars.num_particles-prob_vars.nb_random:] = np.random.randint(0, 3, (prob_vars.nb_random, prob_vars.horizon))
        best_first_action = int(best_action_sequence[0].item())
    
    elif prob_vars.prob == "LunarLander":
        # Randomly change some of the sampled particles
        change_indices = np.random.choice([True, False], len(sampled_particles), p=[prob_vars.change_prob, 1-prob_vars.change_prob])
        sampled_particles[change_indices] = np.random.randint(0, 4, (len(sampled_particles[change_indices]), prob_vars.horizon))
        
        particles[1:prob_vars.num_particles-prob_vars.nb_random] = sampled_particles
        # particles[1:len(particles)-nb_random] = np.clip(best_action_sequence + noise, action_low, action_high)
        
        # Randomly initialize the remaining particles
        particles[prob_vars.num_particles-prob_vars.nb_random:] = np.random.randint(0, 4, (prob_vars.nb_random, prob_vars.horizon))
        best_first_action = int(best_action_sequence[0].item())

    else: # Pendulum, MountainCarContinuous, PandaReacher, MuJoCoReacher
        # Add Gaussian noise to the sampled particles
        noise = np.random.normal(0, prob_vars.std, sampled_particles.shape)
        # particles[1:len(particles)-nb_random] = np.clip(sampled_particles + noise, action_low, action_high)
        
        particles[1:prob_vars.num_particles-prob_vars.nb_random] = np.clip(sampled_particles + noise, prob_vars.action_low, prob_vars.action_high)
        # Randomly initialize the remaining particles
        # print("len(particles) ", len(particles), "\n")
        # print("nb_random ", nb_random, "\n")
        
        # print("(nb_random, horizon*action_dim) ", (nb_random, horizon*action_dim), "\n")
        # print("np.random.uniform(actions_low, actions_high, (nb_random, horizon*action_dim)).shape ", np.random.uniform(action_low, action_high, (nb_random, horizon*action_dim)).shape, "\n")
        # print("particles[len(particles)-nb_random:].shape ", particles[len(particles)-nb_random:].shape, "\n")
        
        if prob_vars.prob == "PandaReacher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "LunarLanderContinuous":
            # print("best_cost ", best_cost, "\n")
            # print("best_action_sequence ", best_action_sequence, "\n")
            
            particles[prob_vars.num_particles-prob_vars.nb_random:] = np.random.uniform(prob_vars.action_low, prob_vars.action_high, (prob_vars.nb_random, prob_vars.horizon*prob_vars.action_dim))
            best_first_action = best_action_sequence[:prob_vars.action_dim] # .item()
        else: # Pendulum, MountainCarContinuous
            particles[prob_vars.num_particles-prob_vars.nb_random:] = np.random.uniform(prob_vars.action_low, prob_vars.action_high, (prob_vars.nb_random, prob_vars.horizon))
            best_first_action = best_action_sequence[0].item()
    
    return best_first_action, particles


