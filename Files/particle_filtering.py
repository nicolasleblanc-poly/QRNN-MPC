import numpy as np
import random
import torch

from mpc import mpc_func

def choose_action(prob, state, horizon, particles, do_RS, use_sampling, use_mid, use_ASGNN, model_QRNN, model_ASN, action_dim, action_low, action_high, states_low, states_high, nb_reps, std, change_prob, nb_top_particles, nb_random, episode=0, step=1, goal_state=None):

    num_particles = particles.shape[0]
    best_cost = float('inf')
    best_action_sequence = None

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
        num_particles = particles.shape[0]
        costs = torch.zeros(num_particles)
        
        sim_states = torch.tensor(state, dtype=torch.float32).repeat(len(particles), 1)

        # best_cost = float('inf')
        # best_action_sequence = None

        

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

        # Get the indices of the top 15 particles
        top_indices = torch.argsort(costs)[:nb_top_particles]
        top_particles = particles[top_indices]

        # Randomly select len(particles)-nb_random-1 particles from the top nb_top_particles with replacement
        sampled_indices = np.random.choice(nb_top_particles, len(particles)-nb_random-1, replace=True)
        sampled_particles = top_particles[sampled_indices]

        # # Add Gaussian noise to the sampled particles
        # noise = np.random.normal(0, std, sampled_particles.shape)
        # # particles[1:len(particles)-nb_random] = np.clip(sampled_particles + noise, action_low, action_high)

        if prob == "CartPole":
            # Randomly change some of the sampled particles
            change_indices = np.random.choice([True, False], len(sampled_particles), p=[change_prob, 1-change_prob])
            sampled_particles[change_indices] = np.random.randint(0, 2, (len(sampled_particles[change_indices]), horizon))
            
            particles[1:len(particles)-nb_random] = sampled_particles
            # particles[1:len(particles)-nb_random] = np.clip(sampled_particles + noise, action_low, action_high)
            
            # Randomly initialize the remaining particles
            particles[len(particles)-nb_random:] = np.random.randint(0, 2, (nb_random, horizon))
            best_first_action = int(best_action_sequence[0].item())
            
        elif prob == "Acrobot" or prob == "MountainCar":
            # Randomly change some of the sampled particles
            change_indices = np.random.choice([True, False], len(sampled_particles), p=[change_prob, 1-change_prob])
            sampled_particles[change_indices] = np.random.randint(0, 3, (len(sampled_particles[change_indices]), horizon))
            
            particles[1:len(particles)-nb_random] = sampled_particles
            # particles[1:len(particles)-nb_random] = np.clip(best_action_sequence + noise, action_low, action_high)
            
            # Randomly initialize the remaining particles
            particles[len(particles)-nb_random:] = np.random.randint(0, 3, (nb_random, horizon))
            best_first_action = int(best_action_sequence[0].item())
        
        elif prob == "LunarLander":
            # Randomly change some of the sampled particles
            change_indices = np.random.choice([True, False], len(sampled_particles), p=[change_prob, 1-change_prob])
            sampled_particles[change_indices] = np.random.randint(0, 4, (len(sampled_particles[change_indices]), horizon))
            
            particles[1:len(particles)-nb_random] = sampled_particles
            # particles[1:len(particles)-nb_random] = np.clip(best_action_sequence + noise, action_low, action_high)
            
            # Randomly initialize the remaining particles
            particles[len(particles)-nb_random:] = np.random.randint(0, 4, (nb_random, horizon))
            best_first_action = int(best_action_sequence[0].item())

        else: # Pendulum, MountainCarContinuous, PandaReacher, MuJoCoReacher
            # Add Gaussian noise to the sampled particles
            noise = np.random.normal(0, std, sampled_particles.shape)
            # particles[1:len(particles)-nb_random] = np.clip(sampled_particles + noise, action_low, action_high)
            
            particles[1:len(particles)-nb_random] = np.clip(sampled_particles + noise, action_low, action_high)
            # Randomly initialize the remaining particles
            # print("len(particles) ", len(particles), "\n")
            # print("nb_random ", nb_random, "\n")
            
            # print("(nb_random, horizon*action_dim) ", (nb_random, horizon*action_dim), "\n")
            # print("np.random.uniform(actions_low, actions_high, (nb_random, horizon*action_dim)).shape ", np.random.uniform(action_low, action_high, (nb_random, horizon*action_dim)).shape, "\n")
            # print("particles[len(particles)-nb_random:].shape ", particles[len(particles)-nb_random:].shape, "\n")
            
            if prob == "PandaReacher" or prob == "MuJoCoReacher" or prob == "PandaPusher" or prob == "MuJoCoPusher" or prob == "LunarLanderContinuous":
                # print("best_cost ", best_cost, "\n")
                # print("best_action_sequence ", best_action_sequence, "\n")
                
                particles[len(particles)-nb_random:] = np.random.uniform(action_low, action_high, (nb_random, horizon*action_dim))
                best_first_action = best_action_sequence[:action_dim] # .item()
            else: # Pendulum, MountainCarContinuous
                particles[len(particles)-nb_random:] = np.random.uniform(action_low, action_high, (nb_random, horizon))
                best_first_action = best_action_sequence[0].item()

    # print("########################## \n")    

    # best_first_action = int(best_action_sequence[0].item())
    return best_action_sequence, best_first_action, best_cost, particles

