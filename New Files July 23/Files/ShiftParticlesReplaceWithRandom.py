import numpy as np

def ShiftParticlesReplaceWithRandom_func(prob_vars, particles):
    
    '''
    Shift the action sequences/particles to the left by removing the first action and adding new random actions at the end
    by sampling uniformly from the action space.
    '''
    
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
    
    return particles
