'''
Created on Nov 11, 2013

@author: ajn
'''

import numpy
import math
import test.mixed_nlg.lindstenschon2.particle_ls2 as particle_ls2
import param_est

class LS2Est(param_est.ParamEstimation):
        
    def create_initial_estimate(self, params, num):
        particles = numpy.empty(num, particle_ls2.ParticleLS2)
        
        for k in range(len(particles)):
            e = numpy.array([numpy.random.normal(0.0,1.0),]).reshape((-1,1))
            z0 = numpy.zeros((3,1))
            P0 = 0.00001*numpy.eye(3,3)
            particles[k] = particle_ls2.ParticleLS2(eta0=e, z0=z0, P0=P0, params=params)
        return particles

if __name__ == '__main__':
    
    num = 50
    nums = 5
    
    theta_true = numpy.array((1.0, 1.0, 0.3, 0.968, 0.315))
   

    # How many steps forward in time should our simulation run
    steps = 200
    sims = 5

    # Create arrays for storing some values for later plotting    
    vals = numpy.zeros((2, num+1, steps+1))

    estimate = numpy.zeros((5,sims))
    
    max_iter = 1000
    
    for k in range(sims):
        theta_guess = numpy.array((numpy.random.uniform(0.0, 2.0),
                                   numpy.random.uniform(0.0, 2.0),
                                   numpy.random.uniform(0.0, 0.6),
                                   numpy.random.uniform(0.0, 1.0),
                                   numpy.random.uniform(0.0, math.pi/2.0)))
        
        # Create reference
        (y, e, z) = particle_ls2.generate_dataset(theta_true, steps)
   
        # Create an array for our particles 
        ParamEstimator = LS2Est(u=None, y=y)
        ParamEstimator.set_params(theta_guess)
        #ParamEstimator.simulate(num, nums, False)

        (param, Q) = ParamEstimator.maximize(param0=theta_guess, num_part=num, num_traj=nums, max_iter=max_iter,
                                             update_before_predict=False, callback=None)
        
        print param
        
    print "exit"