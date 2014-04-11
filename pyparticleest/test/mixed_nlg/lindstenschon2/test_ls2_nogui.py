'''
Created on Nov 11, 2013

@author: ajn
'''

import numpy
import math
import pyparticleest.test.mixed_nlg.lindstenschon2.particle_ls2 as particle_ls2
import pyparticleest.param_est as param_est

if __name__ == '__main__':
    
    num = 50
    nums = 10
    
    theta_true = numpy.array((1.0, 1.0, 0.3, 0.968, 0.315))
   

    # How many steps forward in time should our simulation run
    steps = 200
    sims = 20

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
        #numpy.random.seed(k)
        (y, e, z) = particle_ls2.generate_dataset(theta_true, steps)
   
        # Create an array for our particles 
        model = particle_ls2.ParticleLS2(theta_guess)
        ParamEstimator = param_est.ParamEstimation(model=model, u=None, y=y)
        ParamEstimator.set_params(theta_guess)
        #ParamEstimator.simulate(num, nums, False)

        (param, Q) = ParamEstimator.maximize(param0=theta_guess, num_part=num, num_traj=nums, max_iter=max_iter,
                                             callback=None, analytic_gradient=True)
        
        print "%.4f %.4f %.4f %.4f %.4f" % tuple(param)
        
    print "exit"