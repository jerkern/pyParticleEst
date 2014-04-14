'''
Created on Nov 11, 2013

@author: ajn
'''

import pyparticleest.param_est as param_est
import numpy
import math
import pyparticleest.test.mixed_nlg.lindstenschon2.particle_ls2 as particle_ls2

if __name__ == '__main__':
    num = 50
    nums = 5
    numpy.random.seed(4) #3
    theta_true = numpy.array((1.0, 1.0, 0.3, 0.968, 0.315))

    # How many steps forward in time should our simulation run
    steps = 200
    sims = 1

    # Create arrays for storing some values for later plotting    
    vals = numpy.zeros((2, num+1, steps+1))

    estimate = numpy.zeros((5,sims))

    (y, e, z) = particle_ls2.generate_dataset(theta_true, steps)
    
    model = particle_ls2.ParticleLS2(theta_true)
    # Create an array for our particles 
    gt = param_est.GradientTest(model=model, u=None, y=y)
    gt.set_params(theta_true)
    
    param_id = 4
    param_steps = 101
    tval = theta_true[param_id]
    param_vals = numpy.linspace(tval-math.fabs(tval), tval+math.fabs(tval), param_steps)
    gt.test(param_id, param_vals, nums=nums)

    gt.plot_y.plot(1)
    gt.plot_xn.plot(2)
    gt.plot_x0.plot(3)