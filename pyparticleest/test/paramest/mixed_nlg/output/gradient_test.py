'''
Created on Sep 17, 2013

@author: ajn
'''
import pyparticleest.param_est as param_est
import numpy
import matplotlib.pyplot as plt

import particle_param_output
from pyparticleest.test.mixed_nlg.output.particle_param_output import ParticleParamOutput as PartModel # Our model definition

    
if __name__ == '__main__':
    
    num = 1
    nums=1
    sims = 1
    
    c_true = 2.5
    c_guess = numpy.array((2.5,))
    
    # How many steps forward in time should our simulation run
    steps = 32
    
    # Create a random input vector to drive our "correct state"
    mean = -numpy.hstack((-1.0*numpy.ones(steps/4), 1.0*numpy.ones(steps/2),-1.0*numpy.ones(steps/4) ))
    diff_vec = numpy.random.normal(mean, .1, steps)
    uvec = numpy.vstack((1.0-diff_vec/2, 1.0+diff_vec/2))

    # Create reference
    (ulist, ylist, states) = particle_param_output.generate_refernce(z0, P0, Qz, R, uvec, steps, c_true)
    model = PartModel.ParticleParamOutput((theta_true,), 
                                         R=R,
                                         Qxi=Qxi,
                                         Qz=Qz)
    # Create an array for our particles 
    gt = param_est.GradientTest(u=ulist, y=ylist)
    gt.set_params(numpy.array((c_true,)).reshape((-1,1)))
    
    param_steps = 101
    param_vals = numpy.linspace(-10.0, 10.0, param_steps)
    gt.test(0, param_vals)

    gt.plot_y.plot(1)
    gt.plot_xn.plot(2)
    gt.plot_x0.plot(3)
    