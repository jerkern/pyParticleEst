#!/usr/bin/python

import numpy
import param_est

import matplotlib.pyplot as plt
import test.mixed_nlg.initial.particle_param_init as particle_param_init
from particle_param_init import ParticleParamInit as PartModel

P0 = 100.0*numpy.eye(1)
Qz = numpy.diag([0.01])
R = numpy.diag([0.01])

class ParticleParamInitEst(param_est.ParamEstimation):
        
    def create_initial_estimate(self, params, num):
        particles = numpy.empty(num, PartModel)
        for k in range(len(particles)):
            particles[k] = PartModel(P0=P0, Qz=Qz, R=R, params=params)
        return particles

if __name__ == '__main__':
    
    num = 1
    nums=1
    sims = 50
    
    z0_true = numpy.array((2.0,))
    z0_guess = numpy.array((5.0,))
    
    # How many steps forward in time should our simulation run
    steps = 32
    
    
    estimate = numpy.zeros((1,sims))
    plt.ion()
    fig1 = plt.figure()
    fig2 = plt.figure()
    
    for k in range(sims):
        # Create reference
        (ylist, states) = particle_param_init.generate_reference(P0=P0, Qz=Qz, R=R, params=z0_true, steps=steps)

        print "estimation start"
        
        # Create an array for our particles 
        ParamEstimator = ParticleParamInitEst(None, ylist)
        param = ParamEstimator.maximize(param0=z0_guess, num_part=num, num_traj=nums, tol=0.001)
        
        estimate[0,k] = param
        
        plt.figure(fig1.number)
        plt.clf()
        
        svals = numpy.zeros((1, nums, steps+1))
        for i in range(steps+1):
            for j in range(nums):
                svals[:,j,i]=ParamEstimator.straj[j].traj[i].kf.z.ravel()
                
        for j in range(nums):
            plt.plot(range(steps+1),svals[0,j,:],'g-')

                
        plt.plot(range(steps+1),states[0,:],'go')
        fig1.show()
        
        plt.figure(fig2.number)
        plt.clf()
        plt.hist(estimate[0,:(k+1)].T)
        fig2.show()
        plt.show()
        plt.draw()

    plt.ioff()
    
    plt.hist(estimate.T)
    plt.ioff()
    plt.show()
    plt.draw()
    print "exit"