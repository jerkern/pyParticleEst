#!/usr/bin/python

import PF
import PS
import numpy
import param_est

import matplotlib.pyplot as plt
import particle_param_output
from particle_param_output import ParticleParamOutput # Our model definition

z0 = numpy.array([[5.0], [0.0]])
P0 = numpy.eye(2)
Q_in = numpy.diag([ 0.01, 0.01])
B = numpy.array([[0.0, 0.0], [1.0, -1.0]])
Qz = numpy.diag([ 0.0000001, 0.0000001])+B.dot(Q_in.dot(B.T))
R = numpy.array([[1.0]])

class ParticleParamOutputEst(param_est.ParamEstimation):
        
    def create_initial_estimate(self, params, num):
        particles = numpy.empty(num, ParticleParamOutput)
        for k in range(len(particles)):
            particles[k] = ParticleParamOutput(x0=z0, P0=P0, Qz=Qz, R=R, params=params)
        return particles

if __name__ == '__main__':
    
    num = 1
    nums=1
    sims = 400
    
    c_true = 2.5
    # TODO, seems to be problem when c_guess > c_true
    c_guess = numpy.array((2.0,))
    
    # How many steps forward in time should our simulation run
    steps = 32
    
    # Create a random input vector to drive our "correct state"
    mean = -10.0*numpy.hstack((-1.0*numpy.ones(steps/4), 1.0*numpy.ones(steps/2),-1.0*numpy.ones(steps/4) ))
    diff_vec = numpy.random.normal(mean, 0.01, steps)
    uvec = numpy.vstack((1.0-diff_vec/2, 1.0+diff_vec/2))

    # Create arrays for storing states for later plotting    
    
    #yvec = numpy.zeros((1, steps))

    estimate = numpy.zeros((1,sims))
    plt.ion()
    fig1 = plt.figure()
    fig2 = plt.figure()
    
  
    
    for k in range(sims):
        # Create reference
        (ulist, ylist, states) = particle_param_output.generate_refernce(z0, P0, Qz, R, uvec, steps, c_true)
    
        print "estimation start"
        
        # Create an array for our particles 
        ParamEstimator = ParticleParamOutputEst(ulist, ylist)
        param = ParamEstimator.maximize(param0=c_guess, num_part=num, num_traj=nums)
        
        estimate[0,k] = param
        
        plt.figure(fig1.number)
        plt.clf()
        
        svals = numpy.zeros((2, nums, steps+1))
        for i in range(steps+1):
            for j in range(nums):
                svals[:,j,i]=ParamEstimator.straj[j].traj[i].kf.z.ravel()
                
        for j in range(nums):
            plt.plot(range(steps+1),svals[0,j,:],'g-')
            plt.plot(range(steps+1),svals[1,j,:],'r-')

                
        plt.plot(range(steps+1),states[0,:],'go')
        plt.plot(range(steps+1),states[1,:],'ro')
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