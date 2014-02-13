#!/usr/bin/python

import numpy
import pyparticleest.param_est as param_est

import matplotlib.pyplot as plt
import pyparticleest.test.ltv.trans.particle_param_trans as particle_param_trans
from pyparticleest.test.ltv.trans.particle_param_trans import ParticleParamTrans # Our model definition

z0 = numpy.array([0.0, 1.0, ])
P0 = numpy.eye(2)

class ParticleParamTransEst(param_est.ParamEstimation):
        
    def create_initial_estimate(self, params, num):
        particles = numpy.empty(num, ParticleParamTrans)
        
        for k in range(len(particles)):
            particles[k] = ParticleParamTrans(z0=z0, P0=P0, params=params)
        return particles

if __name__ == '__main__':
    
    num = 1
    nums = 1
    
    theta_true = 0.1
    theta_guess = 0.3
    #theta_guess = theta_true   
    R = numpy.array([[0.1]])
    Q = numpy.array([ 0.1, 0.1])

    

    # How many steps forward in time should our simulation run
    steps = 200
    sims = 100

    # Create arrays for storing some values for later plotting    
    vals = numpy.zeros((2, num+1, steps+1))
    yvec = numpy.zeros((1, steps))

    estimate = numpy.zeros((1,sims))
    
    plt.ion()
    fig1 = plt.figure()
    fig2 = plt.figure()
    
    for k in range(sims):
        print k
        # Create reference
        (ylist, states) = particle_param_trans.generate_reference(z0, P0, theta_true, steps)
    
        plt.figure(fig1.number)
        plt.clf()
        x = numpy.asarray(range(steps+1))
        plt.plot(x[1:],numpy.asarray(ylist)[:,0],'b.')
        plt.plot(range(steps+1),states[0,:],'go')
        plt.plot(range(steps+1),states[1,:],'ro')
        plt.title("Param = %s" % theta_true)
        plt.draw()
        fig1.show()
            
        # Create an array for our particles 
        ParamEstimator = ParticleParamTransEst(u=None, y=ylist)
        ParamEstimator.set_params(numpy.array((theta_guess,)).reshape((-1,1)))
        #ParamEstimator.simulate(num_part=num, num_traj=nums)
        print "maximization start"
        (param, Q) = ParamEstimator.maximize(param0=numpy.array((theta_guess,)), num_part=num, num_traj=nums)
        
        svals = numpy.zeros((2, nums, steps+1))
        
        for i in range(steps+1):
            for j in range(nums):
                svals[:,j,i]=ParamEstimator.straj[j].traj[i].kf.z.ravel()
                
        for j in range(nums):
            plt.plot(range(steps+1),svals[0,j,:],'g-')
            plt.plot(range(steps+1),svals[1,j,:],'r-')

       
        
        estimate[0,k] = param
        
        plt.figure(fig2.number)
        plt.clf()
        plt.hist(estimate[0,:(k+1)].T)
        fig2.show()
        plt.show()
        plt.draw()

        

    
    plt.hist(estimate.T)
    plt.ioff()
    plt.show()
    plt.draw()
    print "exit"