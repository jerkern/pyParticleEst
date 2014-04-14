#!/usr/bin/python

import numpy
import pyparticleest.param_est as param_est
import math

import matplotlib.pyplot as plt
from pyparticleest.test.ltv.particle_trivial import ParticleTrivial # Our model definition
import pyparticleest.test.ltv.particle_trivial as particle_trivial

z0 = numpy.array([0.0,])
P0 = numpy.eye(1)

def callback_sim(estimator):
    svals = numpy.zeros((1, nums, steps+1))
        
    for i in range(steps+1):
        for j in range(nums):
            svals[:,j,i]=ParamEstimator.straj[j].traj[i].kf.z.ravel()

    plt.figure(3)
    plt.clf()
            
    for j in range(nums):
        plt.plot(range(steps+1),svals[0,j,:],'g-')
    
    print "callback_sim"
     
    plt.show()
    plt.draw()
       
        
            

class ParticleParamTransEst(param_est.ParamEstimation):
        
    def create_initial_estimate(self, params, num):
        particles = numpy.empty(num, ParticleTrivial)
        
        for k in range(len(particles)):
            particles[k] = ParticleTrivial(z0=z0, P0=P0, params=params)
        return particles

if __name__ == '__main__':
    
    num = 1
    nums = 1
    
    theta_true = 0.1
    theta_guess = 2.0
   

    # How many steps forward in time should our simulation run
    steps = 200
    sims = 5

    # Create arrays for storing some values for later plotting    
    yvec = numpy.zeros((1, steps))

    estimate = numpy.zeros((1,sims))
    
    plt.ion()
    fig1 = plt.figure()
    fig2 = plt.figure()
    
    for k in range(sims):
        print k
        # Create reference
        (ylist, states) = particle_trivial.generate_reference(z0, P0, theta_true, steps)
    
        print "estimation start"
        
        plt.figure(fig1.number)
        plt.clf()
        x = numpy.asarray(range(steps+1))
        plt.plot(x[1:],numpy.asarray(ylist)[:,0],'b.')
        plt.plot(range(steps+1),states[0,:],'go')
        plt.title("Param = %s" % theta_true)
        fig1.show()
            
        # Create an array for our particles 
        ParamEstimator = ParticleParamTransEst(u=None, y=ylist)
        ParamEstimator.set_params(numpy.array((theta_guess,)))
        #ParamEstimator.simulate(num_part=num, num_traj=nums)
        print "maximization start"
        (param, Q) = ParamEstimator.maximize(param0=numpy.array((theta_guess,)), num_part=num, num_traj=nums,
                                             callback_sim=callback_sim, tol=0.000001, max_iter=50)
        
        # Extract data from trajectories for plotting
#        i=0
#        for step in ParamEstimator.pt:
#            pa = step.pa
#            for j in range(pa.num):
#                vals[0,j,i]=pa.part[j].eta[0,0]
#                vals[1,j,i]=pa.part[j].kf.z.reshape(-1)
#            i += 1
#        
        svals = numpy.zeros((2, nums, steps+1))
        
        svals_p = numpy.zeros((2, nums, steps+1))
        
        plt.figure(fig1.number)
        
        for i in range(steps+1):
            for j in range(nums):
                svals[:,j,i]=ParamEstimator.straj[j].traj[i].kf.z.ravel()
                
                svals[0,j,i]=ParamEstimator.straj[j].traj[i].kf.z.ravel()
                svals[1,j,i]=ParamEstimator.straj[j].traj[i].z_tN.ravel()
                
                svals_p[0,j,i]=math.sqrt(ParamEstimator.straj[j].traj[i].kf.P[0,0])
                svals_p[1,j,i]=math.sqrt(ParamEstimator.straj[j].traj[i].P_tN[0,0])
                
        for j in range(nums):
            plt.plot(range(steps+1),svals[0,j,:],'g-')
            plt.plot(range(steps+1),svals[0,j,:]+2.0*svals_p[0,j,:],'g-')
            plt.plot(range(steps+1),svals[0,j,:]-2.0*svals_p[0,j,:],'g-')
            plt.plot(range(steps+1),svals[1,j,:],'r--')
            plt.plot(range(steps+1),svals[1,j,:]+2.0*svals_p[1,j,:],'r--')
            plt.plot(range(steps+1),svals[1,j,:]-2.0*svals_p[1,j,:],'r--')

       
        
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