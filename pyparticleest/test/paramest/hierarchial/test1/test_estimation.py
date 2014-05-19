'''
Created on Dec 13, 2013

@author: ajn
'''
#!/usr/bin/python

import numpy
import pyparticleest.paramest.param_est as param_est
import math

import matplotlib.pyplot as plt
import pyparticleest.test.hierarchial.test1.particle as particle



class ParticleParamTransEst(param_est.ParamEstimation):
        
    def create_initial_estimate(self, params, num):
        particles = numpy.empty(num, particle.ParticleTest1)
        
        for k in range(len(particles)):
            e0 = numpy.random.uniform(-math.pi/32.0, math.pi/32.0)
            particles[k] = particle.ParticleTest1(e0=e0)
        return particles

if __name__ == '__main__':
    
    num = 100
    nums = 5

    # How many steps forward in time should our simulation run
    steps = 200

    plt.ion()
    fig1 = plt.figure()
    
    # Create reference
    (ylist, states) = particle.generate_reference(steps)
    
    plt.figure(fig1.number)
    plt.clf()
    x = numpy.asarray(range(steps+1))
    plt.plot(x[1:],numpy.asarray(ylist)[:,0],'b.')
    plt.plot(x[1:],numpy.asarray(ylist)[:,1],'b.')
    plt.plot(range(steps+1),states[0,:],'go')
    plt.plot(range(steps+1),states[1,:],'ro')
    plt.plot(range(steps+1),states[2,:],'ko')
    plt.draw()
    fig1.show()
            
# Create an array for our particles 
    ParamEstimator = ParticleParamTransEst(u=None, y=ylist)
    ParamEstimator.simulate(num_part=num, num_traj=nums)
    svals = numpy.zeros((3, nums, steps+1))
        
    for i in range(steps+1):
        for j in range(nums):
            svals[0,j,i]=ParamEstimator.straj[j].traj[i].eta
            svals[1:,j,i]=ParamEstimator.straj[j].traj[i].kf.z.ravel()
            
    for j in range(nums):
        plt.plot(range(steps+1),svals[0,j,:],'g-')
        plt.plot(range(steps+1),svals[1,j,:],'r-')
        plt.plot(range(steps+1),svals[2,j,:],'k-')

    plt.ioff()
    plt.show()
    plt.draw()
    print "exit"