'''
Created on Nov 29, 2013

@author: ajn
'''

import param_est
import numpy
import math
import matplotlib.pyplot as plt

from test.ltv.particle_trivial import ParticleTrivial # Our model definition
import test.ltv.particle_trivial as particle_trivial

z0 = numpy.array([0.0, ])
P0 = numpy.eye(1)

class ParticleParamTransEst(param_est.ParamEstimation):
        
    def create_initial_estimate(self, params, num):
        particles = numpy.empty(num, ParticleTrivial)
        
        for k in range(len(particles)):
            particles[k] = ParticleTrivial(z0=z0, P0=P0, params=params)
        return particles


if __name__ == '__main__':

    num=1
    nums=1
    theta_true = 1.0
    theta_guess = 1.0
    #theta_guess = theta_true   

    # How many steps forward in time should our simulation run
    steps = 2000

    (ylist, states) = particle_trivial.generate_reference(z0, P0, theta_true, steps)
        
    ParamEstimator = ParticleParamTransEst(u=None, y=ylist)
    ParamEstimator.set_params(numpy.array((theta_guess,)))
    ParamEstimator.simulate(num_part=num, num_traj=nums)
    
    svals = numpy.zeros((2, nums, steps+1))
    
    svals_p = numpy.zeros((2, nums, steps+1))
    
    x = numpy.asarray(range(steps+1))
    plt.plot(x[1:],numpy.asarray(ylist)[:,0],'b.')
    plt.plot(range(steps+1),states[0,:],'go')
    plt.title("Param = %s" % theta_true)
    
    for i in range(steps+1):
        for j in range(nums):
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
        
    plt.show()
        