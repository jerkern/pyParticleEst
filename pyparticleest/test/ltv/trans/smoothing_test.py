'''
Created on Nov 28, 2013

@author: ajn
'''

import param_est
import numpy
import math
import matplotlib.pyplot as plt

from test.ltv.trans.particle_param_trans import ParticleParamTrans # Our model definition
import test.ltv.trans.particle_param_trans as particle_param_trans

z0 = numpy.array([0.0, 1.0, ])
P0 = numpy.eye(2)

class ParticleParamTransEst(param_est.ParamEstimation):
        
    def create_initial_estimate(self, params, num):
        particles = numpy.empty(num, ParticleParamTrans)
        
        for k in range(len(particles)):
            particles[k] = ParticleParamTrans(z0=z0, P0=P0, params=params)
        return particles


if __name__ == '__main__':

    num=1
    nums=1
    theta_true = 0.1
    theta_guess = 0.1
    #theta_guess = theta_true   
    R = numpy.array([[0.1]])
    Q = numpy.array([ 0.1, 0.1])



    # How many steps forward in time should our simulation run
    steps = 200

    (ylist, states) = particle_param_trans.generate_reference(z0, P0, theta_true, steps)
        
    ParamEstimator = ParticleParamTransEst(u=None, y=ylist)
    ParamEstimator.set_params(numpy.array((theta_guess,)).reshape((-1,1)))
    ParamEstimator.simulate(num_part=num, num_traj=nums)
    
    svals = numpy.zeros((4, nums, steps+1))
    
    svals_p = numpy.zeros((4, nums, steps+1))
    
    x = numpy.asarray(range(steps+1))
    plt.plot(x[1:],numpy.asarray(ylist)[:,0],'b.')
    plt.plot(range(steps+1),states[0,:],'go')
    plt.plot(range(steps+1),states[1,:],'ro')
    plt.title("Param = %s" % theta_true)
    
    for i in range(steps+1):
        for j in range(nums):
            svals[:2,j,i]=ParamEstimator.straj[j].traj[i].kf.z.ravel()
            svals[2:,j,i]=ParamEstimator.straj[j].traj[i].z_tN.ravel()
            
            svals_p[0,j,i]=math.sqrt(ParamEstimator.straj[j].traj[i].kf.P[0,0])
            svals_p[1,j,i]=math.sqrt(ParamEstimator.straj[j].traj[i].kf.P[1,1])
            svals_p[2,j,i]=math.sqrt(ParamEstimator.straj[j].traj[i].P_tN[0,0])
            svals_p[3,j,i]=math.sqrt(ParamEstimator.straj[j].traj[i].P_tN[1,1])

            
    for j in range(nums):
        plt.plot(range(steps+1),svals[0,j,:],'g-')
        plt.plot(range(steps+1),svals[0,j,:]+2.0*svals_p[0,j,:],'g.')
        plt.plot(range(steps+1),svals[0,j,:]-2.0*svals_p[0,j,:],'g.')
        plt.plot(range(steps+1),svals[1,j,:],'r-')
        plt.plot(range(steps+1),svals[1,j,:]+2.0*svals_p[1,j,:],'r.')
        plt.plot(range(steps+1),svals[1,j,:]-2.0*svals_p[1,j,:],'r.')
        plt.plot(range(steps+1),svals[2,j,:],'b--')
        plt.plot(range(steps+1),svals[2,j,:]+2.0*svals_p[2,j,:],'b.')
        plt.plot(range(steps+1),svals[2,j,:]-2.0*svals_p[2,j,:],'b.')
        plt.plot(range(steps+1),svals[3,j,:],'k--')
        plt.plot(range(steps+1),svals[3,j,:]+2.0*svals_p[3,j,:],'k.')
        plt.plot(range(steps+1),svals[3,j,:]-2.0*svals_p[3,j,:],'k.')
        
    plt.show()
        