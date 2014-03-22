'''
Created on Nov 11, 2013

@author: ajn
'''

import numpy
import math
import matplotlib.pyplot as plt
import pyparticleest.test.mixed_nlg.lsB.particle_lsb as particle_lsb
import pyparticleest.param_est as param_est
import scipy.io

if __name__ == '__main__':
    
    num = 300
    nums = 10
    
    # How many steps forward in time should our simulation run
    steps = 100
    sims = 1

    # Create arrays for storing some values for later plotting    
    vals = numpy.zeros((2, num+1, steps+1))

    estimate = numpy.zeros((5,sims))
    
    plt.ion()
    C_theta = numpy.array([[ 0.0, 0.04, 0.044, 0.08],])
    for k in range(sims):
        print k
        # Create reference
        numpy.random.seed(1)
        #numpy.random.seed(10)
        (y, e, z) = particle_lsb.generate_dataset(steps)
        # Store values for last time-step aswell    
    
        print "estimation start"
        
        x = numpy.asarray(range(steps+1))
        model = particle_lsb.ParticleLSB()
        # Create an array for our particles 
        ParamEstimator = param_est.ParamEstimation(model=model, u=None, y=y)
        ParamEstimator.simulate(num, nums, res=0.67, filter='PF')

        
        svals = numpy.zeros((2, nums, steps+1))
        vals = numpy.zeros((2, num, steps+1))
 
        for i in range(steps+1):
            (xil, zl, Pl) = model.get_states(ParamEstimator.straj.straj[i])
            svals[0,:,i] = numpy.vstack(xil).ravel()
            svals[1,:,i] = 25.0+C_theta.dot(numpy.hstack(zl)).ravel()
            (xil, zl, Pl) = model.get_states(ParamEstimator.pt.traj[i].pa.part)
            vals[0,:,i]=numpy.vstack(xil).ravel()
            vals[1,:,i]=25.0+C_theta.dot(numpy.hstack(zl)).ravel()
                
        svals_mean = numpy.mean(svals,1)
        plt.figure()

        for j in range(num):   
            plt.plot(range(steps+1),vals[0,j,:],'.', markersize=3.0, color='#BBBBBB')
        plt.plot(x, e.T,'k-',markersize=1.0)
        for j in range(nums):
            plt.plot(range(steps+1),svals[0,j,:],'--', markersize=2.0, color='#999999', dashes=(7,50))
        #plt.plot(range(steps+1),svals_mean[0,:],'--', markersize=1.0, color='1.0')

        #plt.savefig('rbps_fail_xi.eps', bbox_inches='tight')
        plt.show()
        
        plt.figure()
        for j in range(num):   
            plt.plot(range(steps+1),vals[1,j,:],'.', markersize=3.0, color='#BBBBBB')
        plt.plot(x, (25.0+C_theta.dot(z)).ravel(),'k-',markersize=1.0)
        for j in range(nums):
            plt.plot(range(steps+1),svals[1,j,:],'--', markersize=2.0, color='#999999', dashes=(10,25))
        #plt.plot(range(steps+1),svals_mean[1,:],'--', markersize=1.0, color='1.0')
        #plt.savefig('rbps_fail_theta.eps', bbox_inches='tight')
        plt.show()

        plt.draw()
        # Export data for plotting in matlab
        scipy.io.savemat('test_lsb.mat', {'svals_mean': svals_mean, 'vals': vals, 
                                          'svals': svals, 'y': y, 'z': z, 'e':e})
    
    plt.ioff()
    plt.show()
    plt.draw()
    print "exit"