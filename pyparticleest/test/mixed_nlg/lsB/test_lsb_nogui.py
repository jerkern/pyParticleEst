'''
Created on Nov 11, 2013

@author: ajn
'''

import numpy
import math
import matplotlib.pyplot as plt
import test.mixed_nlg.lsB.particle_lsb as particle_lsb
import param_est

class LSBEst(param_est.ParamEstimation):
        
    def create_initial_estimate(self, params, num):
        particles = numpy.empty(num, particle_lsb.ParticleLSB)
        
        for k in range(len(particles)):
            particles[k] = particle_lsb.ParticleLSB()
        return particles

if __name__ == '__main__':
    
    num = 300
    nums = 10

    # How many steps forward in time should our simulation run
    steps = 100
    sims = 1000

    # Create arrays for storing some values for later plotting    
    vals = numpy.zeros((2, num+1, steps+1))

    estimate = numpy.zeros((5,sims))
    
    plt.ion()
    
    sqr_err_eta = numpy.zeros((sims, steps))
    sqr_err_theta = numpy.zeros((sims, steps))
    sqr_err_eta_single = numpy.zeros((sims, nums, steps))
    sqr_err_theta_single = numpy.zeros((sims, nums, steps))
    t = numpy.asarray(range(steps+1))
    C_theta = numpy.array([[ 0.0, 0.04, 0.044, 0.08],])
    for k in range(sims):
        # Create reference
        (y, e, z) = particle_lsb.generate_dataset(steps)

        # Create an array for our particles 
        ParamEstimator = LSBEst(u=None, y=y)
        ParamEstimator.simulate(num, nums, False)

        svals = numpy.zeros((2, nums, steps+1))
        
        for i in range(steps+1):
            for j in range(nums):
                svals[0,j,i]=ParamEstimator.straj[j].traj[i].get_nonlin_state().ravel()
                svals[1,j,i]=25.0+C_theta.dot(ParamEstimator.straj[j].traj[i].kf.z)

        # Use average of trajectories
        svals_mean = numpy.mean(svals,1)

        for i in range(steps):
            theta = 25.0+C_theta.dot(z[:,i].reshape((-1,1)))
            sqr_err_eta[k,i] = (svals_mean[0,i] - e[0,i])**2
            sqr_err_theta[k,i] = (svals_mean[1,i] - theta)**2
            for j in range(nums):
                sqr_err_eta_single[k,j,i] = (svals[0,j,i] - e[0,i])**2
                sqr_err_theta_single[k,j,i] = (svals[1,j,i] - theta)**2
            
        mse_eta = numpy.mean(sqr_err_eta[:k+1,:], 0)
        mse_theta = numpy.mean(sqr_err_theta[:k+1,:], 0)
        rmse_eta = numpy.sqrt(mse_eta)
        rmse_theta = numpy.sqrt(mse_theta)
        print "RMSE (%d): eta=%f theta=%f" % (k, numpy.mean(rmse_eta), numpy.mean(rmse_theta))
         
    
    mse_eta = numpy.mean(sqr_err_eta, 0)
    mse_theta = numpy.mean(sqr_err_theta, 0)
    rmse_eta = numpy.sqrt(mse_eta)
    rmse_theta = numpy.sqrt(mse_theta)
    
    print "RMSE: eta=%f theta=%f" % (numpy.mean(rmse_eta), numpy.mean(rmse_theta))   
    print "RMSE, individual: eta=%f theta=%f" % (numpy.sqrt(numpy.mean(sqr_err_eta_single.ravel())),
                                                 numpy.sqrt(numpy.mean(sqr_err_theta_single.ravel())))
    print "RMSE, all: eta=%f, theta=%f" % (numpy.sqrt(numpy.mean(sqr_err_eta.ravel())),
                                                 numpy.sqrt(numpy.mean(sqr_err_theta.ravel())))
