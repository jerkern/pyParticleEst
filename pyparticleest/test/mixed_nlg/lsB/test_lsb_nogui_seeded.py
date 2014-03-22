'''
Created on Nov 11, 2013

@author: ajn
'''

import numpy
import matplotlib.pyplot as plt
import pyparticleest.test.mixed_nlg.lsB.particle_lsb as particle_lsb
import pyparticleest.param_est as param_est

class LSBEst(param_est.ParamEstimation):
        
    def create_initial_estimate(self, params, num):
        particles = numpy.empty(num, particle_lsb.ParticleLSB)
        
        for k in range(len(particles)):
            particles[k] = particle_lsb.ParticleLSB()
        return particles

if __name__ == '__main__':
    
    num = 300
    nums = 5

    # How many steps forward in time should our simulation run
    steps = 100
    sims = 1000

    # Create arrays for storing some values for later plotting    
    vals = numpy.zeros((2, num+1, steps+1))

    estimate = numpy.zeros((5,sims))
    
    plt.ion()
    
    sqr_err_eta = numpy.zeros((sims, steps+1))
    sqr_err_theta = numpy.zeros((sims, steps+1))
    sqr_err_eta_single = numpy.zeros((sims, nums, steps))
    sqr_err_theta_single = numpy.zeros((sims, nums, steps))
    t = numpy.asarray(range(steps+1))
    C_theta = numpy.array([[ 0.0, 0.04, 0.044, 0.08],])
    for k in range(sims):
        # Create reference
        numpy.random.seed(k)
        (y, e, z) = particle_lsb.generate_dataset(steps)

        model = particle_lsb.ParticleLSB()
        # Create an array for our particles 
        ParamEstimator = param_est.ParamEstimation(model=model, u=None, y=y)
        ParamEstimator.simulate(num, nums, res=0.67, filter='PF', smoother='normal')

        svals = numpy.zeros((2, nums, steps+1))
        
        for i in range(steps+1):
            (xil, zl, Pl) = model.get_states(ParamEstimator.straj.straj[i])
            svals[0,:,i] = numpy.vstack(xil).ravel()
            svals[1,:,i] = 25.0+C_theta.dot(numpy.hstack(zl)).ravel()
        
        # Use average of trajectories
        svals_mean = numpy.mean(svals,1)

        theta = 25.0+C_theta.dot(z.reshape((4,-1)))
        sqr_err_eta[k,:] = (svals_mean[0,:] - e[0,:])**2
        sqr_err_theta[k,:] = (svals_mean[1,:] - theta)**2

#        for i in range(steps):
#            theta = 25.0+C_theta.dot(z[:,i].reshape((-1,1)))
#            sqr_err_eta[k,i] = (svals_mean[0,i] - e[0,i])**2
#            sqr_err_theta[k,i] = (svals_mean[1,i] - theta)**2
#            for j in range(nums):
#                sqr_err_eta_single[k,j,i] = (svals[0,j,i] - e[0,i])**2
#                sqr_err_theta_single[k,j,i] = (svals[1,j,i] - theta)**2
        
        
        rmse_eta = numpy.sqrt(numpy.mean(sqr_err_eta[k,:]))
        rmse_theta = numpy.sqrt(numpy.mean(sqr_err_theta[k,:]))
        print "%d %f %f" % (k, numpy.mean(rmse_eta), numpy.mean(rmse_theta))
         
    
