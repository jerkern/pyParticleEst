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

class LSBEstJN(param_est.ParamEstimation):
        
    def create_initial_estimate(self, params, num):
        particles = numpy.empty(num, particle_lsb.ParticleLSB_JN)
        
        for k in range(len(particles)):
            particles[k] = particle_lsb.ParticleLSB()
        return particles

def do_test(num, nums, filter, y, e, z, k, steps):
    
    # How many steps forward in time should our simulation run
    C_theta = numpy.array([[ 0.0, 0.04, 0.044, 0.08],])

    # Create an array for our particles 
    if (filter == 'APF_JN'):
        ParamEstimator = LSBEstJN(u=None, y=y)
        filter='APF'
    else:
        ParamEstimator = LSBEst(u=None, y=y)
    resamplings = ParamEstimator.simulate(num, nums, res=0.67, filter=filter)

    svals = numpy.zeros((2, nums, steps+1))
    
    for i in range(steps+1):
        for j in range(nums):
            svals[0,j,i]=ParamEstimator.straj[j].traj[i].get_nonlin_state().ravel()
            svals[1,j,i]=25.0+C_theta.dot(ParamEstimator.straj[j].traj[i].kf.z)

    # Use average of trajectories
    svals_mean = numpy.mean(svals,1)

    theta = 25.0+C_theta.dot(z.reshape((4,-1)))
    sqr_err_eta = (svals_mean[0,:] - e[0,:])**2
    sqr_err_theta = (svals_mean[1,:] - theta)**2

    rmse_eta = numpy.sqrt(numpy.mean(sqr_err_eta))
    rmse_theta = numpy.sqrt(numpy.mean(sqr_err_theta))
    return (numpy.mean(rmse_eta), numpy.mean(rmse_theta), resamplings)
         
if __name__ == '__main__':
    num_range = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300, 500]
    steps = 100
    sims = 1000
    
    rmse_eta_pf = numpy.zeros((sims, len(num_range)))
    rmse_eta_apf = numpy.zeros((sims, len(num_range)))
    rmse_eta_jn = numpy.zeros((sims, len(num_range)))
    rmse_theta_pf = numpy.zeros((sims, len(num_range)))
    rmse_theta_apf = numpy.zeros((sims, len(num_range)))
    rmse_theta_jn = numpy.zeros((sims, len(num_range)))
    
    resamplings_pf = numpy.zeros((sims, len(num_range)))
    resamplings_apf = numpy.zeros((sims, len(num_range)))
    resamplings_jn = numpy.zeros((sims, len(num_range)))
    for i in range(len(num_range)):
        for k in range(sims):
            numpy.random.seed(k)
            (y, e, z) = particle_lsb.generate_dataset(steps)
            (rmse_eta_pf[k,i], rmse_theta_pf[k,i], resamplings_pf[k,i]) = do_test(num_range[i], 5, 'PF', y, e, z, k, steps)
            (rmse_eta_apf[k,i], rmse_theta_apf[k,i], resamplings_apf[k,i]) = do_test(num_range[i], 5, 'APF', y, e, z, k, steps)
            (rmse_eta_jn[k,i], rmse_theta_jn[k,i], resamplings_jn[k,i]) = do_test(num_range[i], 5, 'APF_JN', y, e, z, k, steps)
            
        print "N = %d, pf = %.4f/%.4f, apf=%.4f/%.4f, jn=%.4f/%.4f, resamplings=%.2f/%.2f/%.2f" % (num_range[i],
                                                    numpy.mean(rmse_eta_pf[:,i]),
                                                    numpy.mean(rmse_theta_pf[:,i]),
                                                    numpy.mean(rmse_eta_apf[:,i]),
                                                    numpy.mean(rmse_theta_apf[:,i]),
                                                    numpy.mean(rmse_eta_jn[:,i]),
                                                    numpy.mean(rmse_theta_jn[:,i]),
                                                    numpy.mean(resamplings_pf[:,i]),
                                                    numpy.mean(resamplings_apf[:,i]),
                                                    numpy.mean(resamplings_jn[:,i]))
    
