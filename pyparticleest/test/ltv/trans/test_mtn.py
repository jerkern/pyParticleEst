'''
Created on Sep 17, 2013

@author: ajn
'''
import param_est
import numpy
import matplotlib.pyplot as plt

from test.ltv.trans.particle_param_trans import ParticleParamTrans as PartModel # Our model definition
import test.ltv.trans.particle_param_trans as particle_param_trans

class ParticleParamTransEst(param_est.ParamEstimation):
        
    def create_initial_estimate(self, params, num):
        particles = numpy.empty(num, PartModel)
        
        for k in range(len(particles)):
            particles[k] = PartModel(z0=z0, P0=P0, params=params)
        return particles

z0 = numpy.array([0.0, 0.0]).reshape((-1,1))
P0 = 100*numpy.eye(2)    
    
if __name__ == '__main__':
    
    num = 1
    
    theta_true = 0.1
    theta_guess = 0.1

    steps = 200
    
    num_tests=200
    num_guesses=30
    
    
    guesses = numpy.linspace(0.01, 0.30, num_guesses)
    mean = numpy.zeros((num_guesses,))
    stdd = numpy.zeros((num_guesses,))
    
    for j in range(len(guesses)):
    
        param_max = numpy.zeros((num_tests,))
        for i in range(num_tests):
        
            # Create a reference which we will try to estimate using a RBPS
            (ylist, states) = particle_param_trans.generate_reference(z0, P0, theta_true, steps)
            correct = PartModel(z0=z0,P0=P0, params=(theta_true,))
        
            # Create an array for our particles 
            
            ParamEstimator = ParticleParamTransEst(u=None, y=ylist)
            ParamEstimator.set_params(numpy.array((guesses[j],)).reshape((-1,1)))
            #ParamEstimator.simulate(num_part=num, num_traj=nums)
            print "maximization start (%d)" % i
            (param, Q) = ParamEstimator.maximize(param0=numpy.array((guesses[j],)), num_part=num, num_traj=1, max_iter=1)
            param_max[i] = param
            
            mean[j] = numpy.average(param_max)
            stdd[j] = numpy.std(param_max)
            
#            bins = numpy.linspace(0.05, 0.15, 1000)
#            plt.hist(param_max, bins=bins)
#            print "mean: %s" % 
#            print "stdd: %s" % 
            
    plt.plot(guesses,guesses, 'r')
    plt.plot(guesses, theta_true*numpy.ones((len(guesses),)), 'g')
    plt.plot(guesses, mean, 'b')
    plt.plot(guesses, mean-stdd, 'b--')
    plt.plot(guesses, mean+stdd, 'b--')
    
    bins = numpy.linspace(0.05, 0.15, 1000)
    plt.hist(param_max, bins=bins)
    plt.show()