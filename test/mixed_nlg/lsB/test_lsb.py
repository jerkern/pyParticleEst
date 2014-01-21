'''
Created on Nov 11, 2013

@author: ajn
'''

import numpy
import math
import matplotlib.pyplot as plt
import test.mixed_nlg.lsB.particle_lsb as particle_lsb
import param_est

class LS2Est(param_est.ParamEstimation):
        
    def create_initial_estimate(self, params, num):
        particles = numpy.empty(num, particle_lsb.ParticleLSB)
        
        for k in range(len(particles)):
            particles[k] = particle_lsb.ParticleLSB()
        return particles

if __name__ == '__main__':
    
    num = 300
    nums = 10
    
    theta_true = numpy.array((1.0, 1.0, 0.3, 0.968, 0.315))
   

    # How many steps forward in time should our simulation run
    steps = 100
    sims = 1

    # Create arrays for storing some values for later plotting    
    vals = numpy.zeros((2, num+1, steps+1))

    estimate = numpy.zeros((5,sims))
    
    plt.ion()
    
    for k in range(sims):
        print k
        theta_guess = numpy.array((numpy.random.uniform(0.0, 2.0),
                                   numpy.random.uniform(0.0, 2.0),
                                   numpy.random.uniform(0.0, 0.6),
                                   numpy.random.uniform(0.0, 1.0),
                                   numpy.random.uniform(0.0, math.pi/2.0)))
        
        #theta_guess = numpy.copy(theta_true)
        
        # Create reference
        numpy.random.seed(86)
        (y, e, z) = particle_lsb.generate_dataset(steps)
        # Store values for last time-step aswell    
    
        print "estimation start"
        
        x = numpy.asarray(range(steps+1))
       
        # Create an array for our particles 
        ParamEstimator = LS2Est(u=None, y=y)
        ParamEstimator.simulate(num, nums, False)

        
        svals = numpy.zeros((5, nums, steps+1))
        vals = numpy.zeros((5, num, steps+1))
 
        fig3 = plt.figure()
        fig4 = plt.figure()
        fig5 = plt.figure()
        fig6 = plt.figure()
        fig7 = plt.figure()
        
        for i in range(steps+1):
            for j in range(nums):
                svals[0,j,i]=ParamEstimator.straj[j].traj[i].get_nonlin_state().ravel()
                svals[1:,j,i]=ParamEstimator.straj[j].traj[i].kf.z.ravel()
            for j in range(num):
                vals[0,j,i]=ParamEstimator.pt[i].pa.part[j].get_nonlin_state().ravel()
                vals[1:,j,i]=ParamEstimator.pt[i].pa.part[j].kf.z.ravel()
                
        svals_mean = numpy.mean(svals,1)
        plt.figure(fig3.number)
        plt.clf()

        for j in range(num):   
            plt.plot(range(steps+1),vals[0,j,:],'m.')
        for j in range(nums):
            plt.plot(range(steps+1),svals[0,j,:],'g-')
        plt.plot(range(steps+1),svals_mean[0,:],'b-')
        plt.plot(x[:-1], e.T,'r-')
        #plt.plot(x[:-1], e,'r-')
        fig3.show()
        
        plt.figure(fig4.number)
        plt.clf()
        # TODO, does these smoothed estimates really look ok??
        for j in range(num):   
            plt.plot(range(steps+1),vals[1,j,:],'m.')
        for j in range(nums):
            plt.plot(range(steps+1),svals[1,j,:],'g-')
        plt.plot(range(steps+1),svals_mean[1,:],'b-')
        plt.plot(x[:-1], z[0,:],'r-')
        #plt.plot(x[:-1], e,'r-')
        fig4.show()
        
        plt.figure(fig5.number)
        plt.clf()
        # TODO, does these smoothed estimates really look ok??
        for j in range(num):   
            plt.plot(range(steps+1),vals[2,j,:],'m.')
        for j in range(nums):
            plt.plot(range(steps+1),svals[2,j,:],'g-')
        plt.plot(range(steps+1),svals_mean[2,:],'b-')
        plt.plot(x[:-1], z[1,:],'r-')
        #plt.plot(x[:-1], e,'r-')
        fig5.show()
        
        plt.figure(fig6.number)
        plt.clf()
        # TODO, does these smoothed estimates really look ok??
        for j in range(num):   
            plt.plot(range(steps+1),vals[3,j,:],'m.')
        for j in range(nums):
            plt.plot(range(steps+1),svals[3,j,:],'g-')

        plt.plot(range(steps+1),svals_mean[3,:],'b-')
        plt.plot(x[:-1], z[2,:],'r-')
        #plt.plot(x[:-1], e,'r-')
        fig6.show()

        plt.figure(fig7.number)
        plt.clf()
        # TODO, does these smoothed estimates really look ok??
        for j in range(num):   
            plt.plot(range(steps+1),vals[4,j,:],'m.')
        for j in range(nums):
            plt.plot(range(steps+1),svals[4,j,:],'g-')

        plt.plot(range(steps+1),svals_mean[4,:],'b-')
        plt.plot(x[:-1], z[3,:],'r-')
        #plt.plot(x[:-1], e,'r-')
        fig7.show()
 
        plt.draw()
    
    plt.ioff()
    plt.show()
    plt.draw()
    print "exit"