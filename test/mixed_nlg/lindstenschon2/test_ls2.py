'''
Created on Nov 11, 2013

@author: ajn
'''

import numpy
import math
import matplotlib.pyplot as plt
import test.mixed_nlg.lindstenschon2.particle_ls2 as particle_ls2
import param_est

class LS2Est(param_est.ParamEstimation):
        
    def create_initial_estimate(self, params, num):
        particles = numpy.empty(num, particle_ls2.ParticleLS2)
        
        for k in range(len(particles)):
            e = numpy.array([numpy.random.normal(0.0,1.0),]).reshape((-1,1))
            z0 = numpy.zeros((3,1))
            P0 = 0.00001*numpy.eye(3,3)
            particles[k] = particle_ls2.ParticleLS2(eta0=e, z0=z0, P0=P0, params=params)
        return particles

if __name__ == '__main__':
    
    num = 50
    nums = 5
    
    theta_true = numpy.array((1.0, 1.0, 0.3, 0.968, 0.315))
   

    # How many steps forward in time should our simulation run
    steps = 200
    sims = 1

    # Create arrays for storing some values for later plotting    
    vals = numpy.zeros((2, num+1, steps+1))

    estimate = numpy.zeros((5,sims))
    
    plt.ion()
    fig1 = plt.figure()
    fig2 = plt.figure()
    
    
    max_iter = 1000
    
    for k in range(sims):
        print k
        theta_guess = numpy.array((numpy.random.uniform(0.0, 2.0),
                                   numpy.random.uniform(0.0, 2.0),
                                   numpy.random.uniform(0.0, 0.6),
                                   numpy.random.uniform(0.0, 1.0),
                                   numpy.random.uniform(0.0, math.pi/2.0)))
        
        #theta_guess = numpy.copy(theta_true)
        
        # Create reference
        (y, e, z) = particle_ls2.generate_dataset(theta_true, steps)
        # Store values for last time-step aswell    
    
        print "estimation start"
        
        plt.figure(fig1.number)
        plt.clf()
        x = numpy.asarray(range(steps+1))
        plt.plot(x[1:],numpy.asarray(y)[:,:],'.')
        fig1.show()
        plt.draw()
        
        params_it = numpy.zeros((max_iter, len(theta_guess)))
        Q_it = numpy.zeros((max_iter))
        it = 0
        
        def callback(params, Q):
            global it
            params_it[it,:] = params
            Q_it[it] = Q
            it = it+1
            plt.figure(fig2.number)
            plt.clf()
            plt.plot(range(it), params_it[:it, 0], 'b-')
            plt.plot((0.0, it), (theta_true[0], theta_true[0]), 'b--')
            plt.plot(range(it), params_it[:it, 1], 'r-')
            plt.plot((0.0, it), (theta_true[1], theta_true[1]), 'r--')
            plt.plot(range(it), params_it[:it, 2], 'g-')
            plt.plot((0.0, it), (theta_true[2], theta_true[2]), 'g--')
            plt.plot(range(it), params_it[:it, 3], 'c-')
            plt.plot((0.0, it), (theta_true[3], theta_true[3]), 'c--')
            plt.plot(range(it), params_it[:it, 4], 'k-')
            plt.plot((0.0, it), (theta_true[4], theta_true[4]), 'k--')
            plt.show()
            plt.draw()
            return
        
        # Create an array for our particles 
        ParamEstimator = LS2Est(u=None, y=y)
        ParamEstimator.set_params(theta_guess)
        #ParamEstimator.simulate(num, nums, False)

        param = ParamEstimator.maximize(param0=theta_guess, num_part=num, num_traj=nums, max_iter=max_iter,
                                        update_before_predict=False, callback=callback)
        
        svals = numpy.zeros((4, nums, steps+1))
 
        fig3 = plt.figure()
        fig4 = plt.figure()
        fig5 = plt.figure()
        fig6 = plt.figure()
 
        
        for i in range(steps+1):
            for j in range(nums):
                svals[0,j,i]=ParamEstimator.straj[j].traj[i].get_nonlin_state().ravel()
                svals[1:,j,i]=ParamEstimator.straj[j].traj[i].kf.z.ravel()
                
        plt.figure(fig3.number)
        plt.clf()
        # TODO, does these smoothed estimates really look ok??
        for j in range(nums):
            plt.plot(range(steps+1),svals[0,j,:],'g-')
            #plt.plot(range(steps+1),svals[1,j,:],'r-')
        plt.plot(x[:-1], e.T,'rx')
        #plt.plot(x[:-1], e,'r-')
        fig3.show()
        
        plt.figure(fig4.number)
        plt.clf()
        # TODO, does these smoothed estimates really look ok??
        for j in range(nums):
            plt.plot(range(steps+1),svals[1,j,:],'g-')
            #plt.plot(range(steps+1),svals[1,j,:],'r-')
        plt.plot(x[:-1], z[0,:],'rx')
        #plt.plot(x[:-1], e,'r-')
        fig4.show()
        
        plt.figure(fig5.number)
        plt.clf()
        # TODO, does these smoothed estimates really look ok??
        for j in range(nums):
            plt.plot(range(steps+1),svals[2,j,:],'g-')
            #plt.plot(range(steps+1),svals[1,j,:],'r-')
        plt.plot(x[:-1], z[1,:],'rx')
        #plt.plot(x[:-1], e,'r-')
        fig5.show()
        
        plt.figure(fig6.number)
        plt.clf()
        # TODO, does these smoothed estimates really look ok??
        for j in range(nums):
            plt.plot(range(steps+1),svals[2,j,:],'g-')
            #plt.plot(range(steps+1),svals[1,j,:],'r-')
        plt.plot(x[:-1], z[2,:],'rx')
        #plt.plot(x[:-1], e,'r-')
        fig6.show()

    
        plt.draw()

        
        print "maximization start"
        
        estimate[:,k] = param
        
#        plt.figure(fig2.number)
#        plt.clf()
#        plt.hist(estimate[:,:(k+1)].T)
#        fig2.show()
#        plt.show()
#        plt.draw()

        

    
#    plt.hist(estimate.T)
    plt.ioff()
    plt.show()
    plt.draw()
    print "exit"