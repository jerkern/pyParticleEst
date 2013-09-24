#!/usr/bin/python

import PF
import PS
import numpy
import param_est
import copy

import matplotlib.pyplot as plt
from test.mixed_nlg.trans.particle_param_trans import ParticleParamTrans # Our model definition


class ParticleParamTransEst(param_est.ParamEstimation):
        
    def create_initial_estimate(self, params, num):
        particles = numpy.empty(num, ParticleParamTrans)
        
        for k in range(len(particles)):
            e = numpy.array([numpy.random.normal(0.0,1.0),])
            z = numpy.array([0.0,])
            P = numpy.eye(1)
            particles[k] = ParticleParamTrans(eta0=e, z0=z, P0=P, params=params)
        return (particles, z0, P0)

if __name__ == '__main__':
    
    num = 50
    nums=10
    
    theta_true = 0.1
    theta_guess = 0.3   
    R = numpy.array([[0.1]])
    Q = numpy.array([ 0.1, 0.1])
    e0 = numpy.array([-0.5, ])
    z0 = numpy.array([0.5, ])
    P0 = numpy.eye(1)
    

    # How many steps forward in time should our simulation run
    steps = 200
    sims = 5

    # Create arrays for storing some values for later plotting    
    vals = numpy.zeros((2, num+1, steps+1))
    yvec = numpy.zeros((1, steps))

    

    estimate = numpy.zeros((1,sims))
    
    plt.ion()
    fig1 = plt.figure()
    
    for k in range(sims):
        # Create reference
        e = numpy.random.normal(0.0, 1.0)
        z = numpy.random.normal(1.0, 1.0)
        for i in range(steps):
    
            # Extract linear states
            vals[0,num,i]=e
            # Extract non-linear state
            vals[1,num,i]=z
            
            
            e = e + theta_true * z + numpy.random.normal(0.0, 0.1)
            z = z + numpy.random.normal(0.0, 0.1)
            y = e
            yvec[0,i] = y
    
        # Store values for last time-step aswell    
        vals[0,num,steps]=e
        vals[1,num,steps]=z
    
        y_noise = yvec.T.tolist()
        for i in range(len(y_noise)):
            y_noise[i][0] += 0.0*numpy.random.normal(0.0,R)
        
        print "estimation start"
        
        # Create an array for our particles 
        ParamEstimator = ParticleParamTransEst(u=None, y=y_noise)
        ParamEstimator.set_params(numpy.array((theta_guess,)).reshape((-1,1)))
        param = ParamEstimator.maximize(param0=numpy.array((theta_guess,)), num_part=num, num_traj=nums, tol=10.0**-2)
        
        # Extract data from trajectories for plotting
        i=0
        for step in ParamEstimator.pt:
            pa = step.pa
            for j in range(pa.num):
                vals[0,j,i]=pa.part[j].eta[0,0]
                vals[1,j,i]=pa.part[j].kf.z.reshape(-1)
            i += 1
        plt.clf()
        x = numpy.asarray(range(steps+1))    
        for j in range(num):
            plt.plot(x,vals[0,j,:],'gx')
            plt.plot(x,vals[1,j,:],'rx')
        
        svals = numpy.zeros((3, nums, steps+1))
        
        for i in range(steps+1):
            for j in range(nums):
                svals[0,j,i]=ParamEstimator.straj[j].traj[i].get_nonlin_state().ravel()
                svals[1,j,i]=ParamEstimator.straj[j].traj[i].kf.z.ravel()
                
        for j in range(nums):
            plt.plot(range(steps+1),svals[0,j,:],'g-')
            plt.plot(range(steps+1),svals[1,j,:],'r-')
    
        #plt.plot(x[1:],yvec[0,:],'bx')
        plt.plot(x[1:],numpy.asarray(y_noise)[:,0],'b+')
                
            
        plt.plot(range(steps+1),vals[0,num,:],'go')
        plt.plot(range(steps+1),vals[1,num,:],'ro')
        fig1.show()
        plt.title("Param = %s" % theta_true)
        plt.show()
        plt.draw()
        
        print "maximization start"
        
        estimate[0,k] = param
    

    fig2 = plt.figure()
    plt.hist(estimate.T)
    plt.ioff()
    plt.show()
    plt.draw()
    print "exit"