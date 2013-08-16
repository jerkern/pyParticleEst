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
        e0 = numpy.array([0.0,])
        z0 = numpy.array([0.0,])
        P0 = numpy.eye(1)
        for k in range(len(particles)):
            particles[k] = ParticleParamTrans(eta0=e0, z0=z0, P0=P0, params=params)
        return (particles, z0, P0)

if __name__ == '__main__':
    
    num = 50
    
    theta_true = 0.1
    R = numpy.array([[0.1]])
    Q = numpy.array([ 0.1, 0.1])
    e0 = numpy.array([0.0, ])
    z0 = numpy.array([0.0, ])
    P0 = numpy.eye(1)
    
    # Create a reference which we will try to estimate using a RBPS
    correct = ParticleParamTrans(eta0=e0, z0=z0,P0=P0, params=(theta_true,))

    # How many steps forward in time should our simulation run
    steps = 40
    

    # Create arrays for storing some values for later plotting    
    vals = numpy.zeros((2, num+1, steps+1))
    yvec = numpy.zeros((1, steps))


    # Create reference
    for i in range(steps):
        
        # Extract linear states
        vals[0,num,i]=correct.kf.z.reshape(-1)
        # Extract non-linear state
        vals[1,num,i]=correct.eta[0,0]

        # Drive the correct particle using the true input
        noise = correct.sample_process_noise()
        correct.update(u=None, noise=noise)
        correct.kf.z += numpy.random.normal(0.0, Q[1])
        # use the correct particle to generate the true measurement
        yvec[0,i] = correct.eta[0,0]

    
    # Store values for last time-step aswell    
    vals[0,num,steps]=correct.kf.z.reshape(-1)
    vals[1,num,steps]=correct.eta[0,0]

    y_noise = yvec.T.tolist()
    for i in range(len(y_noise)):
        y_noise[i][0] += numpy.random.normal(0.0,R)
    
    print "estimation start"
    
    # Create an array for our particles 
    ParamEstimator = ParticleParamTransEst(u=None, y=y_noise)

    nums=5

    plt.ion()
    fig1 = plt.figure()
    param_steps = 20
    param_vals = numpy.linspace(0.0, 1.0, param_steps)
    logpy = numpy.zeros((param_steps,))
    logpxnext = numpy.zeros((param_steps,))
    for k in range(param_steps):
        fig1.clf()
        ParamEstimator.set_params(numpy.array((param_vals[k],)).reshape((-1,1)))
        ParamEstimator.simulate(num_part=num, num_traj=nums)
        
        logpy[k] = ParamEstimator.eval_logp_y()
        logpxnext[k] = ParamEstimator.eval_logp_xnext()

        # Extract data from trajectories for plotting
        i=0
        for step in ParamEstimator.pt:
            pa = step.pa
            for j in range(pa.num):
                vals[0,j,i]=pa.part[j].kf.z.reshape(-1)
                vals[1,j,i]=pa.part[j].eta[0,0]
            i += 1
        
        x = numpy.asarray(range(steps+1))    
        for j in range(num):
            plt.plot(x,vals[0,j,:],'gx')
            plt.plot(x,vals[1,j,:],'rx')
        
        svals = numpy.zeros((3, nums, steps+1))
        
        for i in range(steps+1):
            for j in range(nums):
                svals[0,j,i]=ParamEstimator.straj[j].traj[i].kf.z.ravel()
                svals[1,j,i]=ParamEstimator.straj[j].traj[i].get_nonlin_state().ravel()
                
        for j in range(nums):
            plt.plot(range(steps+1),svals[0,j,:],'g-')
            plt.plot(range(steps+1),svals[1,j,:],'r-')

        #plt.plot(x[1:],yvec[0,:],'bx')
        plt.plot(x[1:],numpy.asarray(y_noise)[:,0],'b+')
                
            
        plt.plot(range(steps+1),vals[0,num,:],'go')
        plt.plot(range(steps+1),vals[1,num,:],'ro')
        fig1.show()
        plt.title("Param = %s" % param_vals[k])
        plt.show()
        plt.draw()
    
    plt.ioff()
    
    fig2 = plt.figure()
    plt.plot(param_vals, logpy,'g')
    plt.plot(param_vals, logpxnext,'b')
    plt.show()