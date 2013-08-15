#!/usr/bin/python

import PF
import PS
import numpy
import param_est
import copy

import matplotlib.pyplot as plt
from test.mixed_nlg.output.particle_param_output import ParticleParamOutput # Our model definition

class ParticleParamOutputEst(param_est.ParamEstimation):
        
    def create_initial_estimate(self, params, num):
        particles = numpy.empty(num, ParticleParamOutput)
        z0 = numpy.array([[0.0], [0.0]])
        P0 = 1000.0*numpy.eye(2)
        for k in range(len(particles)):
            particles[k] = ParticleParamOutput(x0=z0, P0=P0, params=params)
        return (particles, z0, P0)

if __name__ == '__main__':
    
    num = 1
    
    c_true = 2.5
    
    # Create a reference which we will try to estimate using a RBPS
    correct = ParticleParamOutput(x0=numpy.array([1.0, -0.5]),P0=numpy.eye(2),
                                  params=(c_true,))

    # How many steps forward in time should our simulation run
    steps = 20
    
    # Create a random input vector to drive our "correct state"
    mean = -numpy.hstack((-1.0*numpy.ones(steps/4), 1.0*numpy.ones(steps/2),-1.0*numpy.ones(steps/4) ))
    diff_vec = numpy.random.normal(mean, .1, steps)
    uvec = numpy.vstack((1.0-diff_vec/2, 1.0+diff_vec/2))

    # Create arrays for storing some values for later plotting    
    vals = numpy.zeros((3, num+1, steps+1))
    yvec = numpy.zeros((1, steps))


    # Create reference
    for i in range(steps):
        
        # Extract linear states
        vals[:2,num,i]=correct.kf.z.reshape(-1)
        # Extract non-linear state
        vals[2,num,i]=correct.eta[0,0]

        # Extract u for this time-step    
        u = uvec[:,i].reshape(-1,1)
        # Drive the correct particle using the true input
        noise = correct.sample_process_noise(u)
        correct.update(u, 0.0*noise)
    
        # use the correct particle to generate the true measurement
        yvec[0,i] = correct.kf.C.dot(correct.kf.z).ravel()

    
    # Store values for last time-step aswell    
    vals[:2,num,steps]=correct.kf.z.reshape(-1)
    vals[2,num,steps]=correct.eta[0,0]

    u_noise = numpy.copy(uvec)
    y_noise = copy.deepcopy(yvec)
    for i in range(y_noise.shape[1]):
        u_noise[:,i] += numpy.random.normal((0.0,0.0),(0.1,0.1)).ravel()
        y_noise[:,i] += numpy.random.normal(0.0,1.0)
    
    print "estimation start"
    
    # Create an array for our particles 
    ParamEstimator = ParticleParamOutputEst(u_noise, y_noise)

    nums=1

    plt.ion()
    fig1 = plt.figure()
    param_steps = 25
    param_vals = numpy.linspace(0.5, 4.5, param_steps)
    logpy = numpy.zeros((param_steps,))
    for i in range(param_steps):
        fig1.clf()
        ParamEstimator.set_params(numpy.array((param_vals[i],)).reshape((-1,1)))
        ParamEstimator.simulate(num_part=num, num_traj=nums)
        
        logpy[i] = ParamEstimator.eval_logp_y()

        # Extract data from trajectories for plotting
        i=0
        for step in ParamEstimator.pt:
            pa = step.pa
            for j in range(pa.num):
                vals[:2,j,i]=pa.part[j].kf.z.reshape(-1)
                vals[2,j,i]=pa.part[j].eta[0,0]
            i += 1
        
        x = numpy.asarray(range(steps+1))    
        for j in range(num):
            plt.plot(x,vals[0,j,:],'gx')
            plt.plot(x,vals[1,j,:],'rx')
        
        svals = numpy.zeros((3, nums, steps+1))
        
        for i in range(steps+1):
            for j in range(nums):
                svals[:2,j,i]=ParamEstimator.straj[j].traj[i].kf.z.ravel()
                svals[2,j,i]=ParamEstimator.straj[j].traj[i].get_nonlin_state().ravel()
                
        for j in range(nums):
            plt.plot(range(steps+1),svals[0,j,:],'g-')
            plt.plot(range(steps+1),svals[1,j,:],'r-')
            plt.plot(x,0.0*x+ParamEstimator.params[0],'k-')
                
        y_est = correct.kf.C.dot(svals[:2,0,:])
        #plt.plot(x[1:],yvec[0,:],'bx')
        plt.plot(x[1:],y_noise[0,:]/c_true,'b+')
                
            
        plt.plot(range(steps+1),vals[0,num,:],'go')
        plt.plot(range(steps+1),vals[1,num,:],'ro')
        fig1.show()
        plt.show()
        plt.draw()
    
    plt.ioff()
    
    fig2 = plt.figure()
    plt.plot(param_vals, logpy)
    plt.show()