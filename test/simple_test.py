#!/usr/bin/python

import PF
import PS
import numpy

import matplotlib.pyplot as plt
from simple_particle import SimpleParticle # Our model definition

if __name__ == '__main__':
    
    num = 50
    
    # Create a reference which we will try to estimate using a RBPS
    correct = SimpleParticle(numpy.array([1.0, -0.5]),2.5)
    
    # Create an array for our particles 
    particles = numpy.empty(num, type(correct))
    
    
    z0 = numpy.array([[0.0], [0.0]])
    # Initialize particles
    for k in range(len(particles)):
        # Let the initial value of the non-linear state be U(2,3)
        particles[k] = SimpleParticle(z0, numpy.random.uniform(2, 3))
    
    # Create a particle approximation object from our particles
    pa = PF.ParticleApproximation(particles=particles)
    
    # Initialise a particle filter with our particle approximation of the initial state,
    # set the resampling threshold to 0.67 (effective particles / total particles )
    pt = PF.ParticleTrajectory(pa,0.67)
    
    # How many steps forward in time should our simulation run
    steps = 20
    
    # Create a random input vector to drive our "correct state"
    mean = -numpy.hstack((-1.0*numpy.ones(steps/4), 1.0*numpy.ones(steps/2),-1.0*numpy.ones(steps/4) ))
    diff_vec = numpy.random.normal(mean, .1, steps)
    uvec = numpy.vstack((1.0-diff_vec/2, 1.0+diff_vec/2))
    

    
    fig1 = plt.figure()

    # Create arrays for storing some values for later plotting    
    vals = numpy.zeros((3, num+1, steps+1))
    yvec = []
    
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
        y = correct.kf.C.dot(correct.kf.z)
        yvec.append(y)
    
    # Store values for last time-step aswell    
    vals[:2,num,steps]=correct.kf.z.reshape(-1)
    vals[2,num,steps]=correct.eta[0,0]
    
        
    # Run particle filter using the above generated data
    for i in range(steps):
        
        
        u = uvec[:,i].reshape(-1,1)
        tmp = numpy.random.normal((0.0,0.0),(0.1,0.1)).reshape((-1,1))
        
        # Run PF using noise corrupted input signal
        pt.update(u+tmp)
        
        # Use noise corrupted measurements
        pt.measure(yvec[i]+numpy.random.normal(0.0,1.))
        
    # Use the filtered estimates above to created smoothed estimates
    nums = 10 # Number of backward trajectories to generate
    straj = PS.do_smoothing(pt, nums)   # Do sampled smoothing
    for st in straj:
        st.constrained_smoothing(z0=z0,
                                 P0=100000*numpy.diag([1.0, 1.0]))
    
    # Extract data from trajectories for plotting
    i=0
    for step in pt:
        pa = step.pa
        for j in range(pa.num):
            vals[:2,j,i]=pa.part[j].kf.z.reshape(-1)
            vals[2,j,i]=pa.part[j].eta[0,0]
        i += 1
        
    for j in range(num):
        plt.plot(range(steps+1),vals[0,j,:],'g.')
        plt.plot(range(steps+1),vals[1,j,:],'r.')
        plt.plot(range(steps+1),vals[2,j,:],'k.')
    
    
    svals = numpy.zeros((3, nums, steps+1))
    
    for i in range(steps+1):
        for j in range(nums):
            svals[:2,j,i]=straj[j].traj[i].kf.z.ravel()
            svals[2,j,i]=straj[j].traj[i].get_nonlin_state().ravel()
            
    for j in range(nums):
        plt.plot(range(steps+1),svals[0,j,:],'g-')
        plt.plot(range(steps+1),svals[1,j,:],'r-')
        plt.plot(range(steps+1),svals[2,j,:],'k-')
            
        
        
    plt.plot(range(steps+1),vals[0,num,:],'go')
    plt.plot(range(steps+1),vals[1,num,:],'ro')
    plt.plot(range(steps+1),vals[2,num,:],'ko')
    fig1.show()
    plt.show()
    
