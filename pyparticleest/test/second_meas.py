'''
Created on Sep 20, 2013

@author: ajn
'''

#!/usr/bin/python

import pyparticleest.pf as pf
import numpy

import matplotlib.pyplot as plt
from pyparticleest.test.mixed_nlg.trans.particle_param_trans import ParticleParamTrans # Our model definition

if __name__ == '__main__':
    
    num = 100
    
    theta_true = 0.1   
    R = numpy.array([[0.1]])
    Q = numpy.array([ 0.1, 0.1])
    e0 = numpy.array([0.0, ])
    z0 = numpy.array([0.0, ])
    P0 = numpy.eye(1.0)
    

    # How many steps forward in time should our simulation run
    steps = 200
    

    # Create arrays for storing some values for later plotting    
    vals = numpy.zeros((2, num+1, steps+1))
    yvec = numpy.zeros((1, steps))

    e = numpy.copy(e0)
    z = numpy.copy(z0)
    # Create reference
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
    
    nums=1
    
    particles = numpy.empty(num, ParticleParamTrans)
    params = numpy.array((theta_true,))
    for k in range(num):
            particles[k] = ParticleParamTrans(eta0=e0, z0=z0, P0=P0, params=params)
    
    pa = pf.ParticleApproximation(particles=particles)
    
    # Initialise a particle filter with our particle approximation of the initial state,
    # set the resampling threshold to 0.67 (effective particles / total particles )
    pt = pf.ParticleTrajectory(pa,0.67)
  
    print "filtering start"
    
    fig1 = plt.figure()

    for i in range(steps):
        pt.update(u=None)
        pt.measure(y_noise[i])

    # Extract data from trajectories for plotting
    i=0
    for step in pt:
        pa = step.pa
        for j in range(pa.num):
            vals[0,j,i]=pa.part[j].eta[0,0]
            vals[1,j,i]=pa.part[j].kf.z.reshape(-1)
        i += 1
        
    x = numpy.asarray(range(steps+1))    
    for j in range(num):
        plt.plot(x,vals[0,j,:],'gx')
        plt.plot(x,vals[1,j,:],'rx')
    
    plt.plot(x[1:],numpy.asarray(y_noise)[:,0],'b+')
            
    plt.plot(range(steps+1),vals[0,num,:],'g-')
    plt.plot(range(steps+1),vals[1,num,:],'r-')
    fig1.show()
    plt.show()