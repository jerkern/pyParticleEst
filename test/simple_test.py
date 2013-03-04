#!/usr/bin/python

import PF
import PS
import numpy

import matplotlib.pyplot as plt
from simple_particle import SimpleParticle

if __name__ == '__main__':
    
    num = 50
    
    correct = SimpleParticle(numpy.array([1.0, 0.0]),2.5)
    
    particles = numpy.empty(num, type(correct))
    
    # Initialize particles
    for k in range(len(particles)):
        particles[k] = SimpleParticle(numpy.random.normal(numpy.array([[0.0], [0.0]])),
                                      numpy.random.uniform(2, 3))
    
    pa = PF.ParticleApproximation(particles=particles)
    
    pt = PF.ParticleTrajectory(pa,0.67)
    
    steps = 20
    
    mean = -numpy.hstack((-1.0*numpy.ones(steps/4), 1.0*numpy.ones(steps/2),-1.0*numpy.ones(steps/4) ))
    diff_vec = numpy.random.normal(mean, .1, steps)
    uvec = numpy.vstack((1.0-diff_vec/2, 1.0+diff_vec/2, numpy.zeros(steps)))
    
    
    #uvec = numpy.random.normal(0.0, 0.1, [2, steps])
    
    fig1 = plt.figure()
    
    vals = numpy.zeros((3, num+1, steps+1))
    
    yvec = []
    
    # Create reference
    for i in range(steps):
        
        vals[:2,num,i]=correct.kf.x_new.reshape(-1)
        vals[2,num,i]=correct.c
    
        #u = numpy.reshape(uvec[:,i],(-1,1))
        #u = numpy.vstack((u, numpy.array([[0,],])))
        u = uvec[:,i].reshape(-1,1)
        correct.update(u)
    
        y = correct.kf.C.dot(correct.kf.x_new)
        yvec.append(y)
        
    vals[:2,num,steps]=correct.kf.x_new.reshape(-1)
    vals[2,num,steps]=correct.c
    
        
    # Create PF trajectory
    for i in range(steps):
        
        #u = numpy.reshape(uvec[:,i],(-1,1))
        #u = numpy.vstack((u, numpy.array([[0,],])))
        u = uvec[:,i].reshape(-1,1)
        tmp = numpy.random.normal((0.0,0.0,0.0),(0.1,0.1,0.0000000001)).reshape((-1,1))
        pt.update(u+tmp)
        
        pt.measure(yvec[i]+numpy.random.normal(0.0,1.))
        
    # Perform smoothing
    nums = 10
    straj = PS.do_smoothing(pt, nums)
    straj = PS.do_rb_smoothing(straj)
    
    # Extract data from trajectories for plotting
    i=0
    for step in pt:
        pa = step.pa
        for j in range(pa.num):
            vals[:2,j,i]=pa.part[j].kf.x_new.reshape(-1)
            vals[2,j,i]=pa.part[j].c
        i += 1
        
    for j in range(num):
        plt.plot(range(steps+1),vals[0,j,:],'g.')
        plt.plot(range(steps+1),vals[1,j,:],'r.')
        plt.plot(range(steps+1),vals[2,j,:],'k.')
    
    
    svals = numpy.zeros((3, nums, steps+1))
    
    for i in range(steps+1):
        for j in range(nums):
            svals[:2,j,i]=straj[j].traj[i].kf.x_new.ravel()
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
    
