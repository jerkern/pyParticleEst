#!/usr/bin/python

import numpy
import pyparticleest.param_est as param_est

import matplotlib.pyplot as plt
import pyparticleest.test.mixed_nlg.trans.particle_param_trans  as PartModel

e0 = numpy.array([0.0, ])
z0 = numpy.array([1.0, ])
P0 = numpy.eye(1)

if __name__ == '__main__':
    
    num = 50
    nums = 5
    theta_true = 0.3
    R = 0.1*numpy.eye(1)
    Qxi = 0.1*numpy.eye(1)
    Qz = 0.1*numpy.eye(1)

    # How many steps forward in time should our simulation run
    steps = 200
    sims = 130
    
    max_iter = 100

    # Create arrays for storing some values for later plotting    
    vals = numpy.zeros((2, num+1, steps+1))
    yvec = numpy.zeros((1, steps))

    

    estimate = numpy.zeros((1,sims))
    
    plt.ion()
    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    fig4 = plt.figure()
    
    for k in range(sims):
        print k
        
        (x, y) = PartModel.generate_data(theta_true, Qxi, Qz, R, steps)

        # Create an array for our particles 
        model = PartModel.ParticleParamTrans((theta_true,), 
                                             R=R,
                                             Qxi=Qxi,
                                             Qz=Qz)
        
        print "estimation start"
        
        plt.figure(fig1.number)
        plt.clf()
        t = numpy.asarray(range(steps+1))
        plt.plot(t[1:],numpy.asarray(y),'b.')
        plt.plot(t,numpy.asarray(x)[:,0],'g-')
        plt.plot(t,numpy.asarray(x)[:,1],'r-')
        plt.title("Param = %s" % theta_true)
        fig1.show()
        plt.draw()
        
        theta_guess = numpy.random.uniform()
        # Create an array for our particles 
        model = PartModel.ParticleParamTrans((theta_true,), 
                                             R=R,
                                             Qxi=Qxi,
                                             Qz=Qz)
        ParamEstimator = param_est.ParamEstimation(model, u=None, y=y)
        ParamEstimator.set_params(numpy.array((theta_guess,)).reshape((-1,1)))
        
        params_it = numpy.zeros((max_iter))
        Q_it = numpy.zeros((max_iter))
        it = 0
        def callback(params, Q):
            global it
            params_it[it] = params[0]
            Q_it[it] = Q
            it = it+1
            plt.figure(fig3.number)
            plt.clf()
            plt.plot(range(it), params_it[:it], 'b-')
            plt.plot((0.0, it), (theta_true, theta_true), 'b--')
            plt.figure(fig4.number)
            plt.plot(range(it), Q_it[:it], 'r-')
            plt.show()
            plt.draw()
            return
        
        
        (param, Q) = ParamEstimator.maximize(param0=numpy.array((theta_guess,)),
                                             num_part=num,
                                             num_traj=nums,
                                             max_iter=max_iter,
                                             callback=callback)
        
        # Extract data from trajectories for plotting
#        i=0
#        for step in ParamEstimator.pt:
#            pa = step.pa
#            for j in range(pa.num):
#                vals[0,j,i]=pa.part[j].eta[0,0]
#                vals[1,j,i]=pa.part[j].kf.z.reshape(-1)
#            i += 1
#        
        svals = numpy.zeros((3, nums, steps+1))
        
        for i in range(steps+1):
            for j in range(nums):
                svals[0,j,i]=ParamEstimator.straj.traj[i,j,0]
                svals[1,j,i]=ParamEstimator.straj.traj[i,j,1]
        plt.figure(fig1.number)
        # Does this really look right?        
        for j in range(nums):
            plt.plot(range(steps+1),svals[0,j,:],'g-')
            plt.plot(range(steps+1),svals[1,j,:],'r-')


        

        
        
        print "maximization start"
        
        estimate[0,k] = param
        
        plt.figure(fig2.number)
        plt.clf()
        plt.hist(estimate[0,:(k+1)].T)
        fig2.show()
        plt.show()
        plt.draw()

        

    
    plt.hist(estimate.T)
    plt.ioff()
    plt.show()
    plt.draw()
    print "exit"