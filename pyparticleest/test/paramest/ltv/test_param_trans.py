#!/usr/bin/python

import numpy
import pyparticleest.param_est as param_est

import matplotlib.pyplot as plt
from pyparticleest.models.ltv import LTV

R = numpy.array([[0.01]])
Q = numpy.diag([ 0.01, 0.1])
gradient_test = True

def generate_reference(z0, P0, theta_true, steps):
    A = numpy.asarray(((1.0, theta_true), (0.0, 1.0)))
    C = numpy.array([[1.0, 0.0]])
    
    states = numpy.zeros((steps+1,2))
    y = numpy.zeros((steps,1))
    x0 = numpy.random.multivariate_normal(z0.ravel(), P0)
    states[0] = numpy.copy(x0)
    for i in range(steps):
            
        # Calc linear states
        x = states[i].reshape((-1,1))
        xn = A.dot(x) + numpy.random.multivariate_normal(numpy.zeros(x0.shape), Q).reshape((-1,1))
        states[i+1]=xn.ravel()

        # use the correct particle to generate the true measurement
        y[i] = C.dot(xn).ravel() + numpy.random.multivariate_normal((0.0,),R).reshape((-1,1))
        
    return (y, states)

class ParticleParamTrans(LTV):
    """ Implement a simple system by extending the MixedNLGaussian class """
    def __init__(self, z0, P0, params):
        """ Define all model variables """
        self.params = numpy.copy(params)
        A= numpy.array([[1.0, params[0]], [0.0, 1.0]])
        C = numpy.array([[1.0, 0.0]])
        z0 =  numpy.copy(z0).reshape((-1,1))
        # Linear states handled by base-class
        super(ParticleParamTrans,self).__init__(z0=z0, P0=P0, A=A,
                                                C=C, R=R, Q=Q,)
        
    def set_params(self, params):
        """ New set of parameters """
        # Update all needed matrices and derivates with respect
        # to the new parameter set
        self.params = numpy.copy(params)
        A= numpy.array([[1.0, params[0]], [0.0, 1.0]])
        A_grad= numpy.array([[0.0, 1.0], [0.0, 0.0]])
        self.kf.set_dynamics(A=A)
        
z0 = numpy.array([0.0, 1.0, ])
P0 = numpy.eye(2)

def callback(params, Q):
    #print "Q=%f, params=%s" % (Q, params)
    return

if __name__ == '__main__':
    
    num = 1
    nums = 1
    
    theta_true = 0.1
    theta_guess = 0.3
    #theta_guess = theta_true   
   

    # How many steps forward in time should our simulation run
    steps = 200
    sims = 10
    max_iter = 200
    tol = 0.0001
    model = ParticleParamTrans(z0=z0,P0=P0, params=(theta_true,))
    
    if (gradient_test):
        # Create a reference which we will try to estimate using a RBPS
        (y, x) = generate_reference(z0, P0, theta_true, steps)
        plt.plot(range(steps+1), x[:, 0], 'r-')
        plt.plot(range(steps+1), x[:, 1], 'b-')
        
        # Create an array for our particles 
        gt = param_est.ParamEstimation(model, u=None, y=y)
        gt.simulate(num, nums)
        
        plt.plot(range(steps+1), gt.straj.traj[:,0,0], 'r--')
        plt.plot(range(steps+1), gt.straj.traj[:,0,1], 'b--')
        plt.plot(range(steps+1), gt.straj.traj[:,0,0]-numpy.sqrt(gt.straj.traj[:,0,2]), 'r--')
        plt.plot(range(steps+1), gt.straj.traj[:,0,0]+numpy.sqrt(gt.straj.traj[:,0,2]), 'r--')
        plt.plot(range(steps+1), gt.straj.traj[:,0,1]-numpy.sqrt(gt.straj.traj[:,0,5]), 'b--')
        plt.plot(range(steps+1), gt.straj.traj[:,0,1]+numpy.sqrt(gt.straj.traj[:,0,5]), 'b--')
        plt.show()
#        gt.set_params(numpy.array((theta_guess,)))
#        
#        param_steps = 101
#        param_vals = numpy.linspace(-0.1, 0.3, param_steps)
#        gt.test(0, param_vals, num=num)
#    
#        gt.plot_y.plot(1)
#        gt.plot_xn.plot(2)
#        gt.plot_x0.plot(3)
    else:
    
        # Create arrays for storing some values for later plotting    
        vals = numpy.zeros((2, num+1, steps+1))
        yvec = numpy.zeros((1, steps))
    
        estimate = numpy.zeros((1,sims))
        
        plt.ion()
        fig1 = plt.figure()
        fig2 = plt.figure()
        
        for k in range(sims):
            print k
            # Create reference
            (ylist, states) = particle_param_trans.generate_reference(z0, P0, theta_true, steps)
        
            plt.figure(fig1.number)
            plt.clf()
            x = numpy.asarray(range(steps+1))
            plt.plot(x[1:],numpy.asarray(ylist)[:,0],'b.')
            plt.plot(range(steps+1),states[0,:],'go')
            plt.plot(range(steps+1),states[1,:],'ro')
            plt.title("Param = %s" % theta_true)
            plt.draw()
            fig1.show()
                
            # Create an array for our particles 
            model = PartModel(z0=z0,P0=P0, params=(theta_guess,))
            ParamEstimator = param_est.ParamEstimation(model=model, u=None, y=ylist)
            ParamEstimator.set_params(numpy.array((theta_guess,)).reshape((-1,1)))
            #ParamEstimator.simulate(num_part=num, num_traj=nums)
            print "maximization start"
            (param, Q) = ParamEstimator.maximize(param0=numpy.array((theta_guess,)), num_part=num, num_traj=nums,
                                                 max_iter=max_iter, tol=tol)
            
            svals = numpy.zeros((2, nums, steps+1))
            
            for i in range(steps+1):
                for j in range(nums):
                    svals[:,j,i]=ParamEstimator.straj.traj[i,j,:2]
                    
            for j in range(nums):
                plt.plot(range(steps+1),svals[0,j,:],'g-')
                plt.plot(range(steps+1),svals[1,j,:],'r-')
    
           
            
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