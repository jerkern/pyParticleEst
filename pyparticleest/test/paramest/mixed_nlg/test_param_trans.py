#!/usr/bin/python

import numpy
import pyparticleest.param_est as param_est
from pyparticleest.models.mlnlg import MixedNLGaussianInitialGaussian
import matplotlib.pyplot as plt

R = numpy.array([[1.0]])
Q = numpy.diag([ 1.0, 1.0])
xi0_true = numpy.array([0.0, ])
z0_true = numpy.array([1.0, ])
P0 = numpy.eye(1)
gradient_test = False

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


class ParticleParamTrans(MixedNLGaussianInitialGaussian):
    """ Implement a simple system by extending the MixedNLGaussian class """
    def __init__(self, params, R, Qxi, Qz):
        """ Define all model variables """
        C = numpy.array([[0.0,]])
        self.params=numpy.copy(params)
        Axi = params[0]*numpy.eye(1.0)
        Az = numpy.eye(1.0)
        self.A_grad = numpy.array([[[1.0,],[0.0,]]])
        
        z0 = numpy.copy(z0_true)
        xi0 = numpy.copy(xi0_true)
        
        Pz0 = numpy.eye(1)    
        Pxi0 = numpy.eye(1)
        # Linear states handled by base-class
        super(ParticleParamTrans,self).__init__(Az=Az, C=C, Axi=Axi,
                                                R=R, Qxi=Qxi, Qz=Qz,
                                                z0=z0, xi0=xi0,
                                                Pz0=Pz0, Pxi0=Pxi0)

    def get_nonlin_pred_dynamics(self, particles, u, t):
        xil = numpy.vstack(particles)[:,0]
        fxil = xil[:,numpy.newaxis,numpy.newaxis]
        return (None, fxil, None)
        
    def get_meas_dynamics(self, y, particles, t):
        xil = numpy.vstack(particles)[:,0]
        h = xil[:,numpy.newaxis,numpy.newaxis]
        return (numpy.asarray(y).reshape((-1,1)), None, h, None)
    
    def set_params(self, params):
        """ New set of parameters """
        # Update all needed matrices and derivates with respect
        # to the new parameter set
        #Axi = numpy.array([[params[0],]])
        #self.set_dynamics(Axi=Axi)
        self.params=numpy.copy(params)
        Axi = params[0]*numpy.eye(1.0)
        Az = numpy.eye(1.0)
        self.set_dynamics(Az=Az, Axi=Axi)
        
    def get_pred_dynamics_grad(self, particles, u, t):
        N = len(particles)
        return (numpy.repeat(self.A_grad[numpy.newaxis], N, 0), None, None)
    

if __name__ == '__main__':

    theta_true = 0.1
    Qxi = Q[0,0].reshape((1,1))
    Qz = Q[1,1].reshape((1,1))
    P0 = numpy.eye(2)
    # How many steps forward in time should our simulation run  
    steps = 200
    #model = ParticleParamTrans((theta_true,), R=R, Qxi=Qxi, Qz=Qz)
    model = ParticleParamTrans((theta_true,), R=R, Qxi=Qxi, Qz=Qz)
    if (gradient_test):
        
        num = 50
        nums = 5
        #numpy.random.seed(1)
        x0 = numpy.vstack((xi0_true, z0_true))
        (y, x) = generate_reference(x0, P0, theta_true, steps)
        gt = param_est.GradientTest(model, u=None, y=y)
        gt.set_params(numpy.array((theta_true,)))
        gt.simulate(num, nums)
        param_steps = 51
        param_vals = numpy.linspace(-0.3, 0.7, param_steps)
        gt.test(0, param_vals, num=num, nums=nums)
        plt.ion()
        plt.clf()
        plt.plot(range(steps+1), x[:, 0], 'r-')
        plt.plot(range(steps+1), x[:, 1], 'b-')
        
        
        for j in xrange(nums):
            plt.plot(range(steps+1), gt.straj.traj[:,j,0], 'g--')
            plt.plot(range(steps+1), gt.straj.traj[:,j,1], 'k--')
            plt.plot(range(steps+1), gt.straj.traj[:,j,1]-numpy.sqrt(gt.straj.traj[:,j,2]), 'k-.')
            plt.plot(range(steps+1), gt.straj.traj[:,j,1]+numpy.sqrt(gt.straj.traj[:,j,2]), 'k-.')
        
        plt.show()
    
        plt.draw()
        plt.ioff()
        #gt.plot_y.plot(1)
        gt.plot_xn.plot(1)
        #gt.plot_x0.plot(3)
        plt.show()
    else:
        max_iter = 100
#        num = 50
#        nums = 5
        iterations = numpy.asarray(range(max_iter))
        num = numpy.ceil(5.0 + 45.0/(iterations[-1]**3)*iterations**3).astype(int)
        nums = numpy.ceil(1.0 + 4.0/(iterations[-1]**3)*iterations**3).astype(int)
        R = 0.1*numpy.eye(1)
        Qxi = 0.1*numpy.eye(1)
        Qz = 0.1*numpy.eye(1)
    
        # How many steps forward in time should our simulation run
        steps = 200
        sims = 130
        
        estimate = numpy.zeros((1,sims))
        
        plt.ion()
        fig1 = plt.figure()
        fig2 = plt.figure()
#        fig3 = plt.figure()
#        fig4 = plt.figure()
        
        for k in range(sims):
            print k
            
            x0 = numpy.vstack((xi0_true, z0_true))
            (y, x) = generate_reference(x0, P0, theta_true, steps)
    
            # Create an array for our particles 
            
            
            print "estimation start"
            
            plt.figure(fig1.number)
            plt.clf()
            t = numpy.asarray(range(steps+1))
            plt.plot(t[1:],numpy.asarray(y),'b.')
            plt.plot(t,x[:,0],'g-')
            plt.plot(t,x[:,1],'r-')
            plt.title("Param = %s" % theta_true)
            fig1.show()
            plt.draw()
            
            theta_guess = numpy.random.uniform()
 
            ParamEstimator = param_est.ParamEstimation(model, u=None, y=y)
            ParamEstimator.set_params(numpy.array((theta_guess,)).reshape((-1,1)))
            
#            params_it = numpy.zeros((max_iter))
#            Q_it = numpy.zeros((max_iter))
#            it = 0
#            def callback(params, Q):
#                global it
#                params_it[it] = params[0]
#                Q_it[it] = Q
#                it = it+1
#                plt.figure(fig3.number)
#                plt.clf()
#                plt.plot(range(it), params_it[:it], 'b-')
#                plt.plot((0.0, it), (theta_true, theta_true), 'b--')
#                plt.figure(fig4.number)
#                plt.plot(range(it), Q_it[:it], 'r-')
#                plt.show()
#                plt.draw()
#                return
            
            
            (param, Qval) = ParamEstimator.maximize(param0=numpy.array((theta_guess,)),
                                                    num_part=num,
                                                    num_traj=nums,
                                                    #callback=callback,
                                                    analytic_gradient=True,
                                                    max_iter=max_iter,
                                                    )
            
            plt.figure(fig1.number)
            
            print "maximization start"
            
            estimate[0,k] = param
            
            plt.figure(fig2.number)
            plt.clf()
            bins=numpy.linspace(-0.5, 1.0, 30)
            
            plt.hist(estimate[0,:(k+1)].T, bins=bins, normed=True)
            fig2.show()
            plt.show()
            plt.draw()
            
        print "mean: %f" % numpy.mean(estimate)
        print "stdd: %f" % numpy.std(estimate)

        plt.ioff()
        plt.clf()
        plt.hist(estimate.T, normed=True)
        plt.show()
        plt.draw()
    print "exit"