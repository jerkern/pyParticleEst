#!/usr/bin/python

import numpy
import matplotlib.pyplot as plt
import pyparticleest.models.mlnlg as mlnlg
import pyparticleest.param_est as param_est

gradient_test = False

x0 = numpy.array([[1.0,], [1.0,]])
P0 = numpy.eye(2)
Q = numpy.diag([ 0.01, 0.01])
B = numpy.array([[0.0,], 
                 [1.0,]])
R = numpy.array([[1.0]])
A = numpy.asarray(((1.0, 1.0),
                   (0.0, 1.0)))

def generate_reference(uvec, steps, c_true):
    
    C = numpy.array([[c_true, 0.0]])
        
    x = numpy.zeros((steps+1,2,1))
    u = numpy.zeros((steps,1,1))
    y = numpy.zeros((steps,1,1))
    xo = numpy.random.multivariate_normal(x0.ravel(), P0).reshape((-1,1))
    x[0,:] = numpy.copy(xo)
    for i in range(steps):
        # Calc linear states
        utmp = uvec[i].reshape(-1,1)
        xn = A.dot(xo) + B.dot(utmp) + numpy.random.multivariate_normal(numpy.zeros((2,)), Q).reshape((-1,1))
        x[i+1,:]=xn
        u[i] = utmp
        y[i] = C.dot(xn).ravel() + numpy.random.multivariate_normal((0.0,),R).reshape((-1,1))
        xo=numpy.copy(xn)
    return (u, y, x)


class ParticleParamOutput(mlnlg.MixedNLGaussianInitialGaussian):
    """ Implement a simple system by extending the MixedNLGaussian class """
    def __init__(self, params):
        """ Define all model variables """
        C = numpy.array([[0.0,],])
        
        self.params = numpy.copy(params)
        
        xi0 = x0[0,0].reshape((1,1))
        z0 = x0[1,0].reshape((1,1))
        Pxi0 = P0[0,0].reshape((1,1))
        Pz0 = P0[1,1].reshape((1,1))
        Qxi = Q[0,0].reshape((1,1))
        Qz = Q[1,1].reshape((1,1))
        
        Axi = numpy.copy(A[0,0]).reshape(1,1)
        Az = numpy.copy(A[1,1]).reshape(1,1)
                   
        # Linear states handled by base-class
        super(ParticleParamOutput,self).__init__(z0=z0, xi0=xi0, Pxi0=Pxi0,
                                                 Pz0=Pz0, Axi=Axi, Az=Az, C=C,
                                                 Qxi=Qxi, Qz=Qz, R=R)
        
    def get_nonlin_pred_dynamics(self, particles, u, t):
        xil = numpy.vstack(particles)[:,0]
        fxil = xil[:,numpy.newaxis,numpy.newaxis]
        fu = B[0,0]*u
        f = fxil+fu
        return (None, f, None)
    
    def get_lin_pred_dynamics(self, particles, u, t):
        N = len(particles)
        fu = (B[1,0]*u).reshape(1,1)
        fz = numpy.repeat(fu[numpy.newaxis], N, 0)
        return (None, fz, None)
    
    def get_meas_dynamics(self, y, particles, t):
        xil = particles[:,0]
        h = self.params[0]*xil[:,numpy.newaxis,numpy.newaxis]
        return (numpy.asarray(y).reshape((-1,1)), None, h, None)  
    
    def set_params(self, params):
        """ New set of parameters """
        self.params = numpy.copy(params)
        
    def get_meas_dynamics_grad(self, particles, y, t):
        xil = particles[:,0]
        h_grad = particles[:,numpy.newaxis,0:1]
        return (None, h_grad, None)
    

if __name__ == '__main__':
    theta_true = 2.5
    # How many steps forward in time should our simulation run
    steps = 200
    
    uvec = -10.0*numpy.hstack((-1.0*numpy.ones(steps/4), 1.0*numpy.ones(steps/2),-1.0*numpy.ones(steps/4) ))

    if (gradient_test):
        num = 50
        nums = 5
        #numpy.random.seed(1)
        model = ParticleParamOutput((theta_true,))
        (u, y, x) = generate_reference(uvec, steps, theta_true)
        gt = param_est.GradientTest(model, u=u, y=y)
        gt.set_params(numpy.array((theta_true,)))
        gt.simulate(num, nums)
        param_steps = 101
        param_vals = numpy.linspace(-10.0, 10.0, param_steps)
        gt.test(0, param_vals, num=num, nums=nums)
        plt.ion()
        plt.figure(1)
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
        gt.plot_y.plot(2)
        gt.plot_xn.plot(3)
        gt.plot_x0.plot(4)
        plt.show()
        
    else:
        sims=100
        max_iter = 10
#        num = 50
#        nums = 5
        iterations = numpy.asarray(range(max_iter))
        num = numpy.ceil(10.0 + 20.0/(iterations[-1]**3)*iterations**3).astype(int)
        nums = numpy.ceil(2.0 + 10.0/(iterations[-1]**3)*iterations**3).astype(int)
        
        # Create arrays for storing states for later plotting    
   
        estimate = numpy.zeros((1,sims))
        plt.ion()
        fig1 = plt.figure()
        fig2 = plt.figure()
        
      
        
        for k in range(sims):
            # Create reference
            (u, y, x) = generate_reference(uvec, steps, theta_true)
        
            print "estimation start"
            # Initial guess (-5, 5)
            theta_guess = 10.0*numpy.random.uniform()-5.0
            print theta_guess
            model = ParticleParamOutput((theta_guess,))
            pe = param_est.ParamEstimation(model, u=u, y=y)
            pe.set_params(numpy.array((theta_guess,)).reshape((-1,1)))
            
            params_it = numpy.zeros((max_iter))
            Q_it = numpy.zeros((max_iter))
            it = 0
            def callback(params, Q):
                global it
                params_it[it] = params[0]
                Q_it[it] = Q
                it = it+1
                plt.figure(3)
                plt.clf()
                plt.plot(range(it), params_it[:it], 'b-')
                plt.plot((0.0, it), (theta_true, theta_true), 'b--')
                if (Q != None):
                    plt.figure(4)
                    plt.plot(range(it), Q_it[:it], 'r-')
                plt.show()
                plt.draw()
                return
            
            callback((theta_guess,), None)
            
            (param, Qval) = pe.maximize(param0=numpy.array((theta_guess,)),
                                        num_part=num,
                                        num_traj=nums,
                                        callback=callback,
                                        analytic_gradient=True,
                                        max_iter=max_iter,
                                        )
            estimate[0,k] = param
            
            plt.figure(1)
            plt.clf()
            plt.plot(range(steps+1), x[:, 0], 'r-')
            plt.plot(range(steps+1), x[:, 1], 'b-')
            
            
            for j in xrange(nums[-1]):
                plt.plot(range(steps+1), pe.straj.traj[:,j,0], 'g--')
                plt.plot(range(steps+1), pe.straj.traj[:,j,1], 'k--')
                plt.plot(range(steps+1), pe.straj.traj[:,j,1]-numpy.sqrt(pe.straj.traj[:,j,2]), 'k-.')
                plt.plot(range(steps+1), pe.straj.traj[:,j,1]+numpy.sqrt(pe.straj.traj[:,j,2]), 'k-.')
            
            plt.show()
                    
            plt.figure(fig2.number)
            plt.clf()
            plt.hist(estimate[0,:(k+1)].T, 
                     bins=numpy.linspace(-5.0, 5.0, 100),
                     normed=True)
            fig2.show()
            plt.show()
            plt.draw()
    
        plt.ioff()

        print "mean: %f" % numpy.mean(estimate)
        print "stdd: %f" % numpy.std(estimate)
        
        plt.clf()
        plt.hist(estimate[0,:(k+1)].T, 
                 bins=numpy.linspace(-5.0, 5.0, 100),
                 normed=True)
        plt.ioff()
        plt.show()
        plt.draw()
    print "exit"