#!/usr/bin/python

import numpy
import pyparticleest.paramest.paramest as param_est

import matplotlib.pyplot as plt
from pyparticleest.models.ltv import LTV
from pyparticleest.models.mlnlg import MixedNLGaussianInitialGaussian

R = numpy.array([[0.1]])
Q = numpy.diag([ 0.1, 0.1])
gradient_test = False

def generate_reference(z0, P0, theta_true, steps):
    A = numpy.asarray(((1.0, theta_true), (0.0, 1.0)))
    C = numpy.array([[1.0, 0.0]])
    
    states = numpy.zeros((steps+1,2))
    y = numpy.zeros((steps+1,1))
    x0 = numpy.random.multivariate_normal(z0.ravel(), P0)
    states[0] = numpy.copy(x0)
    y[0] = C.dot(x0.reshape((-1,1))).ravel() + numpy.random.multivariate_normal((0.0,),R).ravel()
    for i in range(steps):
            
        # Calc linear states
        x = states[i].reshape((-1,1))
        xn = A.dot(x) + numpy.random.multivariate_normal(numpy.zeros(x0.shape), Q).reshape((-1,1))
        states[i+1]=xn.ravel()

        # use the correct particle to generate the true measurement
        y[i+1] = C.dot(xn).ravel() + numpy.random.multivariate_normal((0.0,),R).ravel()
        
    return (y, states)

class ParticleLTV(LTV):
    """ Implement a simple system by extending the MixedNLGaussian class """
    def __init__(self, z0, P0, params):
        """ Define all model variables """
        self.params = numpy.copy(params)
        A= numpy.array([[1.0, params[0]], [0.0, 1.0]])
        self.A_grad= numpy.array([[0.0, 1.0], [0.0, 0.0]])[numpy.newaxis]
        C = numpy.array([[1.0, 0.0]])
        z0 =  numpy.copy(z0).reshape((-1,1))
        # Linear states handled by base-class
        super(ParticleLTV,self).__init__(z0=z0, P0=P0, A=A,
                                                C=C, R=R, Q=Q,)
        
    def set_params(self, params):
        """ New set of parameters """
        # Update all needed matrices and derivates with respect
        # to the new parameter set
        self.params = numpy.copy(params)
        A= numpy.array([[1.0, params[0]], [0.0, 1.0]])
        self.kf.set_dynamics(A=A)
        
    def get_pred_dynamics_grad(self, u, t):
        return (self.A_grad, None, None)
    
class ParticleMLNLG(MixedNLGaussianInitialGaussian):
    """ Implement a simple system by extending the MixedNLGaussian class """
    def __init__(self, z0, P0, params):
        """ Define all model variables """
        C = numpy.array([[0.0,]])
        self.params=numpy.copy(params)
        Axi = params[0]*numpy.eye(1.0)
        Az = numpy.eye(1.0)
        self.A_grad = numpy.array([[[1.0,],[0.0,]]])
        
        xi0 = numpy.copy(z0[1]).reshape((1,1))
        z0 = numpy.copy(z0[0]).reshape((1,1))
        
        Qxi = Q[0,0].reshape((1,1))
        Qz = Q[1,1].reshape((1,1))
        Pz0 = numpy.copy(P0[0,0]).reshape((1,1))   
        Pxi0 = numpy.copy(P0[1,1]).reshape((1,1))  
        # Linear states handled by base-class
        super(ParticleMLNLG,self).__init__(Az=Az, C=C, Axi=Axi,
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
        
z0 = numpy.array([0.0, 1.0, ])
P0 = numpy.eye(2)

def callback(params, Q):
    print "Q=%f, params=%s" % (Q, params)
    return

if __name__ == '__main__':
    
    num = 50
    nums = 5
    
    theta_true = 0.1

    # How many steps forward in time should our simulation run
    steps = 200
    model_ltv = ParticleLTV(z0=z0,P0=P0, params=(theta_true,))
    model_mlnlg = ParticleMLNLG(z0=z0,P0=P0, params=(theta_true,))

    def callback_sim(pe):
        plt.clf()
        plt.plot(range(steps+1), x[:, 0], 'r-')
        plt.plot(range(steps+1), x[:, 1], 'b-')
        plt.plot(range(steps+1), pe.straj.traj[:,0,0], 'r--')
        plt.plot(range(steps+1), pe.straj.traj[:,0,1], 'b--')
        plt.plot(range(steps+1), pe.straj.traj[:,0,0]-numpy.sqrt(pe.straj.traj[:,0,2]), 'r--')
        plt.plot(range(steps+1), pe.straj.traj[:,0,0]+numpy.sqrt(pe.straj.traj[:,0,2]), 'r--')
        plt.plot(range(steps+1), pe.straj.traj[:,0,1]-numpy.sqrt(pe.straj.traj[:,0,5]), 'b--')
        plt.plot(range(steps+1), pe.straj.traj[:,0,1]+numpy.sqrt(pe.straj.traj[:,0,5]), 'b--')
        plt.draw()

    max_iter = 100
    sims = 10 #130
    tol = 0.0
    plt.ion()
    theta_guess = 0.3

    estimate_ltv = numpy.zeros((1,sims))
    estimate_mlnlg = numpy.zeros((1,sims))
    estimate_mlnlg_analytic = numpy.zeros((1,sims))
    
    plt.ion()
    fig1 = plt.figure()
#    fig2 = plt.figure()
    fig3 = plt.figure()
    plt.show()
    
    for k in range(sims):
        # Create reference
        numpy.random.seed(k)
        (y, x) = generate_reference(z0, P0, theta_true, steps)
    
        # Create an array for our particles 
        pe_ltv = param_est.ParamEstimation(model=model_ltv, u=None, y=y)
        pe_mlnlg = param_est.ParamEstimation(model=model_mlnlg, u=None, y=y)
        pe_ltv.set_params(numpy.array((theta_guess,)).reshape((-1,1)))
        pe_mlnlg.set_params(numpy.array((theta_guess,)).reshape((-1,1)))
        
        #ParamEstimator.simulate(num_part=num, num_traj=nums)
#            (param, Qval) = pe.maximize(param0=numpy.array((theta_guess,)), num_part=num, num_traj=nums,
#                                        max_iter=max_iter, callback_sim=callback_sim, tol=tol, callback=callback,
#                                        analytic_gradient=False)
        (param, Qval) = pe_ltv.maximize(param0=numpy.array((theta_guess,)),
                                        num_part=1, num_traj=1,
                                        max_iter=max_iter,
                                        tol=tol,
                                        analytic_gradient=True,
                                        meas_first=True)
        estimate_ltv[0,k] = param
        
#        (param, Qval) = pe_mlnlg.maximize(param0=numpy.array((theta_guess,)),
#                                          num_part=num, num_traj=nums,
#                                          max_iter=max_iter,
#                                          tol=tol,
#                                          analytic_gradient=False,
#                                          smoother='rsas')
#        estimate_mlnlg[0,k] = param
        
        (param, Qval) = pe_mlnlg.maximize(param0=numpy.array((theta_guess,)),
                                          num_part=num, num_traj=nums,
                                          max_iter=max_iter,
                                          tol=tol,
                                          analytic_gradient=True,
                                          smoother='rsas',
                                          meas_first=True)
        estimate_mlnlg_analytic[0,k] = param
        
#        print "%d: LTV: %f MLNG: %f MLNLG Analytic: %f" % (k,
#                                                           estimate_ltv[0,k], 
#                                                           estimate_mlnlg[0,k],
#                                                           estimate_mlnlg_analytic[0,k])
        print "%d: LTV: %f MLNLG Analytic: %f" % (k,
                                                  estimate_ltv[0,k], 
                                                  estimate_mlnlg_analytic[0,k])
        
        plt.figure(fig1.number)
        plt.clf()
        plt.hist(estimate_ltv[0,:(k+1)].T)
        plt.title('LTV')
        
#        plt.figure(fig2.number)
#        plt.clf()
#        plt.hist(estimate_mlnlg[0,:(k+1)].T)
#        plt.title('MLNLG')
               
        plt.figure(fig3.number)
        plt.clf()
        plt.hist(estimate_mlnlg_analytic[0,:(k+1)].T)
        plt.title('MLNLG Analytic')
        
        plt.show()
        plt.draw()

    print "mean LTV: %f" % numpy.mean(estimate_ltv)
    print "stdd LTV: %f" % numpy.std(estimate_ltv)
#    print "mean MLNLG: %f" % numpy.mean(estimate_mlnlg)
#    print "stdd MLNLG: %f" % numpy.std(estimate_mlnlg)
    print "mean MLNLG Analytic: %f" % numpy.mean(estimate_mlnlg_analytic)
    print "stdd MLNLG Analytic: %f" % numpy.std(estimate_mlnlg_analytic)
    
    
    
    plt.figure(fig1.number)
    plt.clf()
    plt.title('LTV')
    plt.hist(estimate_ltv[0,:].T)
    
    plt.ioff()
    
#    plt.figure(fig2.number)
#    plt.clf()
#    plt.title('MLNLG')
#    plt.hist(estimate_mlnlg[0,].T)
#    
    plt.figure(fig3.number)
    plt.clf()
    plt.hist(estimate_mlnlg_analytic[0,:].T)
    plt.title('MLNLG Analytic')
    
    plt.show()
    plt.draw()
    print "exit"