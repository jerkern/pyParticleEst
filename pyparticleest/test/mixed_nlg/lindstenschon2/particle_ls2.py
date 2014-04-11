'''
Created on Nov 11, 2013

@author: ajn
'''

import numpy
import math
import pyparticleest.models.mixed_nl_gaussian
import pyparticleest.kalman as kalman

def sign(x):
    if (x < 0.0):
        return -1.0
    else:
        return 1.0

def calc_h(eta):
    return numpy.asarray(((0.1*eta[0,0]*math.fabs(eta[0,0])),
                          0.0)).reshape((-1,1))
 
def generate_dataset(params, length):
    Ae = numpy.array([[params[1], 0.0, 0.0]])
    Az = numpy.asarray(((1.0, params[2], 0.0), 
                        (0.0,  
                         params[3]*math.cos(params[4]), 
                         -params[3]*math.sin(params[4])),
                         (0.0,  
                         params[3]*math.sin(params[4]), 
                         params[3]*math.cos(params[4]))))
    
    C = numpy.array([[0.0, 0.0, 0.0], [1.0, -1.0, 1.0]])
    
    e_vec = numpy.zeros((1, length))
    z_vec = numpy.zeros((3, length))
    
    e = numpy.asarray(((numpy.random.normal(0.0,1.0),),))
    z = numpy.zeros((3,1))
    
    e_vec[:,0] = e.ravel()
    z_vec[:,0] = z.ravel()
    
    y = numpy.zeros((2, length))
    h = calc_h(e)
    y[:,0] = (h + C.dot(z)).ravel()
    
    for i in range(1,length):
        e = params[0]*numpy.arctan(e) + Ae.dot(z) + numpy.random.normal(0.0,math.sqrt(0.01))
        
        wz = numpy.random.multivariate_normal(numpy.zeros((3,)), 0.01*numpy.eye(3, 3)).ravel().reshape((-1,1))
        
        z = Az.dot(z) + wz
        h = calc_h(e)
        y[:,i] = (h + C.dot(z)).ravel()
        e_vec[:,i] = e.ravel()
        z_vec[:,i] = z.ravel()
    
    return (y.T.tolist(), e_vec, z_vec)    

class ParticleLS2(pyparticleest.models.mixed_nl_gaussian.MixedNLGaussianInitialGaussian):
    """ Implement a simple system by extending the MixedNLGaussian class """
    def __init__(self, params):
        """ Define all model variables """
        Axi = numpy.array([[params[1], 0.0, 0.0]])
        Az = numpy.asarray(((1.0, params[2], 0.0), 
                            (0.0,  
                             params[3]*math.cos(params[4]), 
                             -params[3]*math.sin(params[4])),
                             (0.0,  
                             params[3]*math.sin(params[4]), 
                             params[3]*math.cos(params[4]))))
        
        C = numpy.array([[0.0, 0.0, 0.0], [1.0, -1.0, 1.0]])
        Qxi= numpy.diag([ 0.01,])
        Qz = numpy.diag([ 0.01, 0.01, 0.01])
        R = numpy.diag([0.1, 0.1])
        xi0 = numpy.asarray((0.0,)).reshape((-1,1))
        Pxi0 = numpy.eye(1)
        z0 = numpy.zeros((3,))
        Pz0 = 0.0*numpy.eye(3)

        # Linear states handled by base-class
        super(ParticleLS2,self).__init__(xi0=xi0, z0=z0, Pz0=Pz0, Pxi0=Pxi0, 
                                         Az=Az, C=C, Axi=Axi,
                                         R=R, Qxi=Qxi, Qz=Qz,
                                         params=params)


    def get_nonlin_pred_dynamics(self, particles, u, t):
        xil = particles[:,0]
        fxil = self.params[0]*numpy.arctan(xil)
        return (None, fxil[:,numpy.newaxis,numpy.newaxis], None)
        
    def get_meas_dynamics(self, particles, y, t):
        N = len(particles)
        xil = numpy.vstack(particles)[:,0]
        h = numpy.zeros((N,2,1))
        h[:,0,0] = 0.1*numpy.fabs(xil)*xil
        return (numpy.asarray(y).reshape((-1,1)), None, h, None)

    # Override this method since there is no uncertainty in z0 
    def eval_logp_x0(self, particles, t):
        return self.eval_logp_xi0(particles[:,:self.lxi])
    
    def eval_logp_x0_val_grad(self, particles, t):
        return (self.eval_logp_xi0(particles[:,:self.lxi]),
                self.eval_logp_xi0_grad(particles[:,:self.lxi]))
    
    def get_pred_dynamics_grad(self, particles, u, t):
        N = len(particles)
        xil = particles[:,0]
        f_grad = numpy.zeros((N, 5, 4,1))
        f_grad[:,0,0,0] = numpy.arctan(xil)
        
        return (N*(self.A_grad,), f_grad, None)
 
    
    def set_params(self, params):
        """ New set of parameters """
        # Update all needed matrices and derivates with respect
        # to the new parameter set
        self.params = numpy.copy(params)
        Axi = numpy.array([[params[1], 0.0, 0.0]])
        
        Az = numpy.asarray(((1.0, params[2], 0.0), 
                            (0.0,  
                             params[3]*math.cos(params[4]), 
                             -params[3]*math.sin(params[4])),
                             (0.0,  
                             params[3]*math.sin(params[4]), 
                             params[3]*math.cos(params[4]))))
        
        self.A_grad = (
            numpy.zeros((4,3)),
            numpy.asarray(((1.0, 0.0, 0.0), 
                           (0.0, 0.0, 0.0),
                           (0.0, 0.0, 0.0),
                           (0.0, 0.0, 0.0))),
            numpy.asarray(((0.0, 0.0, 0.0),
                           (0.0, 1.0, 0.0), 
                           (0.0, 0.0, 0.0),
                           (0.0, 0.0, 0.0))),
            numpy.asarray(((0.0, 0.0, 0.0),
                           (0.0, 0.0, 0.0), 
                           (0.0, math.cos(params[4]), -math.sin(params[4])),
                           (0.0, math.sin(params[4]), math.cos(params[4])))),
            numpy.asarray(((0.0, 0.0, 0.0),
                           (0.0, 0.0, 0.0), 
                           (0.0, -params[3]*math.sin(params[4]), -params[3]*math.cos(params[4])),
                           (0.0, params[3]*math.cos(params[4]), -params[3]*math.sin(params[4]))))
            )
        self.set_dynamics(Axi=Axi, Az=Az)
