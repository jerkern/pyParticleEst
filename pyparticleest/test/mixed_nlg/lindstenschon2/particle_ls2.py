'''
Created on Nov 11, 2013

@author: ajn
'''

import numpy
import math
from pyparticleest.models.mixed_nl_gaussian import MixedNLGaussian
import pyparticleest.kalman as kalman

def sign(x):
    if (x < 0.0):
        return -1.0
    else:
        return 1.0

def calc_h(eta):
    return numpy.asarray(((0.1*(eta[0,0]**2)*sign(eta[0,0])),
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
        e = params[0]*numpy.arctan(e) + Ae.dot(z) + numpy.random.normal(0.0,0.01)
        
        wz = numpy.random.multivariate_normal(numpy.zeros((3,)), 0.01*numpy.eye(3, 3)).ravel().reshape((-1,1))
        
        z = Az.dot(z) + wz
        h = calc_h(e)
        y[:,i] = (h + C.dot(z)).ravel()
        e_vec[:,i] = e.ravel()
        z_vec[:,i] = z.ravel()
    
    return (y.T.tolist(), e_vec, z_vec)    

class ParticleLS2(MixedNLGaussian):
    """ Implement a simple system by extending the MixedNLGaussian class """
    def __init__(self, eta0, params):
        """ Define all model variables """
        Ae = numpy.array([[params[1], 0.0, 0.0]])
        Az = numpy.asarray(((1.0, params[2], 0.0), 
                            (0.0,  
                             params[3]*math.cos(params[4]), 
                             -params[3]*math.sin(params[4])),
                             (0.0,  
                             params[3]*math.sin(params[4]), 
                             params[3]*math.cos(params[4]))))
        
        C = numpy.array([[0.0, 0.0, 0.0], [1.0, -1.0, 1.0]])
        z0 = numpy.zeros((3,1))
        P0 = 0.0*numpy.eye(3,3)
        Qe= numpy.diag([ 0.01,])
        Qz = numpy.diag([ 0.01, 0.01, 0.01])
        R = numpy.diag([0.1, 0.1])
        z0 =  numpy.copy(z0).reshape((-1,1))
        fe = params[0]*numpy.arctan(eta0[0,0])
        h = calc_h(eta0)
        # Linear states handled by base-class
        super(ParticleLS2,self).__init__(z0=numpy.reshape(z0,(-1,1)), P0=P0,
                                                 e0 = eta0,
                                                 Az=Az, C=C, Ae=Ae,
                                                 R=R, Qe=Qe, Qz=Qz,
                                                 fe=fe, h=h, params=params)
        
    def prep_update(self, u):
        """ Perform a time update of all states """
        fe = self.params[0]*numpy.arctan(self.eta)
        self.set_dynamics(fe=fe)
        
    def prep_measure(self, y):
        """ Perform a measurement update """
        h = calc_h(self.eta)
        self.set_dynamics(h=h)
        return y
    
    def next_pdf(self, next_cpart, u):
        return super(ParticleLS2,self).next_pdf(next_cpart, None)
    
    # Override this method since there is no uncertainty in z0    
    def eval_logp_x0(self, z0, P0, diff_z0, diff_P0):
        """ Calculate gradient of a term of the I1 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""    
            
        e0 = numpy.asarray((0.0,)).reshape((-1,1))
        P0 = numpy.asarray((1.0,)).reshape((-1,1))
        return (kalman.lognormpdf(self.eta, e0, P0),
                numpy.zeros(self.params.shape))

    def set_params(self, params):
        """ New set of parameters """
        # Update all needed matrices and derivates with respect
        # to the new parameter set
        
        Ae = numpy.array([[params[1], 0.0, 0.0]])
        Ae_grad = [numpy.array([[0.0, 0.0, 0.0]]), #theta_1
                   numpy.array([[1.0, 0.0, 0.0]]), #theta_2
                   numpy.array([[0.0, 0.0, 0.0]]), #theta_3
                   numpy.array([[0.0, 0.0, 0.0]]), #theta_4
                   numpy.array([[0.0, 0.0, 0.0]]), #theta_5
                   ]
        Az = numpy.asarray(((1.0, params[2], 0.0), 
                            (0.0,  
                             params[3]*math.cos(params[4]), 
                             -params[3]*math.sin(params[4])),
                             (0.0,  
                             params[3]*math.sin(params[4]), 
                             params[3]*math.cos(params[4]))))
        Az_grad = [numpy.asarray(((0.0, 0.0, 0.0), 
                                  (0.0, 0.0, 0.0),
                                  (0.0, 0.0, 0.0))), #theta_1
                   numpy.asarray(((0.0, 0.0, 0.0), 
                                  (0.0, 0.0, 0.0),
                                  (0.0, 0.0, 0.0))), #theta_2
                   numpy.asarray(((0.0, 1.0, 0.0), 
                                  (0.0, 0.0, 0.0),
                                  (0.0, 0.0, 0.0))), #theta_3
                   numpy.asarray(((0.0, 0.0, 0.0), 
                                  (0.0,  
                                   math.cos(params[4]), 
                                   -math.sin(params[4])),
                                  (0.0,  
                                   math.sin(params[4]), 
                                   math.cos(params[4])))), #theta_4
                   numpy.asarray(((0.0, 0.0, 0.0), 
                                  (0.0,  
                                   -params[3]*math.sin(params[4]), 
                                   -params[3]*math.cos(params[4])),
                                  (0.0,  
                                   params[3]*math.cos(params[4]), 
                                   -params[3]*math.sin(params[4])))), #theta_5
                   ]
        
        fe = params[0]*numpy.arctan(self.eta)
        fe_grad = [ numpy.array([[numpy.arctan(self.eta)]]),
                   numpy.array([[0.0]]),
                   numpy.array([[0.0]]),
                   numpy.array([[0.0]]),
                   numpy.array([[0.0]])] 
        self.set_dynamics(Ae=Ae, Az=Az, fe=fe)
        self.set_dynamics_gradient(grad_Ae=Ae_grad, grad_Az=Az_grad, grad_fe=fe_grad)
        return super(ParticleLS2, self).set_params(params)
