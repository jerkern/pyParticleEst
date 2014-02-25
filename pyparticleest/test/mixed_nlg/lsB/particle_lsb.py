'''
Example B from 
Rao-Blackwellized particle smoothers for mixed linear/nonlinear state-space models
Fredrik Lindsten, Thomas B. Schon
'''

import numpy
import math
import pyparticleest.models.mixed_nl_gaussian as mixed_nl_gaussian

def generate_dataset(length):
    Az = numpy.array([[3.0, -1.691, 0.849, -0.3201],
                      [2.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.5, 0.0]])
        
    C = numpy.array([[0.0, 0.0, 0.0, 0.0]])
    
    Qe= numpy.diag([ 0.005])
    Qz = numpy.diag([ 0.01, 0.01, 0.01, 0.01])
    R = numpy.diag([0.1,])
    
    e_vec = numpy.zeros((1, length+1))
    z_vec = numpy.zeros((4, length+1))
    
    e = numpy.array([[0.0,]])
    z = numpy.zeros((4,1))
    
    e_vec[:,0] = e.ravel()
    z_vec[:,0] = z.ravel()
    
    y = numpy.zeros((1, length))
    t = 0
    h = calc_h(e)
    #y[:,0] = (h + C.dot(z)).ravel()
    
    for i in range(1,length+1):
        (Ae, fe) = calc_Ae_fe(e, t)
        
        e = fe + Ae.dot(z) + numpy.random.multivariate_normal(numpy.zeros((1,)),Qe)
        
        wz = numpy.random.multivariate_normal(numpy.zeros((4,)), Qz).ravel().reshape((-1,1))
        
        z = Az.dot(z) + wz
        t = t + 1
        h = calc_h(e)
        y[:,i-1] = (h + C.dot(z) + numpy.random.multivariate_normal(numpy.zeros((1,)), R)).ravel()
        e_vec[:,i] = e.ravel()
        z_vec[:,i] = z.ravel()
    
    return (y.T.tolist(), e_vec, z_vec)    

C_theta = numpy.array([[ 0.0, 0.04, 0.044, 0.08],])
def calc_Ae_fe(eta, t):
    Ae = eta/(1+eta**2)*C_theta
    fe = 0.5*eta+25*eta/(1+eta**2)+8*math.cos(1.2*t)
    return (Ae, fe)

def calc_h(eta):
    return 0.05*eta**2

class ParticleLSB(mixed_nl_gaussian.MixedNLGaussian):
    """ Model 60 & 61 from Lindsten & Schon (2011) """
    def __init__(self):
        """ Define all model variables """
        
        # No uncertainty in initial state
        eta = numpy.array([[0.0],])
        z0 =  numpy.array([[0.0],
                           [0.0],
                           [0.0],
                           [0.0]])
        P0 = 0.0*numpy.eye(4)
        
        Az = numpy.array([[3.0, -1.691, 0.849, -0.3201],
                          [2.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 0.5, 0.0]])
        
        (Ae, fe) = calc_Ae_fe(eta, 0)
        h = calc_h(eta)
        C = numpy.array([[0.0, 0.0, 0.0, 0.0]])
        
        Qe= numpy.diag([ 0.005])
        Qz = numpy.diag([ 0.01, 0.01, 0.01, 0.01])
        R = numpy.diag([0.1,])

        super(ParticleLSB,self).__init__(z0=numpy.reshape(z0,(-1,1)),
                                         P0=P0, e0 = eta,
                                         Az=Az, C=C, Ae=Ae,
                                         R=R, Qe=Qe, Qz=Qz,
                                         fe=fe, h=h)
        
    def prep_update(self, u):
        """ Update system dynamics based on current state, called
            before the predict step """
        (Ae, fe) = calc_Ae_fe(self.eta, self.t)
        self.set_dynamics(fe=fe, Ae=Ae)
        
    def prep_measure(self, y):
        """ ppdate system dynamics based on current state, called
            before the measurement step """
        h = calc_h(self.eta)
        self.set_dynamics(h=h)
        return y
