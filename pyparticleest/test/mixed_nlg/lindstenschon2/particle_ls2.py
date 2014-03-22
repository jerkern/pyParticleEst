'''
Created on Nov 11, 2013

@author: ajn
'''

import numpy
import math
import pyparticleest.part_utils
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

class ParticleLS2(pyparticleest.part_utils.MixedNLGaussian):
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
        fz = numpy.zeros((3,1))
        self.xi0 = numpy.asarray((0.0,)).reshape((-1,1))
        self.Pxi0 = numpy.eye(1)
        self.z0 = numpy.zeros((3,))
        self.Pz0 = numpy.eye(3)
        # Linear states handled by base-class
        super(ParticleLS2,self).__init__(Az=Az, C=C, fz=fz, Axi=Axi,
                                         R=R, Qxi=Qxi, Qz=Qz,
                                         params=params)

    def create_initial_estimate(self, N):
        particles = numpy.empty((N,), dtype=numpy.ndarray)
               
        for i in xrange(N):
            particles[i] = numpy.empty(1+3+3*3)
            particles[i][0] = numpy.random.multivariate_normal(self.xi0.ravel(), self.Pxi0)
            particles[i][1:4] = numpy.copy(self.z0).ravel()
            particles[i][4:] = numpy.copy(self.Pz0).ravel()  
        return particles        

    def get_rb_initial(self, xi0):
        return (numpy.copy(self.z0),
                numpy.copy(self.Pz0))

        
    def get_nonlin_pred_dynamics(self, particles, u):
        xil = numpy.vstack(particles)[:,0]
        fxil = self.params[0]*numpy.arctan(xil)
        return (None, fxil[:,numpy.newaxis,numpy.newaxis], None)
        
    def get_meas_dynamics(self, y, particles):
        N = len(particles)
        xil = numpy.vstack(particles)[:,0]
        h = numpy.zeros((N,2,1))
        h[:,0,0] = 0.1*numpy.fabs(xil)*xil
        return (numpy.asarray(y).reshape((-1,1)), None, h, None)

    def set_states(self, particles, xi_list, z_list, P_list):
        """ Set the estimate of the Rao-Blackwellized states """
        N = len(particles)
        for i in xrange(N):
            particles[i][0:1] = xi_list[i].ravel()
            particles[i][1:4] = z_list[i].ravel()
            particles[i][4:] = P_list[i].ravel()
 
    def get_states(self, particles):
        """ Return the estimate of the Rao-Blackwellized states.
            Must return two variables, the first a list containing all the
            expected values, the second a list of the corresponding covariance
            matrices"""
        N = len(particles)
        xil = list()
        zl = list()
        Pl = list()
        for i in xrange(N):
            xil.append(particles[i][0].reshape(1,1))
            zl.append(particles[i][1:4].reshape(3,1))
            Pl.append(particles[i][4:].reshape(3,3))
        
        return (xil, zl, Pl)

    # Override this method since there is no uncertainty in z0    
    def eval_logp_xi0(self, xil):
        """ Calculate gradient of a term of the I1 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""    
            
        N = len(xil)
        return kalman.lognormpdf_vec(xil, N*(self.xi0,), N*(self.Pxi0,))

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
        self.set_dynamics(Axi=Axi, Az=Az)
