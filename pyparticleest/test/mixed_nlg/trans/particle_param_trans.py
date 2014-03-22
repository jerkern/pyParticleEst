import numpy
import pyparticleest.kalman as kalman
from pyparticleest.part_utils import MixedNLGaussian
import math
        
def generate_data(theta, Qxi, Qz, R, steps):
    # Create reference
    y = list()
    x = list()
    e = numpy.random.normal(0.0, 1.0)
    z = numpy.random.normal(1.0, 1.0)
    
    x.append((numpy.vstack((e,z))))
    y.append(None)
    
    for i in range(steps):
        
        #e = e + theta * z + numpy.random.multivariate_normal((0.0,), Qxi)
        #z = z + numpy.random.multivariate_normal((0.0,), Qz)
        e = e + z + numpy.random.multivariate_normal((0.0,), Qxi)
        z = theta * z + numpy.random.multivariate_normal((0.0,), Qz)
        x.append((numpy.vstack((e,z))))
        y.append(e + numpy.random.multivariate_normal((0.0,),R))

    return (x, y)


class ParticleParamTrans(MixedNLGaussian):
    """ Implement a simple system by extending the MixedNLGaussian class """
    def __init__(self, params, R, Qxi, Qz):
        """ Define all model variables """
        #Axi = numpy.array([[params[0],]])
        Axi = numpy.eye(1.0)
        #Az = numpy.eye(1.0)
        Az = numpy.array([[params[0],]])
        C = numpy.array([[0.0,]])
        
        
        self.z0 = numpy.ones((1,1))
        self.Pz0 = numpy.eye(1)
        self.xi0 = numpy.zeros((1,1))
        self.Pxi0 = numpy.eye(1)
        fz = numpy.zeros((1,1))
        # Linear states handled by base-class
        super(ParticleParamTrans,self).__init__(Az=Az, C=C, Axi=Axi,
                                                R=R, Qxi=Qxi, Qz=Qz,
                                                fz=fz)

    def create_initial_estimate(self, N):
        particles = numpy.empty((N,), dtype=numpy.ndarray)
               
        for i in xrange(N):
            particles[i] = numpy.empty(3)
            particles[i][0] = numpy.random.multivariate_normal(self.xi0.ravel(), self.Pxi0)
            particles[i][1] = numpy.copy(self.z0).ravel()
            particles[i][2] = numpy.copy(self.Pz0).ravel()  
        return particles   
        
    def get_rb_initial(self, xi0):
        return (numpy.copy(self.z0),
                numpy.copy(self.Pz0))

    def eval_logp_xi0(self, xil):
        """ Calculate gradient of a term of the I1 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""    
            
        N = len(xil)
        return kalman.lognormpdf_vec(xil, N*(self.xi0,), N*(self.Pxi0,))
        
    def get_nonlin_pred_dynamics(self, particles, u):
        xil = numpy.vstack(particles)[:,0]
        fxil = xil[:,numpy.newaxis,numpy.newaxis]
        return (None, fxil, None)
        
    def get_meas_dynamics(self, y, particles):
        xil = numpy.vstack(particles)[:,0]
        h = xil[:,numpy.newaxis,numpy.newaxis]
        return (numpy.asarray(y).reshape((-1,1)), None, h, None)
    
    def set_states(self, particles, xi_list, z_list, P_list):
        """ Set the estimate of the Rao-Blackwellized states """
        N = len(particles)
        for i in xrange(N):
            particles[i][0:1] = xi_list[i].ravel()
            particles[i][1:2] = z_list[i].ravel()
            particles[i][2:] = P_list[i].ravel()
 
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
            zl.append(particles[i][1].reshape(1,1))
            Pl.append(particles[i][2].reshape(1,1))
        
        return (xil, zl, Pl)
    
    def set_params(self, params):
        """ New set of parameters """
        # Update all needed matrices and derivates with respect
        # to the new parameter set
        #Axi = numpy.array([[params[0],]])
        #self.set_dynamics(Axi=Axi)
        Az = numpy.array([[params[0],]])
        self.set_dynamics(Az=Az)
