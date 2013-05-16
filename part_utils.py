""" Collection of functions and classes used for Particle Filtering/Smoothing """
import abc
import kalman
import numpy

class ParticleFilteringInterface(object):
    """ Base class for particles to be used with particle filtering """
    __metaclass__ = abc.ABCMeta
     
    @abc.abstractmethod
    def sample_input_noise(self, u):
        """ Return a noise perturbed input vector u """
        return
    
    @abc.abstractmethod
    def update(self, data):
        """ Update estimate using 'data' as input """
        return
    
    @abc.abstractmethod    
    def measure(self, y):
        """ Return the log-pdf value of the measurement """
        return
    

class RBPFBase(ParticleFilteringInterface):
    """ Base class for Rao-Blackwellized particles """
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, z0, P0, 
                 Az=None, Bz=None, C=None, D=None,
                  Qz=None, R=None):
        self.kf = kalman.KalmanSmoother(x0=z0, P0=P0,
                                        A=Az, B=Bz,C=C, D=D, 
                                        Q=Qz, R=R)
    
    def set_dynamics(self, Az=None, Bz=None, C=None, D=None, Qz=None, R=None):
        
        if (Az):
            self.kf.A = Az
        if (Bz):
            self.kf.B = Bz
        if (C):
            self.kf.C = C
        if (D):
            self.kf.D = D
        if (Qz):
            self.kf.Q = Qz
        if (R):
            self.kf.R = R
    
    @abc.abstractmethod
    def update(self, data):
        """ Update estimate using 'data' as input """
        self.kf.time_update(u=data)
    
    @abc.abstractmethod    
    def measure(self, y):
        """ Return the log-pdf value of the measurement """
        return self.kf.meas_update(y=y)


class ParticleSmoothingInterface(ParticleFilteringInterface):
    """ Base class for particles to be used with particle smoothing """
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def next_pdf(self, next_cpart, u):
        """ Return the log-pdf value for the possible future state 'next' given input u """
        return
    
    @abc.abstractmethod
    def sample_smooth(self, filt_traj, ind, next_cpart):
        """ Return a collapsed particle with the rao-blackwellized states sampled """
        return
    
    @abc.abstractmethod
    def collapse(self):
        """ Return a sample of the particle where the rao-blackwellized states
        are drawn from the MVN that results from CLGSS structure """
        return
    
class RBPSBase(RBPFBase, ParticleSmoothingInterface):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, z0, P0, 
                 Az=None, Bz=None, C=None, D=None,
                 Qz=None, R=None):
        super(RBPSBase,self).__init__(z0=z0, P0=P0, Az=Az, Bz=Bz, C=C, D=D,
                                      Qz=Qz, R=R)
        
    
    def clin_update(self, u):
        """ Kalman update of the linear states conditioned on the non-linear trajectory estimate """
        A = self.kf.A
        B = self.kf.B
        C = self.kf.C
        Q = self.kf.Q
        R = self.kf.R
        x0 = self.kf.x_new
        P = self.kf.P

        kf = kalman.KalmanFilter(A=A,B=B,C=C, x0=numpy.reshape(x0,(-1,1)), P0=P, Q=Q, R=R)
        kf.time_update(u=u)
        
        return (kf.x_new.reshape((-1,1)), kf.P)
    
    def clin_measure(self, y):
        """ Kalman measuement of the linear states conditioned on the non-linear trajectory estimate """
        A = self.kf.A
        B = self.kf.B
        C = self.kf.C
        Q = self.kf.Q
        R = self.kf.R
        x0 = self.kf.x_new
        P = self.kf.P

        kf = kalman.KalmanFilter(A=A,B=B,C=C, x0=numpy.reshape(x0,(-1,1)), P0=P, Q=Q, R=R)
        kf.meas_update(y)
        
        return (kf.x_new.reshape((-1,1)), kf.P)

    def clin_smooth(self, z_next, u):
        """ Kalman smoothing of the linear states conditioned on the next particles linear states """ 
        A = self.kf.A
        B = self.kf.B
        C = self.kf.C
        Q = self.kf.Q
        R = self.kf.R
        x0 = self.kf.x_new
        P = self.kf.P

        kf = kalman.KalmanSmoother(A=A,B=B,C=C, x0=numpy.reshape(x0,(-1,1)), P0=P, Q=Q, R=R)
        kf.smooth(z_next[0], z_next[1], u)
        
        return (kf.x_new.reshape((-1,1)), kf.P)

    @abc.abstractmethod
    def set_nonlin_state(self, eta):
        """ Set the non-linear state estimates """
        return
    
    @abc.abstractmethod
    def get_nonlin_state(self):
        """ Return the non-linear state estimates """
        return
    
    @abc.abstractmethod
    def set_lin_est(self, lest):
        """ Set the estimate of the rao-blackwellized states """
        return
 
    @abc.abstractmethod
    def get_lin_est(self):
        """ Return the estimate of the rao-blackwellized states """
        return
 
    @abc.abstractmethod
    def linear_input(self, u):
        """ Extract the part of u affect the conditionally rao-blackwellized states """
        return
    
    

