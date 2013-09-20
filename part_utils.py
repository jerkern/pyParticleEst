""" Collection of functions and classes used for Particle Filtering/Smoothing """
import abc
import kalman
import numpy

class ParticleFilteringInterface(object):
    """ Base class for particles to be used with particle filtering """
    __metaclass__ = abc.ABCMeta
     
    @abc.abstractmethod
    def sample_process_noise(self, u):
        """ Return process noise for input u """
        return
    
    @abc.abstractmethod
    def update(self, u, noise):
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
                 Az=None, Bz=None, C=None,
                  Qz=None, R=None, f_k=None, h_k=None):
        
        self.kf = kalman.KalmanSmoother(z0=z0, P0=P0,
                                        A=Az, C=C, 
                                        Q=Qz, R=R,
                                        f_k=f_k, h_k=h_k)
    
    def set_dynamics(self, Az=None, C=None, Qz=None, R=None, f_k=None, h_k=None):

        self.kf.set_dynamics(A=Az, C=C, Q=Qz, R=R, f_k=f_k, h_k=h_k)

    
    def update(self, u, noise):
        """ Update estimate using noise as input """
        etan = self.calc_next_eta(u, noise)
        # Update linear estimate and dynamics by using etan as measurement
        self.meas_eta_next(etan)
        lin_est = self.cond_predict(etan)
        self.set_nonlin_state(etan)
        self.set_lin_est(lin_est)

    @abc.abstractmethod
    def cond_dynamics(self, eta_next):
        """ Condition dynamics on future state 'eta_next'. """
        pass
    
    @abc.abstractmethod
    def calc_next_eta(self, u, noise):
        pass
    
    @abc.abstractmethod
    def meas_eta_next(self, eta_next):
        """ Update estimate using observation of next state """
        pass

    def cond_predict(self, eta_next=None):
        """ Kalman update of the linear states conditioned on the non-linear trajectory estimate,
            Before calling this method clin_dynamics should be called to update the dynamics
            according to the conditioning on the non-linear trajectory """
        (z, P) = self.kf.predict()
        return (z.reshape((-1,1)), P)

    def clin_update(self, next_part=None):
        self.kf.time_update()
    
    def prep_measure(self, y):
        """ Pre-processing of measurement y """
        return y
    
    def measure(self, y):
        """ Return the log-pdf value of the measurement """
        return self.kf.measure(y=y)

    def prep_update(self, u):
        """ Update dynamics with u as input """
        pass
    
    def set_lin_est(self, lest):
        """ Set the estimate of the Rao-Blackwellized states """
        self.kf.z = numpy.copy(lest[0].reshape((-1,1)))
        self.kf.P = numpy.copy(lest[1])
 
    def get_lin_est(self):
        """ Return the estimate of the Rao-Blackwellized states """
        return (self.kf.z, self.kf.P)
    
    @abc.abstractmethod
    def set_nonlin_state(self, eta):
        """ Set the non-linear state estimates """
        return
    
    @abc.abstractmethod
    def get_nonlin_state(self):
        """ Return the non-linear state estimates """
        

class ParticleSmoothingInterface(ParticleFilteringInterface):
    """ Base class for particles to be used with particle smoothing """
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def next_pdf(self, next_cpart, u):
        """ Return the log-pdf value for the possible future state 'next' given input u """
        return
    
    @abc.abstractmethod
    def sample_smooth(self, next_part):
        """ Update ev. Rao-Blackwellized states conditioned on "next_part" """
        return

    
class RBPSBase(RBPFBase, ParticleSmoothingInterface):
    __metaclass__ = abc.ABCMeta
    
#    def __init__(self, z0, P0, 
#                 Az=None, Bz=None, C=None,
#                  Qz=None, R=None, f_k=None, h_k=None):
#        super(RBPSBase,self).__init__(z0=z0, P0=P0, Az=Az, C=C,
#                                      Qz=Qz, R=R, f_k=f_k, h_k=h_k)
        
    def clin_measure(self, y, next_part=None):
        """ Kalman measurement of the linear states conditioned on the non-linear trajectory estimate """
        self.kf.measure(y)

    def clin_smooth(self, next_part):
        """ Kalman smoothing of the linear states conditioned on the next particles linear states
            Before calling this method clin_dynamics should be called to update the dynamics
            according to the conditioning on the non-linear trajectory """
        tmp = (next_part.get_lin_est())
        self.kf.smooth(tmp[0], tmp[1])




