""" Collection of functions and classes used for Particle Filtering/Smoothing """
import abc

class ParticleFilteringBase(object):
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



class ParticleSmoothingBase(ParticleFilteringBase):
    """ Base class for particles to be used with particle smoothing """
    
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
    
class ParticleSmoothingBaseRB(ParticleSmoothingBase):
    
    @abc.abstractmethod
    def clin_update(self, u):
        """ Kalman update of the linear states conditioned on the non-linear trajectory estimate """
        return
    
    @abc.abstractmethod
    def clin_measure(self, y):
        """ Kalman measuement of the linear states conditioned on the non-linear trajectory estimate """
        return

    @abc.abstractmethod
    def clin_smooth(self, z_next, u):
        """ Kalman smoothing of the linear states conditioned on the next particles linear states """ 
        return

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
    
    

