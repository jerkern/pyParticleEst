""" Collection of functions and classes used for Particle Filtering/Smoothing """
import abc
import numpy
import copy

class ParticleFiltering(object):
    """ Base class for particles to be used with particle filtering.
        particles are a model specific array where the first dimension
        indexes the different particles.  """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def create_initial_estimate(self, N):
        """ Sample N particle from initial distribution """
        pass
     
    @abc.abstractmethod
    def sample_process_noise(self, particles, u, t):
        """ Return process noise for input u """
        pass
    
    @abc.abstractmethod
    def update(self, particles, u, t, noise):
        """ Update estimate using 'data' as input """
        pass
    
    @abc.abstractmethod    
    def measure(self, particles, y, t):
        """ Return the log-pdf value of the measurement """
        pass
    
    def copy_ind(self, particles, new_ind=None):
        """ copy the particles from new_ind. Override if model specific particle representation
            does not allow indexing the particle using the first dimension. """
        if (new_ind != None):
            N = len(new_ind)
        else:
            N = len(particles)
            new_ind = range(N)
        new_part = numpy.empty(N, type(particles[0]))
        for k in range(numpy.shape(new_ind)[0]):
            new_part[k] = copy.copy(particles[new_ind[k]])
        return new_part
    
class AuxiliaryParticleFiltering(object):
    """ Base class for particles to be used with particle filtering """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def eval_1st_stage_weights(self, particles, u, y, t):
        """ Evaluate "first stage weights" for the auxiliary particle filter.
            (log-probability of measurement using some propagated statistic, such
            as the mean, for the future state) """
        pass
    
class FFBSi(ParticleFiltering):
    """ Base class for particles to be used with particle smoothing """
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def logp_xnext(self, particles, next_part, u, t):
        """ Return the log-pdf value for the possible future state 'next' given input u """
        pass
    
    def logp_xnext_full(self, particles, future_trajs, u, t):
        """ Return the log-pdf value for the entire future trajectory. Useful for non-markovian
            modeles, that result e.g marginalised state-space models """
            
        # Default implemenation for markovian models, just look at the next state
        return self.logp_xnext(particles, future_trajs[:,0], u[0], t[0])
    
    def sample_smooth(self, particles, next_part, u, y, t):
        """ Return representation of smoothed particle estimates, useful for calulating
            e.g sufficient statistics backward in time """
        # default implementation uses the same format as forward in time
        return numpy.copy(particles)

class FFBSiRS(FFBSi):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def logp_xnext_max(self, particles, u, t):
        """ Return the log-pdf value for the possible future state 'next' given input u """
        pass
    
class SampleProposer(object):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def propose_smooth(self, partp, up, tp, u, y, t, partn):
        """ Sample from a distrubtion q(x_t | x_{t-1}, x_{t+1}, y_t) """
        pass
    @abc.abstractmethod
    def logp_proposal(self, prop_part, partp, up, tp, u, y, t, partn):
        """ Eval log q(x_t | x_{t-1}, x_{t+1}, y_t) """
        pass