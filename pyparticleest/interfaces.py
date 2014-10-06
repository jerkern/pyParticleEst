""" Interface specification defining the methods needed for using the different
classes of algorithms present in the framework

@author: Jerker Nordh
"""
import abc
import numpy

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

    def sample_smooth(self, particles, future_trajs, ut, yt, tt):
        """ Return representation of smoothed particle estimates, useful for
            calulating e.g sufficient statistics backward in time """
        # default implementation uses the same format as forward in time
        # Is part of the ParticleFiltering interface since it is used
        # also when calculating "ancestor trajectories"
        return numpy.copy(particles)

    def copy_ind(self, particles, new_ind=None):
        if (new_ind != None):
            return numpy.copy(particles[new_ind])
        else:
            return numpy.copy(particles)

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

    def logp_xnext_full(self, particles, future_trajs, ut, yt, tt):
        """ Return the log-pdf value for the entire future trajectory.
            Useful for non-markovian modeles, that result from e.g 
            marginalized state-space models """

        # Default implemenation for markovian models, just look at the next state
        return self.logp_xnext(particles, next_part=future_trajs[0],
                               u=ut[0], t=tt[0])

    @abc.abstractmethod
    def logp_xnext(self, particles, next_part, u, t):
        """ Return the log-pdf value for the possible future state 'next'
            given input u """
        pass


class FFBSiRS(FFBSi):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def logp_xnext_max(self, particles, u, t):
        """ Return the log-pdf value for the possible future state 'next'
            given input u """
        pass

class SampleProposer(object):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def propose_smooth(self, partp, up, tp, ut, yt, tt, future_trajs):
        """ Sample from a distribution q(x_t | x_{t-1}, x_{t+1:T}, y_t:T) """
        pass

    @abc.abstractmethod
    def logp_proposal(self, prop_part, partp, up, tp, ut, yt, tt, future_trajs):
        """ Eval log q(x_t | x_{t-1}, x_{t+1:T}, y_t:T) """
        pass
