""" Collection of functions and classes used for Particle Filtering/Smoothing """
import pyparticleest.interfaces as interfaces
import scipy.linalg
import numpy.random
import math
import abc
import pyparticleest.utils.kalman as kalman
from exceptions import ValueError

class NonlinearGaussian(interfaces.FFBSiRS):
    """ Base class for particles of the type mixed linear/non-linear with 
        additive gaussian noise.
    
        Implement this type of system by extending this class and provide 
        the methods for returning the system matrices at each time instant.
        
        This class currently doesn't support analytic gradients when
        performing parameter estimation, however using numerical gradients
        is typically fine """

    __metaclass__ = abc.ABCMeta

    def get_f(self, particles, u, t):
        return None

    def get_Q(self, particles, u, t):
        return None

    def get_g(self, particles, t):
        return None

    def get_R(self, particles, t):
        return None

    def __init__(self, lxi, f=None, g=None, Q=None, R=None):
        if (f != None):
            self.f = numpy.copy(f)
        else:
            self.f = None
        if (g != None):
            self.g = numpy.copy(g)
        else:
            self.g = None
        if (Q != None):
            self.Qchol = scipy.linalg.cho_factor(Q)
            self.Qcholtri = numpy.triu(self.Qchol[0])
            ld = numpy.sum(numpy.log(numpy.diag(self.Qchol[0]))) * 2
            self.logpdfmax = -0.5 * (lxi * math.log(2 * math.pi) + ld)
        if (R != None):
            self.Rchol = scipy.linalg.cho_factor(R)
            self.Rcholtri = numpy.triu(self.Rchol[0])

        self.lxi = lxi

    def sample_process_noise(self, particles, u, t):
        """ Return sampled process noise for the non-linear states """
        N = len(particles)
        Q = self.get_Q(particles=particles, u=u, t=t)
        noise = numpy.random.normal(size=(self.lxi, N))
        if (Q == None):
            noise = self.Qcholtri.dot(noise)
        else:
            for i in xrange(N):
                Qchol = numpy.triu(scipy.linalg.cho_factor(Q[i], check_finite=False)[0])
                noise[:, i] = Qchol.dot(noise[:, i])

        return noise.T

    def update(self, particles, u, t, noise):
        """ Update estimate using 'data' as input """
        f = self.get_f(particles=particles, u=u, t=t)
        if (f == None):
            f = self.f
        particles[:] = f + noise
        return particles

    def measure(self, particles, y, t):
        """ Return the log-pdf value of the measurement """
        g = self.get_g(particles=particles, t=t)
        R = self.get_R(particles=particles, t=t)
        N = len(particles)
        lpy = numpy.empty(N)
        if (g == None):
            g = numpy.repeat(self.g, N, 0)
        diff = y - g
        if (R == None):
            if (self.Rcholtri.shape[0] == 1):
                lpy = kalman.lognormpdf_scalar(diff, self.Rcholtri)
            else:
                lpy = kalman.lognormpdf_cho_vec(diff, self.Rchol)
        else:
            lpy = numpy.empty(N)
            for i in xrange(N):
                Rchol = scipy.linalg.cho_factor(R[i], check_finite=False)
                lpy[i] = kalman.lognormpdf_cho(diff[i], Rchol)

        return lpy

    def eval_1st_stage_weights(self, particles, u, y, t):
        """ For auxiliary particle filtering, predict likelihood of next
            measurement """
        part = numpy.copy(particles)
        noise = numpy.zeros_like(part)
        partn = self.update(part, u, t, noise)
        return self.measure(partn, y, t + 1)

    def logp_xnext_max(self, particles, u, t):
        """ For rejection sampling, maximum value of the logp_xnext function """
        Q = self.get_Q(particles, u, t)
        dim = self.lxi
        l2pi = math.log(2 * math.pi)
        if (Q == None):
            return self.logpdfmax
        else:
            N = len(particles)
            pmax = numpy.empty(N)
            for i in xrange(N):
                Qchol = scipy.linalg.cho_factor(Q[i], check_finite=False)
                ld = numpy.sum(numpy.log(numpy.diag(Qchol[0]))) * 2
                pmax[i] = -0.5 * (dim * l2pi + ld)
            return numpy.max(pmax)

    def logp_xnext(self, particles, next_part, u, t):
        """ Implements the logp_xnext function for NonlinearGaussian models """

        f = self.get_f(particles, u, t)
        if (f == None):
            f = self.f
        diff = next_part - f
        Q = self.get_Q(particles, u, t)
        if (Q == None):
            if (self.Qcholtri.shape[0] == 1):
                lpx = kalman.lognormpdf_scalar(diff, self.Qcholtri)
            else:
                lpx = kalman.lognormpdf_cho_vec(diff, self.Qchol)
        else:
            N = len(particles)
            lpx = numpy.empty(N)
            for i in xrange(N):
                Qchol = scipy.linalg.cho_factor(Q[i], check_finite=False)
                lpx[i] = kalman.lognormpdf_cho(diff[i], Qchol)

        return lpx

    def sample_smooth(self, particles, future_trajs, ut, yt, tt):
        """ Implements the sample_smooth function for MixedNLGaussian models """
        return particles

    def propose_smooth(self, partp, up, tp, ut, yt, tt, future_trajs):
        """ Sample from a distrubtion q(x_t | x_{t-1}, x_{t+1:T}, y_t:T) """
        # Trivial choice of q, discard y_T and x_{t+1}
        if (partp != None):
            noise = self.sample_process_noise(partp, up, tp)
            prop_part = numpy.copy(partp)
            prop_part = self.update(prop_part, up, tp, noise)
        else:
            prop_part = self.create_initial_estimate(future_trajs.shape[1])
        return prop_part

    def logp_proposal(self, prop_part, partp, up, tp, ut, yt, tt, future_trajs):
        """ Eval log q(x_t | x_{t-1}, x_{t+1:T}, y_t) """
        if (partp != None):
            return self.logp_xnext(partp, prop_part, up, tp)
        else:
            return self.eval_logp_x0(prop_part, t=tt[0])

    def set_params(self, params):
        self.params = numpy.copy(params).reshape((-1, 1))



class NonlinearGaussianInitialGaussian(NonlinearGaussian):
    def __init__(self, x0=None, Px0=None, lxi=None, **kwargs):

        if (x0 != None):
            self.x0 = numpy.copy(x0).reshape((-1, 1))
        elif (lxi != None):
            self.x0 = numpy.zeros((lxi, 1))
        elif (Px0 != None):
            self.x0 = numpy.zeros((Px0.shape[0], 1))
        else:
            raise ValueError()

        if (Px0 == None):
            self.Px0 = numpy.zeros((len(self.x0), len(self.x0)))
        else:
            self.Px0 = numpy.copy((Px0))

        super(NonlinearGaussianInitialGaussian, self).__init__(lxi=len(self.x0),
                                                               **kwargs)

    def create_initial_estimate(self, N):
        particles = numpy.repeat(self.x0, N, 1).T
        if (numpy.any(self.Px0)):
            Pchol = scipy.linalg.cho_factor(self.Px0)[0]
            noise = numpy.random.normal(size=(self.lxi, N))
            particles += (Pchol.dot(noise)).T
        return particles

    def eval_logp_x0(self, particles, t):
        """ Calculate gradient of a term of the I1 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""

        N = len(particles)
        res = numpy.empty(N)
        # Assumes Px0 is either full rang or zero
        if ((self.Px0 == 0.0).all()):
            x0 = self.x0.ravel()
            for i in xrange(N):
                if (numpy.array_equiv(particles[i], x0)):
                    res[i] = 0.0
                else:
                    res[i] = -numpy.Inf
        else:
            Pchol = scipy.linalg.cho_factor(self.Px0, check_finite=False)
            for i in xrange(N):
                res[i] = kalman.lognormpdf_cho(particles[i] - self.x0, Pchol)

        return res
