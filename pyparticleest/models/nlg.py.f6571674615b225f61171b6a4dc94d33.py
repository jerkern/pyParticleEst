""" Model definition for base class for Nonlinear Gaussian systems

@author: Jerker Nordh
"""
import pyparticleest.interfaces as interfaces
import scipy.linalg
import numpy.random
import math
import abc
try:
    import pyparticleest.utils.ckalman as kalman
except ImportError:
    print("Falling back to pure python implementaton, expect horrible performance")
    import pyparticleest.utils.kalman as kalman


from builtins import range


class NonlinearGaussian(interfaces.ParticleFiltering, interfaces.FFBSiRS):
    """
    Base class for particles of the type mixed linear/non-linear with
    additive gaussian noise.

    Implement this type of system by extending this class and provide
    the methods for returning the system matrices at each time instant.

    x_{t+1} = f(x_t, u_t) + v, v ~ N(0, Q(x_t, u_t))
    y_t = g(x_t) + e, e ~ N(=, R(x_t))

    This class currently doesn't support analytic gradients when
    performing parameter estimation, however using numerical gradients
    is typically fine

    Args:
     - lxi (int): number of states in model
     - f (array-like): f (if constaint)
     - g (array-like): g (if constaint)
     - Q (array-like): Q (if constaint)
     - R (array-like): R (if constaint)
     """

    __metaclass__ = abc.ABCMeta

    def calc_f(self, particles, u, t):
        """
        Calucate f

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - u (array-like): input signal
         - t (float): time stamp

        Returns:
         (array-like): f for all particles
        """
        return None

    def calc_Q(self, particles, u, t):
        """
        Calucate Q

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - u (array-like): input signal
         - t (float): time stamp

        Returns:
         (array-like): Q for all particles
        """
        return None

    def calc_g(self, particles, t):
        """
        Calucate g

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - t (float): time stamp

        Returns:
         (array-like): g for all particles
        """
        return None

    def calc_R(self, particles, t):
        """
        Calucate R

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - t (float): time stamp

        Returns:
         (array-like): R for all particles
        """
        return None

    def __init__(self, lxi, f=None, g=None, Q=None, R=None):
        if (f is not None):
            self.f = numpy.copy(f)
        else:
            self.f = None
        if (g is not None):
            self.g = numpy.copy(g)
        else:
            self.g = None
        if (Q is not None):
            self.Qchol = scipy.linalg.cho_factor(Q)
            self.Qcholtri = numpy.triu(self.Qchol[0])
            ld = numpy.sum(numpy.log(numpy.diag(self.Qchol[0]))) * 2
            self.logpdfmax = -0.5 * (lxi * math.log(2 * math.pi) + ld)
        if (R is not None):
            self.Rchol = scipy.linalg.cho_factor(R)
            self.Rcholtri = numpy.triu(self.Rchol[0])

        self.lxi = lxi

    def set_Q(self, Q):
        self.Qchol = scipy.linalg.cho_factor(Q)
        self.Qcholtri = numpy.triu(self.Qchol[0])
        ld = numpy.sum(numpy.log(numpy.diag(self.Qchol[0]))) * 2
        self.logpdfmax = -0.5 * (self.lxi * math.log(2 * math.pi) + ld)

    def set_R(self, R):
        self.Rchol = scipy.linalg.cho_factor(R)
        self.Rcholtri = numpy.triu(self.Rchol[0])

    def sample_process_noise(self, particles, u, t):
        """
        Sample process noise

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - u (array-like):  input signal
         - t (float): time-stamp

        Returns:
         (array-like) with first dimension = N
        """
        N = len(particles)
        Q = self.calc_Q(particles=particles, u=u, t=t)
        noise = numpy.random.normal(size=(self.lxi, N))
        if (Q is None):
            noise = self.Qcholtri.T.dot(noise)
        else:
            for i in range(N):
                Qchol = numpy.triu(scipy.linalg.cho_factor(Q[i], check_finite=False)[0])
                noise[:, i] = Qchol.T.dot(noise[:, i])

        return noise.T

    def update(self, particles, u, t, noise):
        """ Propagate estimate forward in time

        Args:

         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - u (array-like):  input signal
         - t (float): time-stamp
         - noise (array-like): noise realization used for the calucations
           , with first dimension = N (number of particles)

        Returns:
         (array-like) with first dimension = N, particle estimate at time t+1
        """
        f = self.calc_f(particles=particles, u=u, t=t)
        if (f is None):
            f = self.f
        particles[:] = f + noise
        return particles

    def measure(self, particles, y, t):
        """
        Return the log-pdf value of the measurement

        Args:

         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - y (array-like):  measurement
         - t (float): time-stamp

        Returns:
         (array-like) with first dimension = N, logp(y|x^i)
        """
        N = len(particles)
        lpy = numpy.empty(N)
        g = self.calc_g(particles=particles, t=t)
        R = self.calc_R(particles=particles, t=t)

        if (g is None):
            g = numpy.repeat(self.g.reshape((1, -1, 1)), N, 0)
        else:
            g = g.reshape((N, -1, 1))
        yrep = numpy.repeat(numpy.asarray(y).reshape((1, -1, 1)), N, 0)
        diff = yrep - g
        if (R is None):
            if (self.Rcholtri.shape[0] == 1):
                lpy = kalman.lognormpdf_scalar(diff, self.Rcholtri)
            else:
                lpy = kalman.lognormpdf_cho_vec(diff, self.Rchol)
        else:
            lpy = numpy.empty(N)
            for i in range(N):
                Rchol = scipy.linalg.cho_factor(R[i], check_finite=False)
                lpy[i] = kalman.lognormpdf_cho(diff[i], Rchol)

        return lpy

    def eval_1st_stage_weights(self, particles, u, y, t):
        """
        Evaluate "first stage weights" for the auxiliary particle filter.
        (log-probability of measurement using some propagated statistic, such
        as the mean, for the future state)

        Args:

         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - u (array-like): input signal
         - y (array-like):  measurement
         - t (float): time-stamp

        Returns:
         (array-like) with first dimension = N, logp(y_{t+1}|\hat{x}_{t+1|t}^i)
        """
        part = numpy.copy(particles)
        noise = numpy.zeros_like(part)
        partn = self.update(part, u, t, noise)
        return self.measure(partn, y, t + 1)

    def logp_xnext_max(self, particles, u, t):
        """
        Return the max log-pdf value for all possible future states'
        given input u

        Args:

         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - next_part (array-like): particle estimate for t+1
         - u (array-like): input signal
         - t (float): time stamps

        Returns:
         (array-like) with first dimension = N, argmax_{x_{t+1}} logp(x_{t+1}|x_t)
        """
        Q = self.calc_Q(particles, u, t)
        dim = self.lxi
        l2pi = math.log(2 * math.pi)
        if (Q is None):
            return self.logpdfmax
        else:
            N = len(particles)
            pmax = numpy.empty(N)
            for i in range(N):
                Qchol = scipy.linalg.cho_factor(Q[i], check_finite=False)
                ld = numpy.sum(numpy.log(numpy.diag(Qchol[0]))) * 2
                pmax[i] = -0.5 * (dim * l2pi + ld)
            return numpy.max(pmax)

    def logp_xnext(self, particles, next_part, u, t):
        """
        Return the log-pdf value for the possible future state 'next'
        given input u

        Args:

         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - next_part (array-like): particle estimate for t+1
         - u (array-like): input signal
         - t (float): time stamps

        Returns:
         (array-like) with first dimension = N, logp(x_{t+1}|x_t^i)
        """

        f = self.calc_f(particles, u, t)
        if (f is None):
            f = self.f
        diff = next_part - f
        Q = self.calc_Q(particles, u, t)
        if (Q is None):
            if (self.Qcholtri.shape[0] == 1):
                lpx = kalman.lognormpdf_scalar(diff, self.Qcholtri)
            else:
                lpx = kalman.lognormpdf_cho_vec(diff, self.Qchol)
        else:
            N = len(particles)
            lpx = numpy.empty(N)
            for i in range(N):
                Qchol = scipy.linalg.cho_factor(Q[i], check_finite=False)
                lpx[i] = kalman.lognormpdf_cho(diff[i], Qchol)

        return lpx

    def propose_smooth(self, ptraj, anc, future_trajs, find, yt, ut, tt, cur_ind):
        """
        Sample from a distribution q(x_t | x_{0:t-1}, x_{t+1:T}, y_t:T)

        Args:
         - ptraj: array of trajectory step objects from previous time-steps,
           last index is step just before the current
         - anc (array-like): index of the ancestor of each particle in part
         - future_trajs (array-like): particle estimate for {t+1:T}
         - find (array-like): index in future_trajs corresponding to each
           generated sample
         - ut (array-like): input signals for {0:T}
         - yt (array-like): measurements for {0:T}
         - tt (array-like): time stamps for {0:T}
         - cur_ind (int): index of current timestep (in ut, yt and tt)

        Returns:
         (array-like) of dimension N, wher N is the dimension of partp and/or
         future_trajs (one of which may be 'None' at the start/end of the dataset)
        """
        # Trivial choice of q, discard y_T and x_{t+1}
        if (ptraj is not None):
            prop_part = numpy.copy(ptraj[-1].pa.part[anc])
            noise = self.sample_process_noise(prop_part, ut[cur_ind - 1], tt[cur_ind - 1])
            prop_part = self.update(prop_part, ut[cur_ind - 1], tt[cur_ind - 1], noise)
        else:
            prop_part = self.create_initial_estimate(len(find))
        return prop_part

    def logp_proposal(self, prop_part, ptraj, anc, future_trajs, find, yt, ut, tt, cur_ind):
        """
        Eval the log-propability of the proposal distribution

        Args:
         - prop_part (array-like): Proposed particle estimate, first dimension
           has length = N
         - ptraj: array of trajectory step objects from previous time-steps,
           last index is step just before the current
         - anc (array-like): index of the ancestor of each particle in part
         - future_trajs (array-like): particle estimate for {t+1:T}
         - find (array-like): index in future_trajs corresponding to each
           generated sample
         - ut (array-like): input signals for {0:T}
         - yt (array-like): measurements for {0:T}
         - tt (array-like): time stamps for {0:T}
         - cur_ind (int): index of current timestep (in ut, yt and tt)

        Returns
         (array-like) with first dimension = N,
         log q(x_t | x_{t-1}, x_{t+1:T}, y_t:T)
        """
        if (ptraj is not None):
            return self.logp_xnext(ptraj[-1].pa.part[anc], prop_part, ut[cur_ind - 1], tt[cur_ind - 1])
        else:
            return self.eval_logp_x0(prop_part, t=tt[0])

    def set_params(self, params):
        """
        This methods should be overriden if the system dynamics depends
        on any parameters, this method should however be called to store
        the new parameter values correctly

        Args:
         - params (array-like): new parameter values
        """
        self.params = numpy.copy(params).reshape((-1, 1))

    def post_smoothing(self, st):
        return self.pre_mhips_pass(st)

    def pre_mhips_pass(self, st):
        return st.traj


class NonlinearGaussianInitialGaussian(NonlinearGaussian):
    """
    Nonlinear gaussian system with initial Gaussian distribution.

    Args:
     - x0 (array-like): mean value of initial state, defaults to 0
     - Px0 (array-like): covariance of initial state, defaults to 0
     - lxi (int): number of states, only needed if neither x0 or Px0 specified
    """

    def __init__(self, x0=None, Px0=None, lxi=None, **kwargs):

        if (x0 is not None):
            self.x0 = numpy.copy(x0).reshape((-1, 1))
        elif (lxi is not None):
            self.x0 = numpy.zeros((lxi, 1))
        elif (Px0 is not None):
            self.x0 = numpy.zeros((Px0.shape[0], 1))
        else:
            raise ValueError()

        if (Px0 is None):
            self.Px0 = numpy.zeros((len(self.x0), len(self.x0)))
        else:
            self.Px0 = numpy.copy((Px0))

        super(NonlinearGaussianInitialGaussian, self).__init__(lxi=len(self.x0),
                                                               **kwargs)

    def create_initial_estimate(self, N):
        """Sample particles from initial distribution

        Args:
         - N (int): Number of particles to sample

        Returns:
         (array-like) with first dimension = N, model specific representation
         of all particles """
        particles = numpy.repeat(self.x0, N, 1).T
        if (numpy.any(self.Px0)):
            Pchol = scipy.linalg.cho_factor(self.Px0)[0]
            noise = numpy.random.normal(size=(self.lxi, N))
            particles += (Pchol.dot(noise)).T
        return particles

    def eval_logp_x0(self, particles, t):
        """
        Evaluate log p(x_0)

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - t (float): time stamp
        """

        N = len(particles)
        res = numpy.empty(N)
        # Assumes Px0 is either full rang or zero
        if ((self.Px0 == 0.0).all()):
            x0 = self.x0.ravel()
            for i in range(N):
                if (numpy.array_equiv(particles[i], x0)):
                    res[i] = 0.0
                else:
                    res[i] = -numpy.Inf
        else:
            Pchol = scipy.linalg.cho_factor(self.Px0, check_finite=False)
            for i in range(N):
                res[i] = kalman.lognormpdf_cho(particles[i].ravel() - self.x0.ravel(), Pchol)

        return res
