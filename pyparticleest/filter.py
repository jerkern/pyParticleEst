""" Particle filters

@author: Jerker Nordh
"""

import numpy
import math
import copy

def sample(w, n):
    """
    Return n random indices, where the probability if index
    is given by w[i].

    Args:
    - w (array_like): probability weights
    - n (int):  number of indices to sample
    """

    wc = numpy.cumsum(w)
    wc /= wc[-1] # Normalize
    u = (range(n) + numpy.random.rand(1)) / n
    return numpy.searchsorted(wc, u)


class ParticleFilter(object):
    """
    Particle Filter class, creates filter estimates by calling appropriate
    methods in the supplied particle objects and handles resampling when
    a specified threshold is reach.

    Args:
     - model (ParticleFiltering): object describing the model to be used
     - res (float): 0 .. 1 , the ratio of effective number of particles that
       triggers resampling. 0 disables resampling
    """

    def __init__(self, model, res=0):

        self.res = res
        self.model = model

    def forward(self, pa, u, y, t):
        """
        Forward the estimate stored in pa from t to t+1 using the motion model
        with input u at time time and measurement y at time t+1

        Args:
         - pa (ParticleApproximation): approximation for time t
         - u (array-like): input at time t
         - y (array-like): measurement at time t +1
         - t (float): time stamp for time t

        Returns (pa, resampled, ancestors)
         - pa (ParticleApproximation): approximation for time t+1
         - resampled (bool): were the particles resampled
         - ancestors (array-like): anecstral indices for particles at time t+1
        """
        pa = ParticleApproximation(self.model.copy_ind(pa.part), pa.w)

        resampled = False
        if (self.res > 0 and pa.calc_Neff() < self.res * pa.num):
            # Store the ancestor of each resampled particle
            ancestors = pa.resample(self.model, pa.num)
            resampled = True
        else:
            ancestors = numpy.arange(pa.num, dtype=int)

        pa = self.update(pa, u=u, t=t)
        if (y != None):
            pa = self.measure(pa, y=y, t=t + 1)
        return (pa, resampled, ancestors)

    def update(self, pa, u, t, inplace=True):
        """
        Update particle approximation of x_t to x_{t+1} using u as input.

        Args:
         - pa (ParticleApproximation): approximation for time t
         - u (array-like): input at time t
         - t (float): time stamp for time t
         - inplace (bool): if True the particles are updated then returned,
           otherwise a new ParticleApproximation is first created
           leaving the original one intact

        Returns:
            ParticleApproximation for time t+1
        """

        u = numpy.reshape(u, (-1, 1))

        # Prepare the particle for the update, eg. for
        # mixed linear/non-linear calculate the variables that
        # depend on the current state
#        for k in range(pa.num):
#            pa.part[k].prep_update(u)

        if (not inplace):
            pa_out = copy.deepcopy(pa)
            pa = pa_out

        v = self.model.sample_process_noise(particles=pa.part, u=u, t=t)
        self.model.update(particles=pa.part, u=u, t=t, noise=v)

        return pa


    def measure(self, pa, y, t, inplace=True):
        """
        Evaluate and update particle approximation using new measurement y

        Args:
         - pa (ParticleApproximation): approximation for time t
         - y (array-like): measurement at time t +1
         - t (float): time stamp for time t
         - inplace (bool): if True the particles are updated then returned,
           otherwise a new ParticleApproximation is first created
           leaving the original one intact

        Returns:
            ParticleApproximation for time t
        """

        if (not inplace):
            pa_out = copy.deepcopy(pa)
            pa = pa_out

        new_weights = self.model.measure(particles=pa.part, y=y, t=t)

        # Try to keep weights from going to -Inf
        m = numpy.max(new_weights)
        pa.w_offset += m
        new_weights -= m

        pa.w = pa.w + new_weights

        # Keep the weights from going to -Inf
        m = numpy.max(pa.w)
        pa.w_offset += m
        pa.w -= m

        return pa


class AuxiliaryParticleFilter(ParticleFilter):
    """ Auxiliary Particle Filer class, creates filter estimates by calling appropriate
        methods in the supplied particle objects and handles resampling when
        a specified threshold is reach """

    def forward(self, pa, u, y, t):
        """
        Use the first stage weights to try to predict which particles will be in
        regions of high likely hood at time t+1, use this information to resample
        the particles before propagating them forward in time

        Args:
         - pa (ParticleApproximation): approximation for time t
         - u (array-like): input at time t
         - y (array-like): measurement at time t +1
         - t (float): time stamp for time t

        Returns (pa, resampled, ancestors)
         - pa (ParticleApproximation): approximation for time t+1
         - resampled (bool): were the particles resampled
         - ancestors (array-like): anecstral indices for particles at time t+1
        """

        pa = copy.deepcopy(pa)
        resampled = False

        if (y != None):
            l1w = self.model.eval_1st_stage_weights(pa.part, u, y, t)
            pa.w += l1w
            pa.w -= numpy.max(pa.w)

        if (self.res and pa.calc_Neff() < self.res * pa.num):
            ancestors = pa.resample(self.model, pa.num)
            resampled = True
            l1w = l1w[ancestors]
        else:
            ancestors = numpy.arange(pa.num, dtype=int)

        pa = self.update(pa, u=u, t=t)

        if (y != None):
            pa.w += self.model.measure(particles=pa.part, y=y, t=t + 1)
            pa.w -= l1w
            pa.w -= numpy.max(pa.w)

        return (pa, resampled, ancestors)


class TrajectoryStep(object):
    """
    Store particle approximation, input, output and timestamp for
    a single time index in a trajectory

    Args:
     - pa (ParticleAppromixation): particle approximation
     - u (array-like): input signals at time t
       (u[t] contains the input for takin x[t] to x[t+1])
     - y (array-like): measurements at time t
       (y[t] is the measurment of x[t])
     - t (float): time stamp for time t
     - ancestors (array-like): indices for each particles ancestor
    """
    def __init__(self, pa, u=None, y=None, t=None, ancestors=None):
        self.pa = pa
        self.u = u
        self.y = y
        self.t = t
        self.ancestors = ancestors

class ParticleTrajectory(object):
    """
    Store particle trajectories, each time instance is saved
    as a TrajectoryStep object

    Args:
     - model (ParticleFiltering): object describing the model specfics
     - N (int): Number of particles to use in the filter
     - resample (float): Ratio of number of effective particle of total number
       of particles that will trigger resampling
     - t0 (float): time stamp for intial time
     - filter (string): Which filtering algorihms to use
    """

    def __init__(self, model, N, resample=2.0 / 3.0, t0=0, filter='PF'):
        particles = model.create_initial_estimate(N)
        pa = ParticleApproximation(particles=particles)
        self.traj = [TrajectoryStep(pa, t=t0, ancestors=numpy.arange(N)), ]

        if (filter.lower() == 'pf'):
            self.pf = ParticleFilter(model=model, res=resample)
        elif (filter.lower() == 'apf'):
            self.pf = AuxiliaryParticleFilter(model=model, res=resample)
        else:
            raise ValueError('Bad filter type')
        return

    def forward(self, u, y):
        """
        Append new time step to traectory

        Args:
         - u (array-like): Input to go from x_t -> x_{t+1}
         - y (array-like): Measurement of x_{t+1}

        Returns:
         (bool) True if the particle approximation was resampled
        """
        self.traj[-1].u = u
        (pa_nxt, resampled, ancestors) = self.pf.forward(self.traj[-1].pa, u=u, y=y, t=self.traj[-1].t)
        self.traj.append(TrajectoryStep(pa_nxt, t=self.traj[-1].t + 1, y=y, ancestors=ancestors))
        return resampled

    def measure(self, y):
        """
        Update estimate using new measurement

        Args:
         - y (array-like): Measurement at current time index

        Returns:
         None
        """
        self.traj[-1].y = y
        self.pf.measure(self.traj[-1].pa, y=y, t=self.traj[-1].t, inplace=True)

    def prep_rejection_sampling(self):
        """
        Find the maximum over all inputs of the pdf for the next timestep,
        used for rejection sampling in the particle smoother
        """
        for k in range(len(self.traj) - 1):
            self.traj[k].peak_fwd_density = self.traj[k].pa.calc_fwd_peak_density(self.traj[k].u)

    def __len__(self):
        return len(self.traj)

    def __getitem__(self, index):
        return self.traj[index]

    def spawn(self):
        """
        Create new ParticleTrajectory starting at the end of
        the current one
        """
        return ParticleTrajectory(copy.deepcopy(self.traj[-1].pa), resample=self.pf.res, t0=self.traj[-1].t, lp_hack=self.pf.lp_hack)

    def extract_signals(self):
        """
        Throw away the particle approxmation and return a list contaning just
        inputs, output and timestamps

        Returns:
         list of 'TrajectoryStep's with pa=None
        """
        signals = []
        for k in range(len(self.traj)):
            signals.append(TrajectoryStep(pa=None, u=self.traj[k].u, y=self.traj[k].y, t=self.traj[k].t))

        return signals

    def perform_smoothing(self, M, method="full", smoother_options=None):
        """

        Run a smoothing algorithms on the filtered estimate

        Args:
         - M (int): number of smoothed trajectories to create
         - method (string): smoothing algorithms to use
         - smoother_options (dict): options that are passed to the smoother

        Returns:
         SmoothTrajectory object containing the smoothed estimates
        """
        from smoother import SmoothTrajectory

        options = {}
        if (method == 'rs' or method == 'rsas'):
            # Calculate coefficients needed for rejection sampling in the backward smoothing
            coeffs = numpy.empty(len(self.traj), dtype=float)
            for k in range(len(self.traj)):
                coeffs[k] = self.pf.model.logp_xnext_max(particles=self.traj[k].pa.part,
                                                         u=self.traj[k].u, t=self.traj[k].t)
            options['maxpdf'] = coeffs
            if (method == 'rs'):
                # Default for max number of attempts before resoriting to evaluate all weights
                N = len(self.traj[0].pa.w)
                options['R'] = 0.1 * N
            if (method == 'rsas'):
                # Default settings for rsas
                options['x1'] = 1.0
                options['P1'] = 1.0
                options['sv'] = 1.0
                options['sw'] = 1.0
                options['ratio'] = 1.0

        if (method == 'mcmc' or method == 'mhips' or method == 'mhbp'):
            # Default value for number of iterations to run the sampler for
            options['R'] = 30
        if (smoother_options != None):
            options.update(smoother_options)

        return SmoothTrajectory(self, M=M, method=method, options=options)


class ParticleApproximation(object):
    """
    Contains collection of particles approximating a pdf

    Use either seed and num or particles (and optionally weights,
    if not uniform)

    Args:
     - particles (array-like): collection of particles
     - weights (array-like): weight for each particle
     - seed (array-like): value to initialize all particles with
     - num (int): number of particles

    """
    def __init__(self, particles=None, logw=None, seed=None, num=None):
        if (particles != None):
            self.part = numpy.asarray(particles)
            num = len(particles)
        else:
            self.part = numpy.empty(num, type(seed))
            for k in range(num):
                self.part[k] = copy.deepcopy(seed)

        if (logw != None):
            self.w = numpy.copy(logw)
        else:
            self.w = -math.log(num) * numpy.ones(num)

        self.num = num

        # Used to keep track of the offest on all weights, this is continually updated
        # when the weights are rescaled to avoid going to -Inf
        self.w_offset = 0.0

    def __len__(self):
        return len(self.part)

    def calc_Neff(self):
        """
        Calculate number of effective particles, common metric used to determine
        when to resample

        Returns:
         (float) number of effective particles
        """
        tmp = numpy.exp(self.w - numpy.max(self.w))
        tmp /= numpy.sum(tmp)
        return 1.0 / numpy.sum(numpy.square(tmp))

    def resample(self, model, N=None):
        """
        Resample approximation so all particles have the same weight

        Args:
         - model: object containing problem specific methods
         - N: new number of particles is N. If 'None' out the number of
           particles remains the same
        """

        if (N == None):
            N = self.num

        # Alwyays keep the largest weight at 0 in logaritmic representation
        tmp = self.w - numpy.max(self.w)
        new_ind = sample(numpy.exp(tmp), N)
        new_part = model.copy_ind(self.part, new_ind)

        self.w = numpy.log(numpy.ones(N, dtype=numpy.float) / N)
        self.part = new_part
        self.num = N
        self.w_offset = 0.0
        return new_ind

    def sample(self):
        """
        Draw one particle at random with probability corresponding to its weight

        Returns:
         (array-like) sampled particle"""
        return self.part[sample(numpy.exp(self.w), 1)[0]]

    def find_best_particles(self, n=1):
        """
        Return particles with largest weights

        Args:
         - n (int): Number of particles to return

        Returns:
         - (array-like) with len=n, representing the n most likely estimates """
        indices = numpy.argsort(self.w)
        return indices[range(n)]
