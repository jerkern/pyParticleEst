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


    def create_initial_estimate(self, N):
        return self.model.create_initial_estimate(N)

    def forward(self, traj, y):
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
        pa = ParticleApproximation(traj[-1].pa.part, traj[-1].pa.w)

        resampled = False
        if (self.res > 0 and pa.calc_Neff() < self.res * pa.num):
            # Store the ancestor of each resampled particle
            ancestors = pa.resample(self.model, pa.num)
            resampled = True
        else:
            ancestors = numpy.arange(pa.num, dtype=int)


        pa = self.update(traj, ancestors, pa, inplace=True)
        if (y != None):
            pa = self.measure(traj=traj, ancestors=ancestors, pa=pa, y=y, t=traj[-1].t + 1)
        return (pa, resampled, ancestors)

    def update(self, traj, ancestors, pa, inplace=True):
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

        # Prepare the particle for the update, eg. for
        # mixed linear/non-linear calculate the variables that
        # depend on the current state
#        for k in range(pa.num):
#            pa.part[k].prep_update(u)

        if (not inplace):
            pa = ParticleApproximation(self.model.copy_ind(traj[-1].pa.part,
                                                           ancestors),
                                       traj[-1].pa.w[ancestors])

        v = self.model.sample_process_noise(particles=pa.part, u=traj[-1].u,
                                            t=traj[-1].t)
        self.model.update_full(particles=pa.part, traj=traj,
                               ancestors=ancestors, noise=v)
        return pa


    def measure(self, traj, ancestors, pa, y, t, inplace=True):
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

        new_weights = self.model.measure_full(traj=traj, ancestors=ancestors,
                                              particles=pa.part, y=y, t=t)

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

class CPF(ParticleFilter):
    """
    Particle Filter class, creates filter estimates by calling appropriate
    methods in the supplied particle objects and handles resampling when
    a specified threshold is reach.

    Args:
     - model (ParticleFiltering): object describing the model to be used
     - res (float): 0 .. 1 , the ratio of effective number of particles that
       triggers resampling. 0 disables resampling
    """

    def __init__(self, model, cond_traj):

        self.ctraj = cond_traj
        self.model = model
        self.cur_ind = 0

    def create_initial_estimate(self, N):
        part = self.model.create_initial_estimate(N)
        part[-1] = self.ctraj[0]
        return part

    def forward(self, traj, y):
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
        N = len(traj[-1].pa.part)
        ancestors = numpy.empty((N,), dtype=int)
        tmp = numpy.exp(traj[-1].pa.w)
        tmp /= numpy.sum(tmp)
        ancestors[:-1] = sample(tmp, N - 1)

        ancestors[-1] = N - 1 #condind

        pa = ParticleApproximation(self.model.copy_ind(traj[-1].pa.part,
                                                       ancestors))


        resampled = True

        pa = self.update(traj, ancestors, pa, inplace=True)
        pa.part[-1] = self.ctraj[self.cur_ind + 1]

        if (y != None):
            pa = self.measure(traj=traj, ancestors=ancestors, pa=pa, y=y, t=traj[-1].t + 1)
        self.cur_ind += 1
        return (pa, resampled, ancestors)

class CPFAS(CPF):
    """
    Particle Filter class, creates filter estimates by calling appropriate
    methods in the supplied particle objects and handles resampling when
    a specified threshold is reach.

    Args:
     - model (ParticleFiltering): object describing the model to be used
     - res (float): 0 .. 1 , the ratio of effective number of particles that
       triggers resampling. 0 disables resampling
    """

    def forward(self, traj, y):
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
        N = len(traj[-1].pa.part)
        ancestors = numpy.empty((N,), dtype=int)
        tmp = numpy.exp(traj[-1].pa.w)
        tmp /= numpy.sum(tmp)
        ancestors[:-1] = sample(tmp, N - 1)

        # TODO:This is ugly and slow, ut, yt, tt must be stored more efficiently
        (ut, yt, tt) = extract_signals(traj)

        #select ancestor for conditional trajectory
        pind = numpy.asarray(range(N), dtype=numpy.int)
        find = numpy.zeros((N,), dtype=numpy.int)
        wtrans = self.model.logp_xnext_full(traj, pind, self.ctraj[self.cur_ind + 1][numpy.newaxis],
                                            find=find, ut=(None,), yt=yt[-1:], tt=tt[-1:])
        wanc = wtrans + traj[-1].pa.w[pind]
        wanc -= numpy.max(wanc)
        tmp = numpy.exp(wanc)
        tmp /= numpy.sum(tmp)
        condind = sample(tmp, 1)
        ancestors[-1] = condind

        pa = ParticleApproximation(self.model.copy_ind(traj[-1].pa.part,
                                                       ancestors))


        resampled = True

        pa = self.update(traj, ancestors, pa, inplace=True)
        pa.part[-1] = self.ctraj[self.cur_ind + 1]

        if (y != None):
            pa = self.measure(traj=traj, ancestors=ancestors, pa=pa, y=y, t=traj[-1].t + 1)
        self.cur_ind += 1
        return (pa, resampled, ancestors)



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

class CPFYAS(object):
    def __init__(self, model, N, cond_traj):
        self.ctraj = numpy.copy(cond_traj)
        self.model = model
        self.N = N
        self.cur_ind = 0


    def create_initial_estimate(self, N):
        self.N = N
        part = self.model.create_initial_estimate(N)
        part[-1] = self.ctraj[0]

    def forward(self, traj, y):
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
        pa = ParticleApproximation(self.model.copy_ind(traj[-1].pa.part),
                                   traj[-1].pa.w)

        ancestors = numpy.empty((self.N), dtype=int)
        ancestors[:-1] = sample(numpy.exp(pa.w), self.N - 1)
        resampled = True

        # TODO:This is ugly and slow, ut, yt, tt must be stored more efficiently
        (ut, yt, tt) = extract_signals(traj)

        #select ancestor for conditional trajectory
        pind = numpy.asarray(range(self.N))
        find = numpy.zeros((self.N,), dtype=numpy.int)
        wac = self.model.logp_xnext_full(traj, pind, self.ctraj[self.cur_ind][numpy.newaxis],
                                         find=find, ut=ut, yt=yt, tt=tt)
        wac += pa.w
        wac -= numpy.max(wac)
        ancestors[-1] = sample(numpy.exp(wac), 1)

        partn = self.model.propose_from_y(self.N, y=y, t=traj[-1].t + 1)
        partn[-1] = self.ctraj[self.cur_ind, 0]
        self.cur_ind += 1

        find = numpy.asarray(range(self.N))

        future_trajs = self.model.sample_smooth(partn, future_trajs=None,
                                                ut=ut, yt=yt, tt=tt)
        wn = self.model.logp_xnext_full(traj, ancestors, future_trajs[numpy.newaxis],
                                        find=find, ut=ut, yt=yt, tt=tt)
        pa.part = partn
        # Try to keep weights from going to -Inf
        m = numpy.max(wn)
        pa.w_offset += m
        wn -= m

        pa.w = pa.w + wn

        # Keep the weights from going to -Inf
        m = numpy.max(pa.w)
        pa.w_offset += m
        pa.w -= m

        return (pa, resampled, ancestors)

    def measure(self, traj, ancestors, pa, y, t, inplace=True):
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

        assert(not inplace)
        part = self.model.propose_from_y(self.N, y=y, t=t)
        part[-1] = self.ctraj[self.cur_ind]
        self.cur_ind += 1
        pa = ParticleApproximation(part)
        return pa

class FFPropY(object):
    """
    Particle Filter class, creates filter estimates by calling appropriate
    methods in the supplied particle objects and handles resampling when
    a specified threshold is reach.

    Args:
     - model (ParticleFiltering): object describing the model to be used
     - res (float): 0 .. 1 , the ratio of effective number of particles that
       triggers resampling. 0 disables resampling
    """

    def __init__(self, model, N, res=0):

        self.res = res
        self.model = model
        self.N = N

    def create_initial_estimate(self, N):
        self.N = N
        return self.model.create_initial_estimate(N)

    def forward(self, traj, y):
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
        pa = ParticleApproximation(self.model.copy_ind(traj[-1].pa.part),
                                   traj[-1].pa.w)

        resampled = False
        if (self.res > 0 and pa.calc_Neff() < self.res * pa.num):
            # Store the ancestor of each resampled particle
            ancestors = pa.resample(self.model, pa.num)
            resampled = True
        else:
            ancestors = numpy.arange(pa.num, dtype=int)

        partn = self.model.propose_from_y(len(pa.part), y=y, t=traj[-1].t + 1)

        find = numpy.asarray(range(self.N))

        # TODO:This is ugly and slow, ut, yt, tt must be stored more efficiently
        (ut, yt, tt) = extract_signals(traj)
        future_trajs = self.model.sample_smooth(partn, future_trajs=None,
                                                ut=ut, yt=yt, tt=tt)
        wn = self.model.logp_xnext_full(traj, ancestors, future_trajs[numpy.newaxis],
                                        find=find, ut=ut, yt=yt, tt=tt)
        pa.part = partn
        # Try to keep weights from going to -Inf
        m = numpy.max(wn)
        pa.w_offset += m
        wn -= m

        pa.w = pa.w + wn

        # Keep the weights from going to -Inf
        m = numpy.max(pa.w)
        pa.w_offset += m
        pa.w -= m

        return (pa, resampled, ancestors)

    def measure(self, traj, ancestors, pa, y, t, inplace=True):
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

        assert(not inplace)
        part = self.model.propose_from_y(self.N, y=y, t=t)
        pa = ParticleApproximation(part)
        return pa


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


def extract_signals(traj):
    """
    Create seperate arrays containing particle estimates, inputs, outputs
    and timestamps

    Returns:
     (ut, yt, tt)
    """
    T = len(traj)

    ut = numpy.empty((T,), dtype=numpy.ndarray)
    yt = numpy.empty((T,), dtype=numpy.ndarray)
    tt = numpy.empty((T,))

    for k in xrange(T):
        ut[k] = traj[k].u
        yt[k] = traj[k].y
        tt[k] = traj[k].t

    return (ut, yt, tt)

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

    def __init__(self, model, N, resample=2.0 / 3.0, t0=0, filter='PF', filter_options={}):
        self.t0 = t0
        sampleInitial = False
        self.using_pfy = False
        self.N = N
        if (filter.lower() == 'pf'):
            self.pf = ParticleFilter(model=model, res=resample)
            sampleInitial = True
        elif (filter.lower() == 'apf'):
            self.pf = AuxiliaryParticleFilter(model=model, res=resample)
            sampleInitial = True
        elif (filter.lower() == 'pfy'):
            self.pf = FFPropY(model=model, N=N, res=resample)
            self.using_pfy = True
        elif (filter.lower() == 'cpfyas'):
            self.pf = CPFYAS(model=model, N=N, cond_traj=filter_options['cond_traj'])
            self.using_pfy = True
        elif (filter.lower() == 'cpfas'):
            self.pf = CPFAS(model=model, cond_traj=filter_options['cond_traj'])
        elif (filter.lower() == 'cpf'):
            self.pf = CPF(model=model, cond_traj=filter_options['cond_traj'])
        else:
            raise ValueError('Bad filter type')

        self.traj = []

        return

    def forward(self, u, y):
        """
        Append new time step to trajectory

        Args:f
         - u (array-like): Input to go from x_t -> x_{t+1}
         - y (array-like): Measurement of x_{t+1}

        Returns:
         (bool) True if the particle approximation was resampled
        """
        if (len(self.traj) == 0):
            particles = self.pf.create_initial_estimate(self.N)
            pa = ParticleApproximation(particles=particles)
            self.traj.append(TrajectoryStep(pa, t=self.t0,
                                            ancestors=numpy.arange(self.N)))
        self.traj[-1].u = u
        (pa_nxt, resampled, ancestors) = self.pf.forward(self.traj, y)
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

        if (self.using_pfy):
            if (len(self.traj) > 0):
                t = self.traj[-1].t + 1
            else:
                t = 0
            self.traj.append(TrajectoryStep(None, t=t, y=y, ancestors=None))
            pa = self.pf.measure(traj=self.traj,
                                 ancestors=self.traj[-1].ancestors,
                                 pa=None,
                                 y=self.traj[-1].y,
                                 t=self.traj[-1].t,
                                 inplace=False)
            self.traj[-1].pa = pa
            self.traj[-1].ancestors = numpy.arange(pa.num, dtype=int)
        else:
            if (len(self.traj) == 0):
                particles = self.pf.create_initial_estimate(self.N)
                pa = ParticleApproximation(particles=particles)
                self.traj.append(TrajectoryStep(pa, t=self.t0,
                                                ancestors=numpy.arange(self.N)))
            self.traj[-1].y = y
            self.pf.measure(traj=self.traj, ancestors=self.traj[-1].ancestors,
                            pa=self.traj[-1].pa, y=self.traj[-1].y,
                            t=self.traj[-1].t, inplace=True)

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
                coeffs[k] = self.pf.model.logp_xnext_max_full(ptraj=self.traj[:k + 1],
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
            self.part = numpy.copy(numpy.asarray(particles))
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
