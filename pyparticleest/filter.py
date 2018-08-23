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
        """
        Create initial particle estimate

        Args:
         - N (int): Number of particles

        Returns:
        (array-like): Sampled particles

        """
        return self.model.create_initial_estimate(N)

    def forward(self, traj, yvec, uvec, tvec, cur_ind):
        """
        Forward the estimate stored in traj from t to t+1 using the motion model
        with input u=uvec[cur_ind] at time t=tvec[cur_ind] and measurement
        y=yvec[cur_ind+1] at t=tvec[cur_ind+1]

        Args:
         - traj (array-like): approximation for time t
         - uvec (array-like): input signals
         - yvec (array-like): measurements
         - tvec (array-like): time stamps
         - cur_ind (int): index of current time-step in (uvec, uvec, tvec)

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


        pa = self.update(traj=traj, ancestors=ancestors,
                         uvec=uvec, yvec=yvec,
                         tvec=tvec, cur_ind=cur_ind,
                         pa=pa, inplace=True,)
        if (yvec is not None):
            pa = self.measure(traj=traj, ancestors=ancestors, pa=pa,
                              #There is no 'u' for last step yet
                              uvec=uvec, yvec=yvec, tvec=tvec, cur_ind=cur_ind + 1)
        return (pa, resampled, ancestors)

    def update(self, traj, ancestors, uvec, yvec, tvec, cur_ind, pa, inplace=True):
        """
        Update particle approximation of x_t to x_{t+1} using u as input.

        Args:
         - traj (array-like): approximation for time t
         - ancestors (array-like): indices which to propagate estimates from
         - uvec (array-like): input signals
         - yvec (array-like): measurements
         - tvec (array-like): time stamps
         - cur_ind (int): index of current time-step in (uvec, uvec, tvec)
         - inplace (bool): if True the particles are updated then returned,
           otherwise a new ParticleApproximation is first created
           leaving the original one intact

        Returns:
            ParticleApproximation for time t+1
        """

        if (not inplace):
            pa = ParticleApproximation(self.model.copy_ind(traj[-1].pa.part,
                                                           ancestors),
                                       traj[-1].pa.w[ancestors])

        v = self.model.sample_process_noise_full(ptraj=traj, ancestors=ancestors,
                                                 ut=uvec[:cur_ind + 1],
                                                 tt=tvec[:cur_ind + 1])
        self.model.update_full(particles=pa.part, traj=traj,
                               uvec=uvec[:cur_ind + 1], yvec=yvec[:cur_ind + 1],
                               tvec=tvec[:cur_ind + 1],
                               ancestors=ancestors, noise=v)
        return pa


    def measure(self, traj, ancestors, pa, uvec, yvec, tvec, cur_ind, inplace=True):
        """
        Evaluate and update particle approximation using new measurement y

        Args:
         - pa (ParticleApproximation): approximation for time t
         - traj (array-like): approximation for time 0:t-1
         - ancestors (array-like): indices which to propagate estimates from
         - uvec (array-like): input signals
         - yvec (array-like): measurements
         - tvec (array-like): time stamps
         - cur_ind (int): index of current time-step in (uvec, uvec, tvec)
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
                                              particles=pa.part, uvec=uvec,
                                              yvec=yvec[:cur_ind + 1],
                                              tvec=tvec[:cur_ind + 1])

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

class SIR(object):
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
        """
        Create initial particle estimate

        Args:
         - N (int): Number of particles

        Returns:
        (array-like): Sampled particles

        """
        return self.model.create_initial_estimate(N)

    def forward(self, traj, yvec, uvec, tvec, cur_ind):
        """
        Forward the estimate stored in pa from t to t+1 using the motion model
        with input u=uvec[cur_ind] at time t=tvec[cur_ind] and measurement
        y=yvec[cur_ind+1] at t=tvec[cur_ind+1]

        Args:
         - traj (array-like): approximation for time t
         - uvec (array-like): input signals
         - yvec (array-like): measurements
         - tvec (array-like): time stamps
         - cur_ind (int): index of current time-step in (uvec, uvec, tvec)

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

        pnext = self.model.qsample(particles=pa.part, u=uvec[cur_ind],
                                   y=yvec[cur_ind + 1], t=tvec[cur_ind])


        qw = self.model.logp_q(particles=pa.part, next_part=pnext,
                               u=uvec[cur_ind], y=yvec[cur_ind + 1],
                               t=tvec[cur_ind])
        nw = self.model.logp_xnext(particles=pa.part, next_part=pnext,
                                   u=uvec[cur_ind], t=tvec[cur_ind])

        yw = self.model.measure(particles=pnext, y=yvec[cur_ind + 1],
                                     t=tvec[cur_ind + 1])

        new_weights = yw + nw - qw

        # Try to keep weights from going to -Inf
        m = numpy.max(new_weights)
        pa.w_offset += m
        new_weights -= m

        pa.w = pa.w + new_weights

        # Keep the weights from going to -Inf
        m = numpy.max(pa.w)
        pa.w_offset += m
        pa.w -= m
        pa.part = pnext

        return (pa, resampled, ancestors)

    def update(self, traj, ancestors, uvec, yvec, tvec, cur_ind, pa, inplace=True):
        """
        Update particle approximation of x_t to x_{t+1} using u as input.

        Args:
         - traj (array-like): approximation for time t
         - ancestors (array-like): indices which to propagate estimates from
         - uvec (array-like): input signals
         - yvec (array-like): measurements
         - tvec (array-like): time stamps
         - cur_ind (int): index of current time-step in (uvec, uvec, tvec)
         - inplace (bool): if True the particles are updated then returned,
           otherwise a new ParticleApproximation is first created
           leaving the original one intact

        Returns:
            ParticleApproximation for time t+1
        """


        return pa


    def measure(self, traj, ancestors, pa, uvec, yvec, tvec, cur_ind, inplace=True):
        """
        Evaluate and update particle approximation using new measurement y

        Args:
         - pa (ParticleApproximation): approximation for time t
         - traj (array-like): approximation for time 0:t-1
         - ancestors (array-like): indices which to propagate estimates from
         - uvec (array-like): input signals
         - yvec (array-like): measurements
         - tvec (array-like): time stamps
         - cur_ind (int): index of current time-step in (uvec, uvec, tvec)
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
                                              particles=pa.part, uvec=uvec,
                                              yvec=yvec[:cur_ind + 1],
                                              tvec=tvec[:cur_ind + 1])

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

class CSIRAS(SIR):
    def __init__(self, model, cond_traj):

        self.ctraj = cond_traj
        self.model = model

    def forward(self, traj, yvec, uvec, tvec, cur_ind):
        """
        Forward the estimate stored in pa from t to t+1 using the motion model
        with input u at time time and measurement y at time t+1

        Args:
         - traj (array-like): approximation for time t
         - uvec (array-like): input signals
         - yvec (array-like): measurements
         - tvec (array-like): time stamps
         - cur_ind (int): index of current time-step in (uvec, uvec, tvec)

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

        #select ancestor for conditional trajectory
        pind = numpy.arange(N, dtype=numpy.int)
        find = numpy.zeros((N,), dtype=numpy.int)

        wtrans = self.model.logp_xnext(particles=traj[cur_ind].pa.part,
                                       next_part=self.ctraj[cur_ind + 1].pa.part[find],
                                       u=uvec[cur_ind], t=tvec[cur_ind])

        wanc = wtrans + traj[-1].pa.w[pind]
        wanc -= numpy.max(wanc)
        tmp = numpy.exp(wanc)
        tmp /= numpy.sum(tmp)
        condind = sample(tmp, 1)
        ancestors[-1] = condind

        pa = ParticleApproximation(self.model.copy_ind(traj[-1].pa.part,
                                                       ancestors))

        pnext = self.model.qsample(particles=pa.part, u=uvec[cur_ind],
                                   y=yvec[cur_ind + 1], t=tvec[cur_ind])

        pnext[-1] = self.ctraj[cur_ind + 1].pa.part[0]

        qw = self.model.logp_q(particles=pa.part, next_part=pnext,
                               u=uvec[cur_ind], y=yvec[cur_ind + 1],
                               t=tvec[cur_ind])
        nw = self.model.logp_xnext(particles=pa.part, next_part=pnext,
                                   u=uvec[cur_ind], t=tvec[cur_ind])

        yw = self.model.measure(particles=pnext, y=yvec[cur_ind + 1],
                                     t=tvec[cur_ind + 1])


        pa.w = yw + nw - qw

        pa.part[:] = pnext
        resampled = True

        return (pa, resampled, ancestors)

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
        """
        Create initial particle estimate

        Args:
         - N (int): Number of particles

        Returns:
        (array-like): Sampled particles

        """
        self.N = N
        return self.model.create_initial_estimate(N)

    def forward(self, traj, yvec, uvec, tvec, cur_ind):
        """
        Forward the estimate stored in pa from t to t+1 using the motion model
        with input u at time time and measurement y at time t+1
        cur_ind is at time t

        Args:
         - traj (array-like): approximation for time t
         - uvec (array-like): input signals
         - yvec (array-like): measurements
         - tvec (array-like): time stamps
         - cur_ind (int): index of current time-step in (uvec, uvec, tvec)

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

        partn = self.model.propose_from_y(len(pa.part), y=yvec[-1], t=tvec[-1])
        find = numpy.arange(self.N, dtype=int)

        wn = self.model.logp_xnext_singlestep(part=traj[cur_ind].pa.part[ancestors],
                                              past_trajs=traj[:cur_ind],
                                              pind=traj[cur_ind].ancestors[ancestors],
                                              future_parts=partn,
                                              find=find,
                                              ut=uvec, yt=yvec, tt=tvec, cur_ind=cur_ind)
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

    def measure(self, traj, ancestors, pa, uvec, yvec, tvec, cur_ind, inplace=True):
        """
        Evaluate and update particle approximation using new measurement y

        Args:
         - pa (ParticleApproximation): approximation for time t
         - traj (array-like): approximation for time 0:t-1
         - ancestors (array-like): indices which to propagate estimates from
         - uvec (array-like): input signals
         - yvec (array-like): measurements
         - tvec (array-like): time stamps
         - cur_ind (int): index of current time-step in (uvec, uvec, tvec)
         - inplace (bool): if True the particles are updated then returned,
           otherwise a new ParticleApproximation is first created
           leaving the original one intact

        Returns:
            ParticleApproximation for time t
        """

        assert(not inplace)
        part = self.model.propose_from_y(self.N, y=yvec[cur_ind], t=tvec[cur_ind])
        pa = ParticleApproximation(part)
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

    def create_initial_estimate(self, N):
        """
        Create initial particle estimate

        Args:
         - N (int): Number of particles

        Returns:
        (array-like): Sampled particles

        """
        part = self.model.create_initial_estimate(N)
        part[-1] = self.ctraj[0].pa.part[0]
        return part

    def forward(self, traj, yvec, uvec, tvec, cur_ind):
        """
        Forward the estimate stored in pa from t to t+1 using the motion model
        with input u at time time and measurement y at time t+1

        Args:
         - traj (array-like): approximation for time t
         - uvec (array-like): input signals
         - yvec (array-like): measurements
         - tvec (array-like): time stamps
         - cur_ind (int): index of current time-step in (uvec, uvec, tvec)

        Returns (pa, resampled, ancestors)
         - pa (ParticleApproximation): approximation for time t+1
         - resampled (bool): were the particles resampled
         - ancestors (array-like): anecstral indices for particles at time t+1
        """
        N = len(traj[cur_ind].pa.part)
        ancestors = numpy.empty((N,), dtype=int)
        tmp = numpy.exp(traj[-1].pa.w)
        tmp /= numpy.sum(tmp)
        ancestors[:-1] = sample(tmp, N - 1)

        ancestors[-1] = N - 1 #condind

        pa = ParticleApproximation(self.model.copy_ind(traj[-1].pa.part,
                                                       ancestors))


        resampled = True

        pa = self.update(traj=traj, ancestors=ancestors, uvec=uvec, yvec=yvec,
                         tvec=tvec, cur_ind=cur_ind, pa=pa, inplace=True)
        pa.part[-1] = self.ctraj[cur_ind + 1].pa.part[0]

        if (yvec is not None and yvec[cur_ind + 1] is not None):
            pa = self.measure(traj=traj, ancestors=ancestors, pa=pa, uvec=uvec,
                              yvec=yvec, tvec=tvec, cur_ind=cur_ind + 1)

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

    def forward(self, traj, yvec, uvec, tvec, cur_ind):
        """
        Forward the estimate stored in pa from t to t+1 using the motion model
        with input u at time time and measurement y at time t+1

        Args:
         - traj (array-like): approximation for time t
         - uvec (array-like): input signals
         - yvec (array-like): measurements
         - tvec (array-like): time stamps
         - cur_ind (int): index of current time-step in (uvec, uvec, tvec

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

        #select ancestor for conditional trajectory
        pind = numpy.arange(N, dtype=numpy.int)
        find = numpy.zeros((N,), dtype=numpy.int)

        wtrans = self.model.logp_xnext_singlestep(part=traj[cur_ind].pa.part[pind],
                                                  past_trajs=traj[:cur_ind],
                                                  pind=traj[cur_ind].ancestors[pind],
                                                  # Single future timestep
                                                  future_parts=self.ctraj[cur_ind + 1].pa.part,
                                                  find=find,
                                                  ut=uvec, yt=yvec, tt=tvec,
                                                  cur_ind=cur_ind)
        wanc = wtrans + traj[-1].pa.w[pind]
        wanc -= numpy.max(wanc)
        tmp = numpy.exp(wanc)
        tmp /= numpy.sum(tmp)
        condind = sample(tmp, 1)
        ancestors[-1] = condind

        pa = ParticleApproximation(self.model.copy_ind(traj[-1].pa.part,
                                                       ancestors))


        resampled = True

        pa = self.update(traj=traj, ancestors=ancestors, uvec=uvec, yvec=yvec,
                         tvec=tvec, cur_ind=cur_ind, pa=pa, inplace=True)
        pa.part[-1] = self.ctraj[cur_ind + 1].pa.part[0]

        if (yvec is not None and yvec[cur_ind + 1] is not None):
            pa = self.measure(traj=traj, ancestors=ancestors, pa=pa, uvec=uvec,
                              yvec=yvec, tvec=tvec, cur_ind=cur_ind + 1)

        return (pa, resampled, ancestors)



class AuxiliaryParticleFilter(ParticleFilter):
    """ Auxiliary Particle Filer class, creates filter estimates by calling appropriate
        methods in the supplied particle objects and handles resampling when
        a specified threshold is reach """

    def forward(self, traj, yvec, uvec, tvec, cur_ind):
        """
        Use the first stage weights to try to predict which particles will be in
        regions of high likely hood at time t+1, use this information to resample
        the particles before propagating them forward in time

        Args:
         - traj (array-like): approximation for time t
         - uvec (array-like): input signals
         - yvec (array-like): measurements
         - tvec (array-like): time stamps
         - cur_ind (int): index of current time-step in (uvec, uvec, tvec

        Returns (pa, resampled, ancestors)
         - pa (ParticleApproximation): approximation for time t+1
         - resampled (bool): were the particles resampled
         - ancestors (array-like): anecstral indices for particles at time t+1
        """

        pa = ParticleApproximation(traj[-1].pa.part, traj[-1].pa.w)

        resampled = False

        if (yvec is not None and yvec[cur_ind + 1] is not None):
            # TODO Generalize to non-Markovian
            l1w = self.model.eval_1st_stage_weights(pa.part, uvec[cur_ind],
                                                    yvec[cur_ind + 1],
                                                    tvec[cur_ind])
            pa.w += l1w
            pa.w -= numpy.max(pa.w)

        if (self.res and pa.calc_Neff() < self.res * pa.num):
            ancestors = pa.resample(self.model, pa.num)
            resampled = True
            l1w = l1w[ancestors]
        else:
            ancestors = numpy.arange(pa.num, dtype=int)

        pa = self.update(traj=traj, ancestors=ancestors,
                         uvec=uvec, yvec=yvec,
                         tvec=tvec, cur_ind=cur_ind,
                         pa=pa, inplace=True,)

        if (yvec is not None):
            pa = self.measure(traj=traj, ancestors=ancestors, pa=pa,
                              #There is no 'u' for last step yet
                              uvec=uvec, yvec=yvec, tvec=tvec, cur_ind=cur_ind + 1)
            pa.w -= l1w
            pa.w -= numpy.max(pa.w)

        return (pa, resampled, ancestors)



class CPFYAS(CPFAS):
    def __init__(self, model, N, cond_traj):
        self.ctraj = numpy.copy(cond_traj)
        self.model = model
        self.N = N

    def forward(self, traj, yvec, uvec, tvec, cur_ind):
        """
        Forward the estimate stored in pa from t to t+1 using the motion model
        with input u at time time and measurement y at time t+1

        Args:
         - traj (array-like): approximation for time t
         - uvec (array-like): input signals
         - yvec (array-like): measurements
         - tvec (array-like): time stamps
         - cur_ind (int): index of current time-step in (uvec, uvec, tvec

        Returns (pa, resampled, ancestors)
         - pa (ParticleApproximation): approximation for time t+1
         - resampled (bool): were the particles resampled
         - ancestors (array-like): ancestral indices for particles at time t+1
        """

        ancestors = numpy.empty((self.N,), dtype=int)
        tmp = numpy.exp(traj[cur_ind].pa.w)
        tmp /= numpy.sum(tmp)
        ancestors[:-1] = sample(tmp, self.N - 1)

        #select ancestor for conditional trajectory
        pind = numpy.arange(self.N, dtype=numpy.int)
        find = numpy.zeros((self.N,), dtype=numpy.int)

        wtrans = self.model.logp_xnext_singlestep(part=traj[cur_ind].pa.part[pind],
                                                  past_trajs=traj[:cur_ind],
                                                  pind=traj[cur_ind].ancestors[pind],
                                                  # Single future timestep
                                                  future_parts=self.ctraj[cur_ind + 1].pa.part,
                                                  find=find,
                                                  ut=uvec, yt=yvec, tt=tvec,
                                                  cur_ind=cur_ind)

        wanc = wtrans + traj[cur_ind].pa.w[pind]
        wanc -= numpy.max(wanc)
        tmp = numpy.exp(wanc)
        tmp /= numpy.sum(tmp)
        condind = sample(tmp, 1)
        ancestors[-1] = condind
        resampled = True

        partn = self.model.propose_from_y(self.N, y=yvec[cur_ind + 1], t=tvec[cur_ind + 1])
        partn[-1] = self.ctraj[cur_ind + 1].pa.part

        find = numpy.arange(self.N)

        wn = self.model.logp_xnext_singlestep(part=traj[-1].pa.part[ancestors],
                                              past_trajs=traj[:cur_ind],
                                              pind=traj[cur_ind].ancestors[ancestors],
                                              future_parts=partn,
                                              find=find,
                                              ut=uvec, yt=yvec, tt=tvec,
                                              cur_ind=cur_ind)
        m = numpy.max(wn)
        wn -= m
        pa = ParticleApproximation(partn, wn)
        pa.w_offset += m
        return (pa, resampled, ancestors)

    def measure(self, traj, ancestors, pa, uvec, yvec, tvec, cur_ind, inplace=True):
        """
        Evaluate and update particle approximation using new measurement y

        Args:
         - pa (ParticleApproximation): approximation for time t
         - traj (array-like): approximation for time 0:t-1
         - ancestors (array-like): indices which to propagate estimates from
         - uvec (array-like): input signals
         - yvec (array-like): measurements
         - tvec (array-like): time stamps
         - cur_ind (int): index of current time-step in (uvec, uvec, tvec)
         - inplace (bool): if True the particles are updated then returned,
           otherwise a new ParticleApproximation is first created
           leaving the original one intact

        Returns:
            ParticleApproximation for time t
        """

        assert(not inplace)
        part = self.model.propose_from_y(self.N, y=yvec[cur_ind], t=tvec[cur_ind])
        part[-1] = self.ctraj[cur_ind].pa.part
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
    def __init__(self, pa, ancestors=None):
        self.pa = pa
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
     - filter_options (dictionary): options passed to the filter
     - T (int): Length of dataset (for non-online computations), pre-allocates
       space for input/output/time vectors
     - utype (array): the datatype of the input signals
     - ytype (array): the datatype of the measurements
    """

    def __init__(self, model, N, resample=2.0 / 3.0, t0=0,
                 filter='PF', filter_options={}, T=None,
                 utype=numpy.ndarray, ytype=numpy.ndarray):

        self.using_pfy = False
        self.N = N
        if (T is not None):
            assert(utype is not None)
            assert(ytype is not None)
            self.uvec = numpy.empty(T, dtype=utype)
            self.yvec = numpy.empty(T, dtype=ytype)
            self.tvec = numpy.arange(T)
            self.T = T
        else:
            self.uvec = numpy.empty(1, dtype=utype)
            self.yvec = numpy.empty(1, dtype=ytype)
            self.tvec = numpy.empty(1, dtype=numpy.float)
            self.T = 0
        #TODO, this isn't correctly used in the code, assumed = 0
        assert(t0 == 0)
        self.tvec[0] = t0
        self.ind = -1
        if (filter.lower() == 'pf'):
            self.pf = ParticleFilter(model=model, res=resample)
        elif (filter.lower() == 'apf'):
            self.pf = AuxiliaryParticleFilter(model=model, res=resample)
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
        elif (filter.lower() == 'sir'):
            self.pf = SIR(model=model, res=resample)
        elif (filter.lower() == 'csiras'):
            self.pf = CSIRAS(model=model, cond_traj=filter_options['cond_traj'])
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
            self.ind = 0
            particles = self.pf.create_initial_estimate(self.N)
            pa = ParticleApproximation(particles=particles)
            self.traj.append(TrajectoryStep(pa, ancestors=numpy.arange(self.N)))

        if (self.ind + 1 >= self.T):
            ushape = numpy.asarray(self.uvec.shape)
            ushape[0] = self.ind + 1
            self.uvec.resize(ushape)
            yshape = numpy.asarray(self.yvec.shape)
            yshape[0] = self.ind + 2
            self.yvec.resize(yshape)
            tshape = numpy.asarray(self.tvec.shape)
            tshape[0] = self.ind + 2
            self.tvec.resize(tshape)
            self.T = self.ind + 2

        ind = self.ind
        self.uvec[ind] = u
        self.yvec[ind + 1] = y
        self.tvec[ind + 1] = ind + 1
        self.ind += 1

        (pa_nxt, resampled, ancestors) = self.pf.forward(traj=self.traj,
                                                         yvec=self.yvec,
                                                         uvec=self.uvec,
                                                         tvec=self.tvec,
                                                         cur_ind=ind)
        self.traj.append(TrajectoryStep(pa_nxt, ancestors=ancestors))

        return resampled

    def measure(self, y):
        """
        Update estimate using new measurement

        Args:
         - y (array-like): Measurement at current time index

        Returns:
         None
        """

        if (self.ind + 1 >= self.T):
            ushape = numpy.asarray(self.uvec.shape)
            ushape[0] = self.ind + 2
            self.uvec.resize(ushape)
            yshape = numpy.asarray(self.yvec.shape)
            yshape[0] = self.ind + 2
            self.yvec.resize(yshape)
            tshape = numpy.asarray(self.tvec.shape)
            tshape[0] = self.ind + 2
            self.tvec.resize(tshape)
            self.T = self.ind + 2

        if (self.using_pfy):
            self.ind += 1
            self.yvec[self.ind] = y
            self.tvec[self.ind] = self.ind

            ancestors = numpy.arange(self.N, dtype=int)
            pa = self.pf.measure(traj=self.traj,
                                 ancestors=ancestors,
                                 pa=None,
                                 uvec=self.uvec,
                                 yvec=self.yvec,
                                 tvec=self.tvec,
                                 cur_ind=self.ind,
                                 inplace=False)
            self.traj.append(TrajectoryStep(pa, ancestors=ancestors))
        else:
            if (len(self.traj) == 0):
                self.ind = 0
                particles = self.pf.create_initial_estimate(self.N)
                pa = ParticleApproximation(particles=particles)
                ancestors = numpy.arange(self.N, dtype=int)
                self.traj.append(TrajectoryStep(pa, ancestors=ancestors))

            self.yvec[self.ind] = y
            self.tvec[self.ind] = self.ind

            self.pf.measure(traj=self.traj, ancestors=self.traj[-1].ancestors,
                            pa=self.traj[-1].pa, uvec=self.uvec,
                            yvec=self.yvec, tvec=self.tvec,
                            cur_ind=self.ind, inplace=True)

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
        from .smoother import SmoothTrajectory

        options = {}
        if (method == 'rs' or method == 'rsas'):
            # Calculate coefficients needed for rejection sampling in the backward smoothing
            coeffs = numpy.empty(len(self.traj), dtype=float)
            for k in range(len(self.traj) - 1):
                coeffs[k] = self.pf.model.logp_xnext_max_full(part=self.traj[k].pa.part,
                                                              past_trajs=self.traj[:k],
                                                              pind=self.traj[k].ancestors,
                                                              uvec=self.uvec,
                                                              yvec=self.yvec,
                                                              tvec=self.tvec,
                                                              cur_ind=k)
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
        if (smoother_options is not None):
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
        if (particles is not None):
            self.part = numpy.copy(numpy.asarray(particles))
            num = len(particles)
        else:
            self.part = numpy.empty(num, type(seed))
            for k in range(num):
                self.part[k] = copy.deepcopy(seed)

        if (logw is not None):
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

        if (N is None):
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
