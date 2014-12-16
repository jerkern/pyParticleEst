""" Collection of smoothing algorithms and support classes for using them

@author: Jerker Nordh
"""

import numpy
import filter as pf

def bsi_full(ptraj, model, future_trajs, ut, yt, tt, cur_ind):
    """
    Perform backward simulation by drawing particles from
    the categorical distribution with weights given by
    \omega_{t|T}^i = \omega_{t|t}^i*p(x_{t+1}|x^i)

    Args:
    - pa (ParticleApproximation): particles approximation from which to sample
    - model (FFBSi): model defining probability density function
    - future_trajs (array-like): trajectory estimate of {t+1:T}
    - ut (array-like): inputs signal for {t:T}
    - yt (array-like): measurements for {t:T}
    - tt (array-like): time stamps for {t:T}
    """

    M = future_trajs.shape[1]
    N = len(ptraj[-1].pa.part)
    res = numpy.empty(M, dtype=int)
    pind = numpy.asarray(range(N))
    for j in xrange(M):
        find = j * numpy.ones((N,), dtype=int)
        p_next = model.logp_xnext_full(ptraj, pind,
                                       future_trajs, find,
                                       ut=ut, yt=yt, tt=tt, ind=cur_ind)

        w = ptraj[-1].pa.w + p_next
        w = w - numpy.max(w)
        w_norm = numpy.exp(w)
        w_norm /= numpy.sum(w_norm)
        res[j] = pf.sample(w_norm, 1)
    return res


def bsi_rs(ptraj, model, future_trajs, ut, yt, tt, cur_ind, maxpdf, max_iter):
    """
    Perform backward simulation by using rejection sampling to draw particles
    from the categorical distribution with weights given by
    \omega_{t|T}^i = \omega_{t|t}^i*p(x_{t+1}|x^i)

    Args:
     - pa (ParticleApproximation): particles approximation from which to sample
     - model (FFBSi): model defining probability density function
     - future_trajs (array-like): trajectory estimate of {t+1:T}
     - ut (array-like): inputs signal for {t:T}
     - yt (array-like): measurements for {t:T}
     - tt (array-like): time stamps for {t:T}
     - maxpdf (float): argmax p(x_{t+1:T}|x_t)
     - max_iter (int): number of attempts before falling back to bsi_full
    """

    pa = ptraj[-1].pa
    M = future_trajs.shape[1]
    todo = numpy.asarray(range(M))
    res = numpy.empty(M, dtype=int)
    weights = numpy.copy(pa.w)
    weights -= numpy.max(weights)
    weights = numpy.exp(weights)
    weights /= numpy.sum(weights)
    for _i in xrange(max_iter):

        ind = numpy.random.permutation(pf.sample(weights, len(todo)))
        pn = model.logp_xnext_full(ptraj, ind,
                                   future_trajs, todo,
                                   ut=ut, yt=yt, tt=tt, ind=cur_ind)
        test = numpy.log(numpy.random.uniform(size=len(todo)))
        accept = test < pn - maxpdf
        res[todo[accept]] = ind[accept]
        todo = todo[~accept]
        if (len(todo) == 0):
            return res

    # TODO, is there an efficient way to store those weights
    # already calculated to avoid double work, or will that
    # take more time than simply evaulating them all again?
    res[todo] = bsi_full(ptraj, model, future_trajs[:, todo], ut=ut, yt=yt, tt=tt, cur_ind=cur_ind)
    return res

def bsi_rsas(ptraj, model, future_trajs, ut, yt, tt, cur_ind, maxpdf, x1, P1, sv, sw, ratio):
    """
    Perform backward simulation by using rejection sampling to draw particles
    from the categorical distribution with weights given by
    \omega_{t|T}^i = \omega_{t|t}^i*p(x_{t+1}|x^i)

    Adaptively determine when to to fallback to bsi_full by using a Kalman
    filter to track the prediceted acceptance rate of the rejection sampler

    Based on "Adaptive Stopping for Fast Particle Smoothing" by
    Taghavi, Lindsten, Svensson and Sch\"{o}n. See orignal article for details
    about the meaning of the Kalman filter paramters

    Args:
     - pa (ParticleApproximation): particles approximation from which to sample
     - model (FFBSi): model defining probability density function
     - future_trajs (array-like): trajectory estimate of {t+1:T}
     - ut (array-like): inputs signal for {t:T}
     - yt (array-like): measurements for {t:T}
     - tt (array-like): time stamps for {t:T}
     - maxpdf (float): argmax p(x_{t+1:T}|x_t)
     - x1 (float): initial state of Kalman filter
     - P1 (float): initial covariance of Kalman filter estimate
     - sv (float): process noise (for Kalman filter)
     - sw (float): measurement noise (for Kalman filter)
     - ratio (float): cost ration of running rejection sampling compared to
       switching to the full bsi (D_0 / D_1)
    """
    pa = ptraj[-1].pa
    M = future_trajs.shape[1]
    todo = numpy.asarray(range(M))
    res = numpy.empty(M, dtype=int)
    weights = numpy.copy(pa.w)
    weights -= numpy.max(weights)
    weights = numpy.exp(weights)
    weights /= numpy.sum(weights)
    pk = x1
    Pk = P1
    stop_criteria = ratio / len(pa)
    while (True):

        ind = numpy.random.permutation(pf.sample(weights, len(todo)))
        pn = model.logp_xnext_full(ptraj, ind,
                                   future_trajs, todo,
                                   ut=ut, yt=yt, tt=tt, ind=cur_ind)
        test = numpy.log(numpy.random.uniform(size=len(todo)))
        accept = test < pn - maxpdf
        ak = numpy.sum(accept)
        mk = len(todo)
        res[todo[accept]] = ind[accept]
        todo = todo[~accept]
        if (len(todo) == 0):
            return res
        # meas update for adaptive stop
        mk2 = mk * mk
        sw2 = sw * sw
        pk = pk + (mk * Pk) / (mk2 * Pk + sw2) * (ak - mk * pk)
        Pk = (1 - (mk2 * Pk) / (mk2 * Pk + sw2)) * Pk
        # predict
        pk = (1 - ak / mk) * pk
        Pk = (1 - ak / mk) ** 2 * Pk + sv * sv
        if (pk < stop_criteria):
            break

    res[todo] = bsi_full(ptraj, model, future_trajs[:, todo], ut=ut, yt=yt, tt=tt, cur_ind=cur_ind)
    return res

def bsi_mcmc(ptraj, model, future_trajs, ut, yt, tt, cur_ind, R, ancestors):
    """
    Perform backward simulation by using Metropolis-Hastings to draw particles
    from the categorical distribution with weights given by
    \omega_{t|T}^i = \omega_{t|t}^i*p(x_{t+1}|x^i)

    Args:
     - pa (ParticleApproximation): particles approximation from which to sample
     - model (FFBSi): model defining probability density function
     - future_trajs (array-like): trajectory estimate of {t+1:T}
     - ut (array-like): inputs signal for {t:T}
     - yt (array-like): measurements for {t:T}
     - tt (array-like): time stamps for {t:T}
     - R (int): number of iterations to run the markov chain
     - ancestor (array-like): ancestor of each particle from the particle filter
    """
    # Perform backward simulation using an MCMC sampler proposing new
    # backward particles, initialized with the filtered trajectory

    M = future_trajs.shape[1]
    find = numpy.asarray(range(M), dtype=int)
    ind = ancestors
    pa = ptraj[-1].pa
    weights = numpy.copy(pa.w)
    weights -= numpy.max(weights)
    weights = numpy.exp(weights)
    weights /= numpy.sum(weights)
    pind = model.logp_xnext_full(ptraj, ind,
                                 future_trajs, find,
                                 ut=ut, yt=yt, tt=tt, ind=cur_ind)
    for _j in xrange(R):
        propind = numpy.random.permutation(pf.sample(weights, M))
        pprop = model.logp_xnext_full(ptraj, propind,
                                      future_trajs, find,
                                      ut=ut, yt=yt, tt=tt, ind=cur_ind)
        diff = pprop - pind
        diff[diff > 0.0] = 0.0
        test = numpy.log(numpy.random.uniform(size=M))
        accept = test < diff
        ind[accept] = propind[accept]
        pind[accept] = pprop[accept]

    return ind

class SmoothTrajectory(object):
    """
    Create smoothed trajectory from filtered trajectory

    Args:
     - pt (ParticleTrajectory): Forward estimates (typically
       generated by a ParticleFilter), combined with inputs and measurements
     - M (int): Number of smoothed trajectories to create
     - method (string): Smoothing method to use
     - options (dict): options to pass on to the smoothing algorithm
    """

    def __init__(self, pt, M=1, method='full', options=None):

        self.traj = None

        self.u = numpy.copy(pt.uvec)
        self.y = numpy.copy(pt.yvec)
        self.t = numpy.copy(pt.tvec)

        self.model = pt.pf.model
        if (method == 'full' or method == 'mcmc' or method == 'rs' or
            method == 'rsas'):
            self.perform_bsi(pt=pt, M=M, method=method, options=options)
        elif (method == 'ancestor'):
            self.perform_ancestors(pt=pt, M=M)
        elif (method == 'mhips'):
            # Initialize using forward trajectories
            self.traj = self.perform_ancestors_int(pt=pt, M=M)

            if 'R' in options:
                R = options['R']
            else:
                R = 10
            for _i in xrange(R):
                # Recover filtering statistics for linear states
                if hasattr(self.model, 'pre_mhips_pass'):
                    self.traj = self.model.pre_mhips_pass(self)
                self.traj = self.perform_mhips_pass(options=options)

            if hasattr(self.model, 'post_smoothing'):
                self.traj = self.model.post_smoothing(self)

        elif (method == 'mhips_reduced'):
            # Initialize using forward trajectories
            self.traj = self.perform_ancestors_int(pt=pt, M=M)

            if 'R' in options:
                R = options['R']
            else:
                R = 10
            for _i in xrange(R):
                # Recover filtering statistics for linear states
                if hasattr(self.model, 'pre_mhips_pass'):
                    self.traj = self.model.pre_mhips_pass(self)
                self.traj = self.perform_mhips_pass_reduced(options=options)

            if hasattr(self.model, 'post_smoothing'):
                self.traj = self.model.post_smoothing(self)

        elif (method == 'mhbp'):
            if 'R' in options:
                R = options['R']
            else:
                R = 10
            self.perform_mhbp(pt=pt, M=M, R=R)
        else:
            raise ValueError('Unknown smoother: %s' % method)

    def __len__(self):
        return len(self.traj)

    def perform_ancestors(self, pt, M):
        """
        Create smoothed trajectories by taking the forward trajectories

        Args:
         - pt (ParticleTrajectory): forward trajetories
         - M (int): number of trajectories to createa
        """
        self.traj = self.perform_ancestors_int(pt, M)

        if hasattr(self.model, 'post_smoothing'):
            # Do e.g. constrained smoothing for RBPS models
            self.traj = self.model.post_smoothing(self)

    def calculate_ancestors(self, pt, ind):
        T = len(pt)
        M = len(ind)
        last_part = self.model.sample_smooth(ptraj=pt, anc=ind,
                                             future_trajs=None,
                                             ut=self.u, yt=self.y,
                                             tt=self.t, cur_ind=T - 1)

        traj = numpy.zeros((T, M, last_part.shape[1]))
        traj[-1] = numpy.copy(last_part)

        ancestors = pt[-1].ancestors[ind]

        for t in reversed(xrange(T - 1)):

            ind = ancestors
            ancestors = pt[t].ancestors[ind]
            # Select 'previous' particle
            traj[t] = numpy.copy(self.model.sample_smooth(ptraj=pt[:(t + 1)],
                                                          anc=ind,
                                                          future_trajs=traj[(t + 1):],
                                                          ut=self.u,
                                                          yt=self.y,
                                                          tt=self.t,
                                                          cur_ind=t))
        return traj

    def perform_ancestors_int(self, pt, M):
        """
        Create smoothed trajectories by taking the forward trajectories, don't
        perform post processing

        Args:
         - pt (ParticleTrajectory): forward trajetories
         - M (int): number of trajectories to createa
        """

        tmp = numpy.copy(pt[-1].pa.w)
        tmp -= numpy.max(tmp)
        tmp = numpy.exp(tmp)
        tmp = tmp / numpy.sum(tmp)
        ind = pf.sample(tmp, M)

        return self.calculate_ancestors(pt, ind)

    def perform_bsi(self, pt, M, method, options):
        """
        Create smoothed trajectories using Backward Simulation

        Args:
         - pt (ParticleTrajectory): forward trajetories
         - M (int): number of trajectories to createa
         - method (string): Type of backward simulation to use
         - optiones (dict): Parameters to the backward simulator
        """

        # Sample from end time estimates
        tmp = numpy.copy(pt[-1].pa.w)
        tmp -= numpy.max(tmp)
        tmp = numpy.exp(tmp)
        tmp = tmp / numpy.sum(tmp)
        ind = pf.sample(tmp, M)
        last_part = self.model.sample_smooth(ptraj=pt, anc=ind,
                                             future_trajs=None,
                                             ut=self.u, yt=self.y,
                                             tt=self, cur_ind=len(pt) - 1)

        self.traj = numpy.zeros((len(pt), M, last_part.shape[1]))
        self.traj[-1] = numpy.copy(last_part)

        if (method == 'full'):
            pass
        elif (method == 'mcmc' or method == 'ancestor' or method == 'mhips'):
            ancestors = pt[-1].ancestors[ind]
        elif (method == 'rs'):
            max_iter = options['R']
        elif (method == 'rsas'):
            x1 = options['x1']
            P1 = options['P1']
            sv = options['sv']
            sw = options['sw']
            ratio = options['ratio']
        else:
            raise ValueError('Unknown sampler: %s' % method)

        for cur_ind in reversed(xrange(len(pt) - 1)):

            ft = self.traj[(cur_ind + 1):]
            ut = self.u
            yt = self.y
            tt = self.t

            if (method == 'rs'):
                ind = bsi_rs(pt[:cur_ind + 1], self.model, ft, ut=ut, yt=yt, tt=tt, cur_ind=cur_ind,
                             maxpdf=options['maxpdf'][cur_ind],
                             max_iter=int(max_iter))
            elif (method == 'rsas'):
                ind = bsi_rsas(pt[:cur_ind + 1], self.model, ft, ut=ut, yt=yt, tt=tt, cur_ind=cur_ind,
                               maxpdf=options['maxpdf'][cur_ind], x1=x1,
                               P1=P1, sv=sv, sw=sw, ratio=ratio)
            elif (method == 'mcmc'):
                ind = bsi_mcmc(pt[:cur_ind + 1], self.model, ft, ut=ut, yt=yt, tt=tt, cur_ind=cur_ind,
                               R=options['R'], ancestors=ancestors)
                ancestors = pt[cur_ind].ancestors[ind]
            elif (method == 'full'):
                ind = bsi_full(pt[:cur_ind + 1], self.model, ft, ut=ut, yt=yt, tt=tt, cur_ind=cur_ind)
            elif (method == 'ancestor'):
                ind = ancestors
                ancestors = pt[cur_ind].ancestors[ind]
            # Select 'previous' particle
            self.traj[cur_ind] = numpy.copy(self.model.sample_smooth(ptraj=pt[:cur_ind + 1],
                                                                     anc=ind,
                                                                     future_trajs=ft, ut=ut,
                                                                     yt=yt,
                                                                     tt=tt,
                                                                     cur_ind=cur_ind))

        if hasattr(self.model, 'post_smoothing'):
            # Do e.g. constrained smoothing for RBPS models
            self.traj = self.model.post_smoothing(self)


    def perform_mhbp(self, pt, M, R):
        """
        Create smoothed trajectories using Metropolis-Hastings Backward Propeser

        Args:
         - pt (ParticleTrajectory): forward trajetories
         - M (int): number of trajectories to createa
         - R (int): Number of proposal for each time step
        """
        T = len(pt)

        # Initialise from end time estimates
        tmp = numpy.copy(pt[-1].pa.w)
        tmp -= numpy.max(tmp)
        tmp = numpy.exp(tmp)
        tmp = tmp / numpy.sum(tmp)
        anc = pf.sample(tmp, M)

        last_part = self.model.sample_smooth(pt, anc=anc,
                                             future_trajs=None,
                                             ut=self.u,
                                             yt=self.y,
                                             tt=self.t,
                                             cur_ind=T - 1)

        self.traj = numpy.zeros((T, M, last_part.shape[1]))

        for t in reversed(xrange(T)):

            # Initialise from filtered estimate
            if (t < T - 1):
                ft = self.traj[(t + 1):]
                self.traj[t] = numpy.copy(self.model.sample_smooth(pt[:t + 1], anc=anc,
                                                                   future_trajs=ft,
                                                                   ut=self.u,
                                                                   yt=self.y,
                                                                   tt=self.t,
                                                                   cur_ind=t))

            else:
                self.traj[t] = numpy.copy(last_part)
                ft = None

            if (t > 0):
                anc = pt[t].ancestors[anc]
                tmp = numpy.copy(pt[t - 1].pa.w)
                tmp -= numpy.max(tmp)
                tmp = numpy.exp(tmp)
                tmp = tmp / numpy.sum(tmp)

            for _ in xrange(R):

                if (t > 0):
                    # Propose new ancestors
                    panc = pf.sample(tmp, M)
                    partp_prop = pt[t - 1].pa.part[panc]
                    partp_curr = pt[t - 1].pa.part[anc]
                    up = self.u[t - 1]
                    tp = self.t[t - 1]
                else:
                    partp_prop = None
                    partp_curr = None
                    up = None
                    tp = None

                (pprop, acc) = mc_step(model=self.model,
                                       partp_prop=partp_prop,
                                       partp_curr=partp_curr,
                                       up=up,
                                       tp=tp,
                                       curpart=self.traj[t, :],
                                       yt=self.y[t:],
                                       ut=self.u[t:],
                                       tt=self.t[t:],
                                       future_trajs=ft)

                # Update with accepted proposals
                self.traj[t, acc] = pprop[acc]
                anc[acc] = panc[acc]

        if hasattr(self.model, 'post_smoothing'):
            # Do e.g. constrained smoothing for RBPS models
            self.traj = self.model.post_smoothing(self)


    def perform_mhips_pass(self, options):
        """
        Perform a single MHIPS pass

        Args:
         - Options (None): Unused
        """
        T = len(self.traj)
        # Handle last time-step seperately
        (prop, acc) = mc_step(model=self.model,
                              partp_prop=self.traj[-2],
                              partp_curr=self.traj[-2],
                              up=self.u[-2], tp=self.t[-2],
                              curpart=self.traj[-1], yt=self.y[-1:],
                              ut=self.u[-1:], tt=self.t[-1:],
                              future_trajs=None)
        self.traj[-1, acc] = prop[acc]
        tmp = self.model.sample_smooth(self.traj[-1],
                                       future_trajs=None,
                                       ut=self.u[-1:],
                                       yt=self.y[-1:],
                                       tt=self.t[-1:])

        straj = numpy.empty((T, tmp.shape[0], tmp.shape[1]))
        straj[-1, :, :tmp.shape[1]] = tmp

        for i in reversed(xrange(1, (T - 1))):

            (prop, acc) = mc_step(model=self.model,
                                  partp_prop=self.traj[i - 1],
                                  partp_curr=self.traj[i - 1],
                                  up=self.u[i - 1], tp=self.t[i - 1],
                                  curpart=self.traj[i], yt=self.y[i:],
                                  ut=self.u[i:], tt=self.t[i:],
                                  future_trajs=straj[(i + 1):])
            # The data dimension is not necessarily the same, since self.traj
            # contains data that has been processed by "post_smoothing".
            # This implementation assumes that the existing trajectory contains
            # enough space to hold the data, if that is not the case the
            # model class should extend "pre_mhips_pass" to allocate a larger
            # array
            self.traj[i, acc] = prop[acc]
            tmp = self.model.sample_smooth(self.traj[i],
                                           future_trajs=straj[(i + 1):],
                                           ut=self.u[i:],
                                           yt=self.y[i:],
                                           tt=self.t[i:])

            straj[i] = tmp

        # Handle last timestep seperately
        (prop, acc) = mc_step(model=self.model,
                              partp_prop=None,
                              partp_curr=None,
                              up=None,
                              tp=None,
                              curpart=self.traj[0], yt=self.y,
                              ut=self.u, tt=self.t,
                              future_trajs=straj[1:])
        self.traj[0, acc] = prop[acc]
        tmp = self.model.sample_smooth(self.traj[0],
                                       future_trajs=straj[1:],
                                       ut=self.u,
                                       yt=self.y,
                                       tt=self.t)

        straj[0] = tmp
        return straj

    def perform_mhips_pass_reduced(self, options):
        """
        Runs MHIPS with the proposal density q as p(x_{t+1}|x_t)

        Args:
         - pt (ParticleTrajectory): Forward esimates
         - M (int): Number of backward trajectories
         - options (None): Unused
        """
        T = len(self.traj)
        # Handle last time-step seperately
        (prop, acc) = mc_step_red(model=self.model,
                                  partp_prop=self.traj[-2],
                                  partp_curr=self.traj[-2],
                                  up=self.u[-2], tp=self.t[-2],
                                  curpart=self.traj[-1], yt=self.y[-1:],
                                  ut=self.u[-1:], tt=self.t[-1:],
                                  future_trajs=None)
        self.traj[-1, acc] = prop[acc]
        tmp = self.model.sample_smooth(self.traj[-1],
                                       future_trajs=None,
                                       ut=self.u[-1:],
                                       yt=self.y[-1:],
                                       tt=self.t[-1:])

        straj = numpy.empty((T, tmp.shape[0], tmp.shape[1]))
        straj[-1, :, :tmp.shape[1]] = tmp

        for i in reversed(xrange(1, (T - 1))):

            (prop, acc) = mc_step_red(model=self.model,
                                      partp_prop=self.traj[i - 1],
                                      partp_curr=self.traj[i - 1],
                                      up=self.u[i - 1], tp=self.t[i - 1],
                                      curpart=self.traj[i], yt=self.y[i:],
                                      ut=self.u[i:], tt=self.t[i:],
                                      future_trajs=straj[(i + 1):])
            # The data dimension is not necessarily the same, since self.traj
            # contains data that has been processed by "post_smoothing".
            # This implementation assumes that the existing trajectory contains
            # enough space to hold the data, if that is not the case the
            # model class should extend "pre_mhips_pass" to allocate a larger
            # array
            self.traj[i, acc] = prop[acc]
            tmp = self.model.sample_smooth(self.traj[i],
                                           future_trajs=straj[(i + 1):],
                                           ut=self.u[i:],
                                           yt=self.y[i:],
                                           tt=self.t[i:])

            straj[i] = tmp

        # Handle last timestep seperately
        (prop, acc) = mc_step_red(model=self.model,
                                  partp_prop=None,
                                  partp_curr=None,
                                  up=None,
                                  tp=None,
                                  curpart=self.traj[0], yt=self.y,
                                  ut=self.u, tt=self.t,
                                  future_trajs=straj[1:])
        self.traj[0, acc] = prop[acc]
        tmp = self.model.sample_smooth(self.traj[0],
                                       future_trajs=straj[1:],
                                       ut=self.u,
                                       yt=self.y,
                                       tt=self.t)

        straj[0] = tmp
        return straj



def mc_step(model, ptraj, curpart, anc_cur, anc_prop, future_trajs, yt, ut, tt, cur_ind):
    """
    Perform a single iteration of the MCMC sampler used for MHIPS and MHBP

    Args:
     - model: model definition
     - partp_prop (array-like): proposed previous particle
     - partp_prop (array-like): current previous particle
     - up (array-like): input at time t-1
     - tp (array-like): timestamp at time t-1
     - curpart: current accepted paricle
     - yt (array-like): measurement at time t
     - ut (array-like): input at time t
     - tt (array-like): timestamp at time t
     - future_trajs (array-like): particle approximations of {x_{t+1:T|T}}
    """
    M = len(anc_prop)

    xprop = model.propose_smooth(ptraj=ptraj,
                                 anc=anc_prop,
                                 future_trajs=future_trajs,
                                 yt=yt,
                                 ut=ut,
                                 tt=tt,
                                 cur_ind=cur_ind
                                 )

    # Accept/reject new sample
    logp_q_prop = model.logp_proposal(xprop,
                                      ptraj=ptraj,
                                      anc=anc_prop,
                                      future_trajs=future_trajs,
                                      yt=yt,
                                      ut=ut,
                                      tt=tt,
                                      cur_ind=cur_ind)
    logp_q_curr = model.logp_proposal(curpart,
                                      ptraj=ptraj,
                                      anc=anc_cur,
                                      future_trajs=future_trajs,
                                      yt=yt,
                                      ut=ut,
                                      tt=tt,
                                      cur_ind=cur_ind)

    if (len(ptraj) > 0):
        find = numpy.arange(M, dtype=int)
        # TODO? Single step forward instead?
        logp_prev_prop = model.logp_xnext_full(past_trajs=ptraj,
                                               pind=anc_prop,
                                               future_trajs=xprop[numpy.newaxis],
                                               find=find,
                                               yt=yt,
                                               ut=ut,
                                               tt=tt,
                                               cur_ind=cur_ind)
        logp_prev_prop = model.logp_xnext_full(past_trajs=ptraj,
                                               pind=anc_cur,
                                               future_trajs=curpart[numpy.newaxis],
                                               find=find,
                                               yt=yt,
                                               ut=ut,
                                               tt=tt,
                                               cur_ind=cur_ind)
    else:
        logp_prev_prop = numpy.zeros(M)
        logp_prev_curr = numpy.zeros(M)

    xpropy = numpy.copy(xprop)
    curparty = numpy.copy(curpart)
    if (yt != None and yt[0] != None):
        logp_y_prop = model.measure(particles=xpropy,
                                    y=yt[0], t=tt[0])

        logp_y_curr = model.measure(particles=curparty,
                                    y=yt[0], t=tt[0])
    else:
        logp_y_prop = numpy.zeros(M)
        logp_y_curr = numpy.zeros(M)

    if (future_trajs != None):
        logp_next_prop = model.logp_xnext_full(particles=xpropy,
                                               future_trajs=future_trajs,
                                               ut=ut,
                                               yt=yt,
                                               tt=tt)
        logp_next_curr = model.logp_xnext_full(particles=curparty,
                                               future_trajs=future_trajs,
                                               ut=ut,
                                               yt=yt,
                                               tt=tt)
    else:
        logp_next_prop = numpy.zeros(M)
        logp_next_curr = numpy.zeros(M)


    # Calc ratio
    ratio = ((logp_prev_prop - logp_prev_curr) +
             (logp_y_prop - logp_y_curr) +
             (logp_next_prop - logp_next_curr) +
             (logp_q_curr - logp_q_prop))

    test = numpy.log(numpy.random.uniform(size=M))
    acc = test < ratio
    return (xpropy, acc)

def mc_step_red(model, partp_prop, partp_curr, up, tp, curpart,
                yt, ut, tt, future_trajs):
    """
    Perform a single iteration of the MCMC sampler used for MHIPS and MHBP

    Args:
     - model: model definition
     - partp_prop (array-like): proposed previous particle
     - partp_prop (array-like): current previous particle
     - up (array-like): input at time t-1
     - tp (array-like): timestamp at time t-1
     - curpart: current accepted paricle
     - yt (array-like): measurement at time t
     - ut (array-like): input at time t
     - tt (array-like): timestamp at time t
     - future_trajs (array-like): particle approximations of {x_{t+1:T|T}}
    """
    M = len(curpart)

    if (partp_prop != None):
        noise = model.sample_process_noise(partp_prop, up, tp)
        xprop = numpy.copy(partp_prop)
        model.update(xprop, up, tp, noise)
    else:
        M = future_trajs.shape[1]
        xprop = model.create_initial_estimate(M)

    xpropy = numpy.copy(xprop)
    curparty = numpy.copy(curpart)
    if (yt != None and yt[0] != None):
        logp_y_prop = model.measure(particles=xpropy,
                                    y=yt[0], t=tt[0])

        logp_y_curr = model.measure(particles=curparty,
                                    y=yt[0], t=tt[0])
    else:
        logp_y_prop = numpy.zeros(M)
        logp_y_curr = numpy.zeros(M)

    if (future_trajs != None):
        logp_next_prop = model.logp_xnext_full(particles=xpropy,
                                               future_trajs=future_trajs,
                                               ut=ut,
                                               yt=yt,
                                               tt=tt)
        logp_next_curr = model.logp_xnext_full(particles=curparty,
                                               future_trajs=future_trajs,
                                               ut=ut,
                                               yt=yt,
                                               tt=tt)
    else:
        logp_next_prop = numpy.zeros(M)
        logp_next_curr = numpy.zeros(M)


    # Calc ratio
    ratio = ((logp_y_prop - logp_y_curr) +
             (logp_next_prop - logp_next_curr))

    test = numpy.log(numpy.random.uniform(size=M))
    acc = test < ratio
    return (xpropy, acc)
