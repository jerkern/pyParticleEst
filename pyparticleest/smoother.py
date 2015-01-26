""" Collection of smoothing algorithms and support classes for using them

@author: Jerker Nordh
"""

import numpy
import filter as pf
import copy
from filter import ParticleApproximation, TrajectoryStep

def bsi_full(model, pa, ptraj, pind, future_trajs, find, ut, yt, tt, cur_ind):
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

    M = len(find)
    N = len(pa.w)
    res = numpy.empty(M, dtype=int)
    #pind = numpy.asarray(range(N))
    for j in xrange(M):
        currfind = find[j] * numpy.ones((N,), dtype=int)
        p_next = model.logp_xnext_full(pa.part, ptraj, pind,
                                       future_trajs, currfind,
                                       ut=ut, yt=yt, tt=tt, cur_ind=cur_ind)

        w = pa.w + p_next
        w = w - numpy.max(w)
        w_norm = numpy.exp(w)
        w_norm /= numpy.sum(w_norm)
        res[j] = pf.sample(w_norm, 1)
    return res


def bsi_rs(model, pa, ptraj, pind, future_trajs, find, ut, yt, tt, cur_ind, maxpdf, max_iter):
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

    M = len(find)
    todo = numpy.asarray(range(M))
    res = numpy.empty(M, dtype=int)
    weights = numpy.copy(pa.w)
    weights -= numpy.max(weights)
    weights = numpy.exp(weights)
    weights /= numpy.sum(weights)
    for _i in xrange(max_iter):

        ind = numpy.random.permutation(pf.sample(weights, len(todo)))
        pn = model.logp_xnext_full(pa.part[ind], ptraj, pind[ind],
                                   future_trajs, todo,
                                   ut=ut, yt=yt, tt=tt, cur_ind=cur_ind)
        test = numpy.log(numpy.random.uniform(size=len(todo)))
        accept = test < pn - maxpdf
        res[todo[accept]] = ind[accept]
        todo = todo[~accept]
        if (len(todo) == 0):
            return res

    # TODO, is there an efficient way to store those weights
    # already calculated to avoid double work, or will that
    # take more time than simply evaulating them all again?
    res[todo] = bsi_full(model, pa, ptraj, pind, future_trajs, todo, ut=ut, yt=yt, tt=tt, cur_ind=cur_ind)
    return res

def bsi_rsas(model, pa, ptraj, pind, future_trajs, find, ut, yt, tt, cur_ind, maxpdf, x1, P1, sv, sw, ratio):
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
    M = len(find)
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
        pn = model.logp_xnext_full(pa.part[ind], ptraj, pind[ind],
                                   future_trajs, todo,
                                   ut=ut, yt=yt, tt=tt, cur_ind=cur_ind)
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

    res[todo] = bsi_full(model, pa, ptraj, pind, future_trajs, todo, ut=ut, yt=yt, tt=tt, cur_ind=cur_ind)
    return res

def bsi_mcmc(model, pa, ptraj, pind, future_trajs, find, ut, yt, tt, cur_ind, R, ancestors):
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

    M = len(find)
    ind = ancestors
    weights = numpy.copy(pa.w)
    weights -= numpy.max(weights)
    weights = numpy.exp(weights)
    weights /= numpy.sum(weights)

    pcurr = model.logp_xnext_full(pa.part[ind], ptraj, pind[ind],
                                  future_trajs, find,
                                  ut=ut, yt=yt, tt=tt, cur_ind=cur_ind)
    for _j in xrange(R):
        propind = numpy.random.permutation(pf.sample(weights, M))
        pprop = model.logp_xnext_full(pa.part[propind], ptraj, pind[propind],
                                   future_trajs, find,
                                   ut=ut, yt=yt, tt=tt, cur_ind=cur_ind)
        diff = pprop - pcurr
        diff[diff > 0.0] = 0.0
        test = numpy.log(numpy.random.uniform(size=M))
        accept = test < diff
        ind[accept] = propind[accept]
        pcurr[accept] = pprop[accept]

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
        self.M = M

        self.model = pt.pf.model
        if (method == 'full' or method == 'mcmc' or method == 'rs' or
            method == 'rsas'):
            self.perform_bsi(pt=pt, M=M, method=method, options=options)
        elif (method == 'ancestor'):
            self.perform_ancestors(pt=pt, M=M)
        elif (method == 'mhips' or method == 'mhips_reduced'):
            if (method == 'mhips'):
                reduced = False
            else:
                reduced = True
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
                self.traj = self.perform_mhips_pass_reduced(options=options, reduced=reduced)

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
        ancestors = pt[-1].ancestors[ind]
        find = numpy.arange(M, dtype=int)
        last_part = self.model.sample_smooth(part=pt[-1].pa.part[ind],
                                             ptraj=pt[:-1], anc=ancestors,
                                             future_trajs=None, find=None,
                                             ut=self.u, yt=self.y,
                                             tt=self.t, cur_ind=T - 1)


        traj = numpy.empty((len(pt),), dtype=object)
        traj[-1] = TrajectoryStep(ParticleApproximation(last_part),
                                       numpy.arange(M, dtype=int))

        for t in reversed(xrange(T - 1)):

            ind = ancestors
            ancestors = pt[t].ancestors[ind]
            # Select 'previous' particle
            traj[t] = TrajectoryStep(ParticleApproximation(self.model.sample_smooth(part=pt[t].pa.part[ind],
                                                          ptraj=pt[:t],
                                                          anc=ancestors,
                                                          future_trajs=traj[(t + 1):],
                                                          find=find,
                                                          ut=self.u,
                                                          yt=self.y,
                                                          tt=self.t,
                                                          cur_ind=t)))
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
        ancestors = pt[-1].ancestors[ind]
        last_part = self.model.sample_smooth(part=pt[-1].pa.part[ind],
                                             ptraj=pt[:-1], anc=ancestors,
                                             future_trajs=None, find=None,
                                             ut=self.u, yt=self.y,
                                             tt=self.t, cur_ind=len(pt) - 1)
        self.traj = numpy.empty((len(pt),), dtype=object)
        self.traj[-1] = TrajectoryStep(ParticleApproximation(last_part),
                                       numpy.arange(M, dtype=int))

        if (method == 'full'):
            pass
        elif (method == 'mcmc' or method == 'ancestor' or method == 'mhips'):
            pass
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

        find = numpy.arange(M, dtype=numpy.int)

        for cur_ind in reversed(xrange(len(pt) - 1)):

            ft = self.traj[(cur_ind + 1):]
            ut = self.u
            yt = self.y
            tt = self.t

            if (method == 'rs'):
                ind = bsi_rs(self.model, pt[cur_ind].pa,
                             pt[:cur_ind], pt[cur_ind].ancestors,
                             ft, find,
                             ut=ut, yt=yt, tt=tt, cur_ind=cur_ind,
                             maxpdf=options['maxpdf'][cur_ind],
                             max_iter=int(max_iter))
            elif (method == 'rsas'):
                ind = bsi_rsas(self.model, pt[cur_ind].pa,
                               pt[:cur_ind], pt[cur_ind].ancestors,
                               ft, find,
                               ut=ut, yt=yt, tt=tt, cur_ind=cur_ind,
                               maxpdf=options['maxpdf'][cur_ind], x1=x1,
                               P1=P1, sv=sv, sw=sw, ratio=ratio)
            elif (method == 'mcmc'):
                ind = bsi_mcmc(self.model, pt[cur_ind].pa,
                               pt[:cur_ind], pt[cur_ind].ancestors,
                               ft, find,
                               ut=ut, yt=yt, tt=tt, cur_ind=cur_ind,
                               R=options['R'], ancestors=ancestors)
                ancestors = pt[cur_ind].ancestors[ind]
            elif (method == 'full'):
                ind = bsi_full(self.model, pt[cur_ind].pa,
                               pt[:cur_ind], pt[cur_ind].ancestors,
                               ft, find,
                               ut=ut, yt=yt, tt=tt, cur_ind=cur_ind)
            elif (method == 'ancestor'):
                ind = ancestors

            ancestors = pt[cur_ind].ancestors[ind]
            # Select 'previous' particle
            find = numpy.arange(M, dtype=int)
            self.traj[cur_ind] = TrajectoryStep(ParticleApproximation(
                                    self.model.sample_smooth(part=pt[cur_ind].pa.part[ind],
                                                             ptraj=pt[:cur_ind],
                                                             anc=ancestors,
                                                             future_trajs=ft,
                                                             find=find,
                                                             ut=ut,
                                                             yt=yt,
                                                             tt=tt,
                                                             cur_ind=cur_ind)))

#        if hasattr(self.model, 'post_smoothing'):
#            # Do e.g. constrained smoothing for RBPS models
#            self.traj = self.model.post_smoothing(self)


    def perform_mhbp(self, pt, M, R, reduced=False):
        """
        Create smoothed trajectories using Metropolis-Hastings Backward Propeser

        Args:
         - pt (ParticleTrajectory): forward trajetories
         - M (int): number of trajectories to createa
         - R (int): Number of proposal for each time step
        """
        T = len(pt)

        if (reduced):
            mc = mc_step_red
        else:
            mc = mc_step

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


    def perform_mhips_pass(self, options, reduced=False):
        """
        Runs MHIPS with the proposal density q as p(x_{t+1}|x_t)

        Args:
         - pt (ParticleTrajectory): Forward esimates
         - M (int): Number of backward trajectories
         - options (None): Unused
        """

        if (reduced):
            mc = mc_step_red
        else:
            mc = mc_step

        T = len(self.traj)
        # Handle last time-step seperately
        ut = self.u
        yt = self.y
        tt = self.t
        pind = numpy.arange(self.M, dtype=numpy.int)

        tmp = numpy.copy(self.model.sample_smooth(part=self.traj[T - 1].pa.part,
                                                  ptraj=self.traj[:(T - 1)],
                                                  anc=pind,
                                                  future_trajs=None,
                                                  find=pind,
                                                  ut=ut,
                                                  yt=yt,
                                                  tt=tt,
                                                  cur_ind=T - 1))

        straj = numpy.empty((T,), dtype=object)
        straj[T - 1] = TrajectoryStep(ParticleApproximation(tmp))

        (prop, acc) = mc(model=self.model,
                         pa=self.traj[-1].pa,
                         ptraj=self.traj[:-1],
                         pind_prop=pind,
                         pind_curr=pind,
                         future_trajs=None, find=pind,
                         ut=ut, yt=yt, tt=tt, cur_ind=T - 1)

        tmp = numpy.copy(self.model.sample_smooth(part=prop[acc],
                                                  ptraj=self.traj[:-1],
                                                  anc=pind[acc],
                                                  future_trajs=None,
                                                  find=pind[acc],
                                                  ut=ut,
                                                  yt=yt,
                                                  tt=tt,
                                                  cur_ind=T - 1))

        straj[T - 1].pa.part[acc] = tmp


        for i in reversed(xrange(1, (T - 1))):

            tmp = numpy.copy(self.model.sample_smooth(part=self.traj[i].pa.part,
                                                  ptraj=self.traj[:i],
                                                  anc=pind,
                                                  future_trajs=straj[i + 1:],
                                                  find=pind,
                                                  ut=ut,
                                                  yt=yt,
                                                  tt=tt,
                                                  cur_ind=i))

            straj[i] = TrajectoryStep(ParticleApproximation(tmp))

            (prop, acc) = mc(model=self.model,
                             pa=self.traj[i].pa,
                             ptraj=self.traj[:i],
                             pind_prop=pind,
                             pind_curr=pind,
                             future_trajs=straj[(i + 1):], find=pind,
                             ut=ut, yt=yt, tt=tt, cur_ind=i,
                             )

            # The data dimension is not necessarily the same, since self.traj
            # contains data that has been processed by "post_smoothing".
            # This implementation assumes that the existing trajectory contains
            # enough space to hold the data, if that is not the case the
            # model class should extend "pre_mhips_pass" to allocate a larger
            # array
            #self.traj[i].pa.part[acc] = prop[acc]
            tmp = self.model.sample_smooth(part=prop[acc],
                                           ptraj=self.traj[:i],
                                           anc=pind[acc],
                                           future_trajs=straj[(i + 1):],
                                           find=pind[acc],
                                           ut=ut,
                                           yt=yt,
                                           tt=tt,
                                           cur_ind=i)

            straj[i].pa.part[acc] = tmp

        # Handle last timestep seperately
        tmp = numpy.copy(self.model.sample_smooth(part=self.traj[0].pa.part,
                                                  ptraj=None,
                                                  anc=pind,
                                                  future_trajs=straj[1:],
                                                  find=pind,
                                                  ut=ut,
                                                  yt=yt,
                                                  tt=tt,
                                                  cur_ind=0))

        straj[0] = TrajectoryStep(ParticleApproximation(tmp))

        (prop, acc) = mc(model=self.model,
                         pa=self.traj[0].pa,
                         ptraj=None,
                         pind_prop=None,
                         pind_curr=None,
                         future_trajs=straj[1:],
                         find=pind,
                         ut=ut, yt=yt, tt=tt, cur_ind=i,
                         )

        tmp = self.model.sample_smooth(prop[acc],
                                       ptraj=None,
                                       anc=pind,
                                       future_trajs=straj[1:],
                                       find=pind,
                                       ut=ut,
                                       yt=yt,
                                       tt=tt,
                                       cur_ind=0)

        straj[0].pa.part[acc] = tmp

        return straj


#def bsi_full(ptraj, model, future_trajs, ut, yt, tt, cur_ind):
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
                                    y=yt[cur_ind], t=tt[cur_ind])

        logp_y_curr = model.measure(particles=curparty,
                                    y=yt[cur_ind], t=tt[cur_ind])
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


def mc_step_red(model, pa, ptraj, pind_prop, pind_curr, future_trajs, find, ut, yt, tt,
                cur_ind):
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


    if (ptraj != None):
        M = len(pind_prop)

        noise = model.sample_process_noise_full(ptraj=ptraj,
                                                ancestors=pind_prop,
                                                ut=ut[:cur_ind],
                                                tt=tt[:cur_ind])

        xprop = numpy.copy(pa.part[pind_prop])

        model.update_full(particles=xprop, traj=ptraj,
                          uvec=ut[:cur_ind], yvec=yt[:cur_ind],
                          tvec=tt[:cur_ind],
                          ancestors=pind_prop, noise=noise)
    else:
        M = len(future_trajs[0].pa.part)
        xprop = model.create_initial_estimate(M)



    xpropy = numpy.copy(xprop)
    # TODO: Doesn't ptraj store particles updated with the new measurements=
    curparty = numpy.copy(pa.part)
    if (yt[cur_ind] != None):
        logp_y_prop = model.measure_full(particles=xpropy, traj=ptraj,
                                         uvec=ut[:cur_ind + 1], yvec=yt[:(cur_ind + 1)],
                                         tvec=tt[:cur_ind + 1], ancestors=pind_prop)

        logp_y_curr = model.measure_full(particles=curparty, traj=ptraj,
                                         uvec=ut[:cur_ind + 1], yvec=yt[:(cur_ind + 1)],
                                         tvec=tt[:cur_ind + 1], ancestors=pind_curr)
    else:
        logp_y_prop = numpy.zeros(M)
        logp_y_curr = numpy.zeros(M)

    if (future_trajs != None):
#logp_xnext_full(self, part, past_trajs, pind, future_trajs, find, ut, yt, tt, cur_ind):
        logp_next_prop = model.logp_xnext_full(part=xpropy,
                                               past_trajs=ptraj,
                                               pind=pind_prop,
                                               future_trajs=future_trajs,
                                               find=find,
                                               ut=ut,
                                               yt=yt,
                                               tt=tt,
                                               cur_ind=cur_ind)
        logp_next_curr = model.logp_xnext_full(part=curparty,
                                               past_trajs=ptraj,
                                               pind=pind_curr,
                                               future_trajs=future_trajs,
                                               find=find,
                                               ut=ut,
                                               yt=yt,
                                               tt=tt,
                                               cur_ind=cur_ind)
    else:
        logp_next_prop = numpy.zeros(M)
        logp_next_curr = numpy.zeros(M)


    # Calc ratio
    ratio = ((logp_y_prop - logp_y_curr) +
             (logp_next_prop - logp_next_curr))

    test = numpy.log(numpy.random.uniform(size=M))
    acc = test < ratio
    return (xpropy, acc)
