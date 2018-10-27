""" Model definition for base class for Mixed Linear/Nonlinear Gaussian systems

@author: Jerker Nordh
"""
from pyparticleest.models.rbpf import RBPSBase
import scipy.linalg
import numpy.random
import math

try:
    import pyparticleest.utils.ckalman as kalman
    import pyparticleest.utils.cmlnlg_compute as mlnlg_compute
except Exception:
    print("Falling back to pure python implementaton, expect horrible performance")
    import pyparticleest.utils.kalman as kalman
    import pyparticleest.utils.mlnlg_compute as mlnlg_compute

from builtins import range

class MixedNLGaussianSampled(RBPSBase):
    """
    Base class for particles of the type mixed linear/non-linear with additive gaussian noise.

    Implement this type of system by extending this class and provide the methods for returning
    the system matrices at each time instant.

        xi_{t+1} = f_xi + A_xi*z_t + v_xi,
        z_{t+1} = f_z + A_z*z_t + v_z,
        y_t = h + C*z_t + e,
        (v_xi, v_z)^T ~ N(0, (Q_xi, Qxiz \\ Qxiz^T, Qz))
        e ~ N (0, R)

    Args:
     - lxi (int): number of nonlinear states
     - lz (int): number of linear states
     - Az (arraylike): Az (if constant)
     - C (arraylike): C (if constant)
     - Qz (arraylike): Qz (if constant)
     - R (arraylike): R (if constant)
     - fz (arraylike): fz (if constant)
     - Axi (arraylike): Axi (if constant)
     - Qxi (arraylike): Qxi (if constant)
     - Qxiz (arraylike): Qxiz (if constant)
     - fxi (arraylike): fxi (if constant)
     - h (arraylike): h (if constant)
     - params (array-like): model parameters (if any)
    """
    def __init__(self, lxi, lz, Az=None, C=None, Qz=None, R=None, fz=None,
                 Axi=None, Qxi=None, Qxiz=None, fxi=None, h=None, params=None,
                 **kwargs):
        if (Axi is not None):
            self.Axi = numpy.copy(Axi)
        else:
            self.Axi = None
        if (fxi is not None):
            self.fxi = numpy.copy(fxi)
        else:
            self.fxi = numpy.zeros((lxi, 1))
        if (Qxi is not None):
            self.Qxi = numpy.copy(Qxi)
        else:
            self.Qxi = None
        if (Qxiz is not None):
            self.Qxiz = numpy.copy(Qxiz)
        else:
            self.Qxiz = None

        self.lxi = lxi

        return super(MixedNLGaussianSampled, self).__init__(lz=lz,
                                                     Az=Az, C=C,
                                                     Qz=Qz, R=R,
                                                     hz=h, fz=fz, **kwargs)

    def set_dynamics(self, Az=None, fz=None, Qz=None, R=None,
                     Axi=None, fxi=None, Qxi=None, Qxiz=None,
                     C=None, h=None):
        """
        Update dynamics, typically used when changing the system dynamics
        due to a parameter change

        Args:
         - lxi (int): number of nonlinear states
         - lz (int): number of linear states
         - Az (arraylike): Az (if constant)
         - C (arraylike): C (if constant)
         - Qz (arraylike): Qz (if constant)
         - R (arraylike): R (if constant)
         - fz (arraylike): fz (if constant)
         - Axi (arraylike): Axi (if constant)
         - Qxi (arraylike): Qxi (if constant)
         - Qxiz (arraylike): Qxiz (if constant)
         - fxi (arraylike): fxi (if constant)
         - h (arraylike): h (if constant)
        """
        super(MixedNLGaussianSampled, self).set_dynamics(Az=Az, C=C, Qz=Qz, R=R, fz=fz, hz=h)

        if (Axi is not None):
            self.Axi = numpy.copy(Axi)
        if (Az is not None):
            self.Az = numpy.copy(Az)
        if (Qxi is not None):
            self.Qxi = numpy.copy(Qxi)
        if (Qxiz is not None):
            self.Qxiz = numpy.copy(Qxiz)
        if (Qz is not None):
            self.Qz = numpy.copy(self.kf.Q)
        if (fz is not None):
            self.fz = numpy.copy(self.kf.f_k)
        if (fxi is not None):
            self.fxi = numpy.copy(fxi)

    def sample_process_noise(self, particles, u, t):
        """
        Return sampled process noise for the non-linear states

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - u (array-like):  input signal
         - t (float): time-stamp

        Returns:
         (array-like) with first dimension = N
        """
        (Axi, _, Qxi, _, _, _) = self.get_nonlin_pred_dynamics_int(particles=particles, u=u, t=t)
        (_xil, _, Pl) = self.get_states(particles)
        N = len(particles)
        # This is probably not so nice performance-wise, but will
        # work initially to profile where the bottlenecks are.

        dim = len(_xil[0])
        noise = numpy.empty((N, dim))
        zeros = numpy.zeros(dim)
        for i in range(N):
            Sigma = Qxi[i] + Axi[i].dot(Pl[i]).dot(Axi[i].T)
            noise[i] = numpy.random.multivariate_normal(zeros, Sigma).ravel()
        return noise

    def calc_xi_next(self, particles, noise, u, t):
        """
        Calculate the next nonlinear state given the input and noise
        realization

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - u (array-like): input signal
         - t (float): time stamp
         - noise (array-like): noise realization for each particle

        Returns:
         (array-like): xi values for future particles
        """
        # the noise term here includes the uncertainty from z
        xi_pred = self.pred_xi(particles=particles, u=u, t=t)

        # additive Gaussian noise
        xi_next = xi_pred + noise

        return xi_next

    def pred_xi(self, particles, u, t):
        """
        Predict the next nonlinear state given the input

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - u (array-like): input signal
         - t (float): time stamp

        Returns:
         (array-like): xi values for future particles
        """

        N = len(particles)
        (Axi, fxi, _, _, _, _) = self.get_nonlin_pred_dynamics_int(particles=particles, u=u, t=t)
        (xil, zl, _Pl) = self.get_states(particles)
        dim = len(xil[0])
        xi_next = numpy.empty((N, dim))
        # This is probably not so nice performance-wise, but will
        # work initially to profile where the bottlenecks are.
        for i in range(N):
            xi_next[i] = Axi[i].dot(zl[i]) + fxi[i]
        return xi_next

    def meas_xi_next(self, particles, xi_next, u, t):
        """
        Update estimate using observation of next state

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - xi_next (array-like): future nonlinear states
         - u (array-like): input signal
         - t (float): time stamp
        """
        # This is what is sometimes called "the second measurement update"
        # for Rao-Blackwellized particle filters

        N = len(particles)
        (xil, zl, Pl) = self.get_states(particles)
        (Axi, fxi, Qxi, _, _, _) = self.get_nonlin_pred_dynamics_int(particles=particles, u=u, t=t)
        for i in range(N):
            self.kf.measure_full(y=xi_next[i].reshape((self.lxi, 1)),
                                 z=zl[i].reshape((self.kf.lz, 1)),
                                 P=Pl[i], C=Axi[i], h_k=fxi[i], R=Qxi[i])

        # Predict next states conditioned on eta_next
        self.set_states(particles, xil, zl, Pl)

    def get_cross_covariance(self, particles, u, t):
        """
        Return cross-covariance between noise for nonlinear
        and linear states

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - u (array-like): input signal
         - t (float): time stamp

        Returns:
         - (array-like): Qxiz(xi_t, u_t, t) for each particle
        """
        return None

    def calc_cond_dynamics(self, particles, xi_next, u, t):
        """
        Calculates the linear dynamics for each particle

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - xi_next (array-like): next non linear state
         - u (array-like): input signal
         - t (float): time stamp

        Returns:
         (Az, fz, Qz):
          - Az (array-like): Az matrix for each particle
          - fz (array-like): fz vector for each particle
          - Qz (array-lie): Noise covariance for each particle

        """
        # Compensate for noise correlation
        N = len(particles)
        # (xil, zl, Pl) = self.get_states(particles)

        (Az, fz, Qz, _, _, _) = self.get_lin_pred_dynamics_int(particles=particles, u=u, t=t)

        Qxiz = self.get_cross_covariance(particles=particles, u=u, t=t)
        if (Qxiz is None and self.Qxiz is None):
            return (Az, fz, Qz)
        if (Qxiz is None):
            Qxiz = N * (self.Qxiz,)

        (Axi, fxi, Qxi, _, _, _) = self.get_nonlin_pred_dynamics_int(particles=particles, u=u, t=t)

        Acond = list()
        fcond = list()
        Qcond = list()

        for i in range(N):
            # TODO linalg.solve instead?
            tmp = Qxiz[i].T.dot(scipy.linalg.inv(Qxi[i]))
            Acond.append(Az[i] - tmp.dot(Axi[i]))
            fcond.append(fz[i] + tmp.dot(xi_next[i] - fxi[i]))
            Qcond.append(Qz[i])

        return (Acond, fcond, Qcond)

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
        xin = self.pred_xi(part, u, t)
        self.cond_predict(part, xin, u, t)
        return self.measure(part, y, t + 1)


    def measure(self, particles, y, t):
        """
        Return the log-pdf value of the measurement and update the statistics
        for the linear states

        Args:

         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - y (array-like):  measurement
         - t (float): time-stamp

        Returns:
         (array-like) with first dimension = N, logp(y|x^i)
        """

        (xil, zl, Pl) = self.get_states(particles)
        N = len(particles)
        (y, Cz, hz, Rz, Cz_identical, _, Rz_identical) = self.get_meas_dynamics_int(particles=particles, y=y, t=t)

        lyz = numpy.empty(N)
        if (Rz_identical):
            if (Cz_identical and Cz[0] is None):
                diff = y - hz
                dim = Rz[0].shape[0]
                if (dim == 1):
                    lyz = kalman.lognormpdf_scalar(diff.ravel(), Rz[0])
                else:
                    Rchol = scipy.linalg.cho_factor(Rz[0])
                    lyz = kalman.lognormpdf_cho_vec(diff.reshape((-1, dim, 1)),
                                                    Rchol)
            else:
                if (Rz[0].shape[0] == 1):
                    for i in range(len(zl)):
                        lyz[i] = self.kf.measure_full_scalar(y=y, z=zl[i],
                                                             P=Pl[i], C=Cz[i],
                                                             h_k=hz[i], R=Rz[i])
                else:
                    for i in range(len(zl)):
                        lyz[i] = self.kf.measure_full(y=y, z=zl[i], P=Pl[i],
                                                      C=Cz[i], h_k=hz[i],
                                                      R=Rz[i])
        else:
            if (Rz[0].shape[0] == 1):
                for i in range(len(zl)):
                    lyz[i] = self.kf.measure_full_scalar(y=y, z=zl[i], P=Pl[i],
                                                         C=Cz[i], h_k=hz[i],
                                                         R=Rz[i])
            else:
                for i in range(len(zl)):
                    lyz[i] = self.kf.measure_full(y=y, z=zl[i], P=Pl[i],
                                                  C=Cz[i], h_k=hz[i], R=Rz[i])

        self.set_states(particles, xil, zl, Pl)
        return lyz

    def calc_A_f_Q(self, particles, u, t):
        """
        Calculate the A, f and Q matrices for the particles. Where A, f and Q
        are the stacked matrices of (A_xi, A_z) and so on

        Args:

         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - u (array-like): input signal
         - y (array-like):  measurement
         - t (float): time-stamp

        Returns:
         (A, f, Q, A_identical f_identical, Q_identical). The last elements
         indicate if the matrices are identical for all particles
        """
        N = len(particles)
        (Az, fz, Qz, Az_identical, fz_identical, Qz_identical) = self.get_lin_pred_dynamics_int(particles=particles, u=u, t=t)
        (Axi, fxi, Qxi, Axi_identical, fxi_identical, Qxi_identical) = self.get_nonlin_pred_dynamics_int(particles=particles, u=u, t=t)
        Qxiz = self.get_cross_covariance(particles=particles, u=u, t=t)
        Qxiz_identical = False

        A_identical = False
        f_identical = False
        Q_identical = False

        if (Qxiz is None):
            Qxiz_identical = True
            if (self.Qxiz is None):
                Qxiz = N * (numpy.zeros((Qxi[0].shape[0], Qz[0].shape[0])),)
            else:
                Qxiz = N * (self.Qxiz,)

        if (Az_identical and Axi_identical):
            A = numpy.repeat(numpy.vstack((Axi[0], Az[0]))[numpy.newaxis], N, 0)
            A_identical = True
        else:
            A = list()
            for i in range(N):
                A.append(numpy.vstack((Axi[i], Az[i])))

        if (fxi_identical and fz_identical):
            f = N * (numpy.vstack((fxi[0], fz[0])),)
            f_identical = True
        else:
            f = list()
            for i in range(N):
                f.append(numpy.vstack((fxi[i], fz[i])))

        if (Qxi_identical and Qz_identical and Qxiz_identical):
            Q = N * (numpy.vstack((numpy.hstack((Qxi[0], Qxiz[0])),
                              numpy.hstack((Qxiz[0].T, Qz[0])))),)
            Q_identical = True
        else:
            Q = list()
            for i in range(N):
                Q.append(numpy.vstack((numpy.hstack((Qxi[i], Qxiz[i])),
                              numpy.hstack((Qxiz[i].T, Qz[i])))))

        return (A, f, Q, A_identical, f_identical, Q_identical)

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
        N = len(particles)
        pmax = numpy.empty(N)
        (_, _, Pl) = self.get_states(particles)
        (A, _, Q, _, _, _) = self.calc_A_f_Q(particles, u=u, t=t)
        dim = self.lxi + self.kf.lz
        l2pi = math.log(2 * math.pi)
        for i in range(N):
            Sigma = Q[i] + A[i].dot(Pl[i]).dot(A[i].T)
            Schol = scipy.linalg.cho_factor(Sigma, check_finite=False)
            ld = numpy.sum(numpy.log(numpy.diag(Schol[0]))) * 2
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

        # During the backward smoothing the next_part contain sampled
        # z-variables, the full distrubition for the z_1:T conditioned on xi_1:T
        # is recovered in the post_smooting step

        N = len(particles)
        Nn = len(next_part)
        if (N > 1 and Nn == 1):
            next_part = numpy.repeat(next_part, N, 0)
        lpx = numpy.empty(N)
        (_, zl, Pl) = self.get_states(particles)
        (A, f, Q, _, _, _) = self.calc_A_f_Q(particles, u=u, t=t)

        for i in range(N):
            x_next = next_part[i, :self.lxi + self.kf.lz].reshape((self.lxi + self.kf.lz, 1))
            xp = f[i] + A[i].dot(zl[i])
            Sigma = A[i].dot(Pl[i]).dot(A[i].T) + Q[i]
            Schol = scipy.linalg.cho_factor(Sigma, check_finite=False)
            lpx[i] = kalman.lognormpdf_cho(x_next - xp, Schol)

        return lpx

    def logp_xnext_singlestep(self, part, past_trajs, pind,
                              future_parts, find, ut, yt, tt, cur_ind):
        """
        Return the log-pdf value for the first step of the future trajectory.
        Needed in e.g MHIPS

        Args:

         - part  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - past_trajs: Trajectory leading up to current time
         - pind: Indices relating part to past_trajs
         - future_parts (array-like): particle estimate for {t+1},
           stored using 'filtered' particle representation, ie. sample_smooth
           has not been performed on them
         - find: Indices relatin part and future_parts
         - ut (array-like): input signals for {1:T}
         - yt (array-like): measurements for {1:T}
         - tt (array-like): time stamps for {1:T}
         - cur_ind: index for current time

        Returns:
         (array-like) with first dimension = N, logp(x_{t+1:T}|x_t^i)
        """

        # This only needs to consider the future nonlinear state
        # Default implemenation for markovian models, just look at the next state
        particles = part
        next_part = future_parts[find]
        u = ut[cur_ind]
        t = tt[cur_ind]

        N = len(particles)

        lpx = numpy.empty(N)
        (_, zl, Pl) = self.get_states(particles)

        (Axi, fxi, Qxi, _, _, _) = self.get_nonlin_pred_dynamics_int(particles=particles, u=u, t=t)
        N = len(particles)

        for i in range(N):
            x_next = next_part[i, :self.lxi].reshape((self.lxi, 1))
            xp = fxi[i] + Axi[i].dot(zl[i])
            Sigma = Qxi[i] + Axi[i].dot(Pl[i]).dot(Axi[i].T)
            Schol = scipy.linalg.cho_factor(Sigma, check_finite=False)
            lpx[i] = kalman.lognormpdf_cho(x_next - xp, Schol)

        return lpx

    def sample_smooth(self, part, ptraj, anc, future_trajs, find, ut, yt, tt, cur_ind):
        """
        Create sampled estimates for the smoothed trajectory. Allows the update
        representation of the particles used in the forward step to include
        additional data in the backward step, can also for certain models be
        used to update the points estimates based on the future information.

        Default implementation uses the same format as forward in time it
        ss part of the ParticleFiltering interface since it is used also when
        calculating "ancestor" trajectories

        Args:
         - part  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - ptraj: array of trajectory step objects from previous time-steps,
           last index is step just before the current
         - anc (array-like): index of the ancestor of each particle in part
         - future_trajs (array-like): particle estimate for {t+1:T}
         - find (array-like): index in future_trajs corresponding to each
           particle in part
         - ut (array-like): input signals for {0:T}
         - yt (array-like): measurements for {0:T}
         - tt (array-like): time stamps for {0:T}
         - cur_ind (int): index of current timestep (in ut, yt and tt)

        Returns:
         (array-like) with first dimension = N
        """

        M = len(part)
        res = numpy.zeros((M, self.lxi + self.kf.lz))
        part = numpy.copy(part)
        (xil, zl, Pl) = self.get_states(part)

        if (future_trajs is not None):
            xinl = future_trajs[0].pa.part[find, :self.lxi].reshape((M, self.lxi, 1))
            znl = future_trajs[0].pa.part[find, self.lxi:].reshape((M, self.kf.lz, 1))
            #(xinl, znl, _unused) = self.get_states(future_trajs[0])
            (Acond, fcond, Qcond) = self.calc_cond_dynamics(part, xinl, u=ut[cur_ind],
                                                            t=tt[cur_ind])

            self.meas_xi_next(part, xinl, u=ut[cur_ind], t=tt[cur_ind])

            (xil, zl, Pl) = self.get_states(part)

            for j in range(M):
                self.kf.measure_full(znl[j], zl[j], Pl[j],
                                     C=Acond[j], h_k=fcond[j], R=Qcond[j])
            #self.set_states(particles, xil, zl, Pl)

        # During the backward smoothing the next_part contain sampled
        # z-variables, the full distrubition for the z_1:T conditioned on xi_1:T
        # is recovered in the post_smooting step

        for j in range(M):
            xi = numpy.copy(xil[j]).ravel()
            z = numpy.random.multivariate_normal(zl[j].ravel(),
                                                 Pl[j]).ravel()
            res[j] = numpy.hstack((xi, z))
        return res

    def propose_smooth(self, ptraj, anc, future_trajs, find, yt, ut, tt, cur_ind):
        """
        Sample from a distribution q(x_t | x_{0:t-1}, x_{t+1:T}, y_0:T)

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
            return self.logp_xnext_singlestep(part=ptraj[-1].pa.part[anc],
                                              past_trajs=ptraj[:-1],
                                              pind=ptraj[-1].ancestors[anc],
                                              future_parts=prop_part,
                                              find=find,
                                              ut=ut, yt=yt, tt=tt,
                                              cur_ind=cur_ind - 1)
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

    def get_pred_dynamics_grad(self, particles, u, t):
        """
        Override this method if (A, f, Q) depends on the parameters

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - u (array-like): input signal
         - t (float): time stamps

        Returns:
         (A_grad, f_grad, Q_grad): Element-wise gradients with respect to all
         the parameters for the system matrices
        """
        return (None, None, None)

    def get_meas_dynamics_grad(self, particles, y, t):
        """
        Override this method if (C, h, R) depends on the parameters

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - y (array-like): measurment
         - t (float): time stamps

        Returns:
         (C_grad, h_grad, R_grad): Element-wise gradients with respect to all
         the parameters for the system matrices
        """
        return (None, None, None)

    def eval_logp_xi0(self, xil):
        """
        Evaluate logprob of the initial non-linear state eta,
        default implementation assumes all are equal, override this
        if another behavior is desired

        Args:
         - xil (list): Initial xi states
        """
        return numpy.zeros(len(xil))

    def eval_logp_xi0_grad(self, xil):
        """
        Evaluate logprob of the initial non-linear state eta,
        default implementation assumes all are equal, override this
        if another behavior is desired

        Args:
         - xil (list): Initial xi states
        """
        return numpy.zeros(self.params.shape)

    def eval_logp_x0(self, particles, t):
        """
        Evaluate sum log p(x_0)

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - t (float): time stamp
        """

        # Calculate l1 according to (19a)
        N = len(particles)
        (xil, zl, Pl) = self.get_states(particles)
        (z0, P0) = self.get_rb_initial(xil)
        lpxi0 = self.eval_logp_xi0(xil)
        lpz0 = numpy.empty((N,))
        for i in range(N):
            z0_diff = zl[i] - z0[i]
            l1 = z0_diff.dot(z0_diff.T) + Pl[i]
            P0cho = scipy.linalg.cho_factor(P0[i], check_finite=False)

            ld = numpy.sum(numpy.log(numpy.diagonal(P0cho[0]))) * 2
            tmp = scipy.linalg.cho_solve(P0cho, l1, check_finite=False)
            lpz0[i] = -0.5 * (ld + numpy.trace(tmp))
        return (lpxi0 + lpz0)

    def eval_logp_x0_val_grad(self, particles, t):
        """
        Evaluate gradient of sum log p(x_0)

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - t (float): time stamp
        """
        lpz0_grad = numpy.zeros(self.params.shape)

        # Calculate l1 according to (19a)
        N = len(particles)
        (xil, zl, Pl) = self.get_states(particles)
        (z0, P0) = self.get_rb_initial(xil)
        (z0_grad, P0_grad) = self.get_rb_initial_grad(xil)
        lpxi0 = self.eval_logp_xi0(xil)
        lpxi0_grad = self.eval_logp_xi0_grad(xil)
        lpz0 = 0.0
        for i in range(N):
            z0_diff = zl[i] - z0[i]
            l1 = z0_diff.dot(z0_diff.T) + Pl[i]
            P0cho = scipy.linalg.cho_factor(P0[i], check_finite=False)

            ld = numpy.sum(numpy.log(numpy.diagonal(P0cho[0]))) * 2
            tmp = scipy.linalg.cho_solve(P0cho, l1, check_finite=False)
            lpz0 -= 0.5 * (ld + numpy.trace(tmp))

            # Calculate gradient
            for j in range(len(self.params)):
                tmp = z0_diff.dot(z0_grad[i][j].T)
                dl1 = -tmp - tmp.T
                lpz0_grad[j] -= 0.5 * mlnlg_compute.compute_logprod_derivative(P0cho, P0_grad[i][j], l1, dl1)

        return ((lpxi0 + lpz0) / N,
                (lpxi0_grad + lpz0_grad) / N)


    def calc_l2(self, xin, zn, Pn, zl, Pl, A, f, M):
        """ Internal helper function """
        N = len(xin)
        dim = self.lxi + self.kf.lz
        xn = numpy.hstack((xin, zn))
        perr = numpy.zeros((N, dim, 1))
        mlnlg_compute.compute_pred_err(N, dim, xn, f, A, zl, perr)
        l2 = numpy.zeros((N, dim, dim))
        mlnlg_compute.compute_l2(N, self.lxi, dim, perr, Pn, A, Pl, M, l2)
        return l2

    def calc_l2_grad(self, xin, zn, Pn, zl, Pl, A, f, M, f_grad, A_grad):
        """ Internal helper function """
        N = len(xin)
        dim = self.lxi + self.kf.lz
        xn = numpy.hstack((xin, zn))
        perr = numpy.zeros((N, dim, 1))
        mlnlg_compute.compute_pred_err(N, dim, xn, f, A, zl, perr)
        l2 = numpy.zeros((N, dim, dim))
        mlnlg_compute.compute_l2(N, self.lxi, dim, perr, Pn, A, Pl, M, l2)
#        diff_l2 = compute_l2_grad(perr, len(self.params), self.lxi, zl, Pl, M, A, f_grad, A_grad)
        diff_l2 = numpy.zeros((N, len(self.params), perr.shape[1], perr.shape[1]), dtype=numpy.double, order='C')
        tmp1 = numpy.zeros((self.lxi + self.kf.lz, self.lxi + self.kf.lz), dtype=numpy.double, order='C')
        tmp2 = numpy.zeros((self.lxi + self.kf.lz, self.kf.lz), dtype=numpy.double, order='C')
        if (f_grad is not None):
            mlnlg_compute.compute_l2_grad_f(N, len(self.params), self.lxi + self.kf.lz, diff_l2,
                              perr, f_grad, tmp1)
        if (A_grad is not None):
            mlnlg_compute.compute_l2_grad_A(N, len(self.params), self.lxi + self.kf.lz, diff_l2,
                              perr, self.lxi, Pn, zl, Pl, M, A, A_grad, tmp1, tmp2)
        return (l2, diff_l2)

    def eval_logp_xnext(self, particles, x_next, u, t):
        """
        Evaluate sum log p(x_{t+1}|x_t)

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - x_next (array-like): future states
         - t (float): time stamp

        Returns: (float)
        """

        # This method differs from logp_xnext in that the x_next contains the
        # Rao-Blackwellized estimates of z condiotioned on the nonlinear
        # trajectory, whereas the other function uses sampled values for z

        # Calculate l2 according to (16)
        N = len(particles)
        lpxn = 0.0

        (_xi, z, P) = self.get_states(particles)
        Mzl = self.get_Mz(particles)
        (xin, zn, Pn) = self.get_states(x_next)

        (A, f, Q, _, _, Q_identical) = self.calc_A_f_Q(particles, u, t)
        l2 = self.calc_l2(xin, zn, Pn, z, P, A, f, Mzl)
        if (Q_identical):
            Qcho = scipy.linalg.cho_factor(Q[0], check_finite=False)
            # (_tmp, ld) = numpy.linalg.slogdet(Q[0])
            ld = numpy.sum(numpy.log(numpy.diagonal(Qcho[0]))) * 2
            for i in range(N):
                tmp = scipy.linalg.cho_solve(Qcho, l2[i], check_finite=False)
                lpxn -= 0.5 * (ld + numpy.trace(tmp))
        else:
            for i in range(N):
                Qcho = scipy.linalg.cho_factor(Q[i], check_finite=False)
                # (_tmp, ld) = numpy.linalg.slogdet(Q[i])
                ld = numpy.sum(numpy.log(numpy.diagonal(Qcho[0]))) * 2
                tmp = scipy.linalg.cho_solve(Qcho, l2[i], check_finite=False)
                lpxn -= 0.5 * (ld + numpy.trace(tmp))

        return lpxn

    def eval_logp_xnext_val_grad(self, particles, x_next, u, t):
        """
        Evaluate value and gradient sum log p(x_{t+1}|x_t)

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - x_next (array-like): future states
         - t (float): time stamp

        Returns: ((float), (array-like))
        """
        N = len(particles)
        lpxn = 0.0
        lpxn_grad = numpy.zeros(self.params.shape)

        (A_grad, f_grad, Q_grad) = self.get_pred_dynamics_grad(particles=particles, u=u, t=t)
        if (A_grad is None and f_grad is None and Q_grad is None):
            lpxn = self.eval_logp_xnext(particles, x_next, u, t)
        else:

            (_xi, zl, Pl) = self.get_states(particles)
            Mzl = self.get_Mz(particles)
            (xin, zn, Pn) = self.get_states(x_next)

            (A, f, Q, _, _, Q_identical) = self.calc_A_f_Q(particles, u, t)


            dim = self.lxi + self.kf.lz

            if (Q_grad is None):
                Q_grad = N * (numpy.zeros((len(self.params), dim, dim)),)

            (l2, l2_grad) = self.calc_l2_grad(xin, zn, Pn, zl, Pl, A, f, Mzl, f_grad, A_grad)
            if (Q_identical):
                Qcho = scipy.linalg.cho_factor(Q[0], check_finite=False)
                # (_tmp, ld) = numpy.linalg.slogdet(Q[0])
                ld = numpy.sum(numpy.log(numpy.diagonal(Qcho[0]))) * 2
                for i in range(N):
                    tmp = scipy.linalg.cho_solve(Qcho, l2[i], check_finite=False)
                    lpxn -= 0.5 * (ld + numpy.trace(tmp))
                    for j in range(len(self.params)):
                        lpxn_grad[j] -= 0.5 * mlnlg_compute.compute_logprod_derivative(Qcho, Q_grad[i][j],
                                                                       l2[i], l2_grad[i][j])
            else:
                for i in range(N):
                    Qcho = scipy.linalg.cho_factor(Q[i], check_finite=False)
                    # (_tmp, ld) = numpy.linalg.slogdet(Q[i])
                    ld = numpy.sum(numpy.log(numpy.diagonal(Qcho[0]))) * 2
                    tmp = scipy.linalg.cho_solve(Qcho, l2[i], check_finite=False)
                    lpxn -= 0.5 * (ld + numpy.trace(tmp))
                    for j in range(len(self.params)):
                        lpxn_grad[j] -= 0.5 * mlnlg_compute.compute_logprod_derivative(Qcho, Q_grad[i][j],
                                                                       l2[i], l2_grad[i][j])

        return (lpxn, lpxn_grad)

    def calc_l3(self, y, zl, Pl, Cl, hl):
        """ internal helper function """
        N = len(zl)
        l3 = numpy.zeros((N, len(y), len(y)))
        for i in range(N):
            meas_diff = self.kf.measurement_diff(y, zl[i], Cl[i], hl[i])
            l3[i] = meas_diff.dot(meas_diff.T) + Cl[i].dot(Pl[i]).dot(Cl[i].T)
        return l3

    def calc_l3_grad(self, y, zl, Pl, Cl, hl, C_grad, h_grad):
        """ internal helper function """
        N = len(zl)
        l3 = numpy.zeros((N, len(y), len(y)))
        diff_l3 = numpy.zeros((N, len(self.params), len(y), len(y)))

        for i in range(N):
            meas_diff = self.kf.measurement_diff(y, zl[i], Cl[i], hl[i])
            l3[i] = meas_diff.dot(meas_diff.T) + Cl[i].dot(Pl[i]).dot(Cl[i].T)

            if (C_grad is not None):
                C_grad = N * (numpy.zeros((len(self.params), len(y), self.kf.lz)),)
                for j in range(len(self.params)):
                    tmp2 = C_grad[i][j].dot(Pl[i]).dot(Cl[i].T)
                    tmp = C_grad[i][j].dot(zl[i]).dot(meas_diff.T)
                    diff_l3[i][j] += -tmp - tmp.T + tmp2 + tmp2.T
            if (h_grad is not None):
                for j in range(len(self.params)):
                    tmp = h_grad[i][j].dot(meas_diff.T)
                    diff_l3[i][j] += -tmp - tmp.T

        return (l3, diff_l3)

    def eval_logp_y(self, particles, y, t):
        """
        Evaluate value of sum log p(y_t|x_t)

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - y (array-like): measurement
         - t (float): time stamp

        Returns: (float)
        """
        N = len(particles)
        (y, Cz, hz, Rz, _, _, Rz_identical) = self.get_meas_dynamics_int(particles, y, t)
        (_xil, zl, Pl) = self.get_states(particles)
        logpy = 0.0
        l3 = self.calc_l3(y, zl, Pl, Cz, hz)
        if (Rz_identical):
            Rzcho = scipy.linalg.cho_factor(Rz[0], check_finite=False)
            # (_tmp, ld) = numpy.linalg.slogdet(Rz[0])
            ld = numpy.sum(numpy.log(numpy.diagonal(Rzcho[0]))) * 2
            for i in range(N):
                tmp = scipy.linalg.cho_solve(Rzcho, l3[i], check_finite=False)
                logpy -= 0.5 * (ld + numpy.trace(tmp))
        else:
            for i in range(N):
            # Calculate l3 according to (19b)
                Rzcho = scipy.linalg.cho_factor(Rz[i], check_finite=False)
                # (_tmp, ld) = numpy.linalg.slogdet(Rz[i])
                ld = numpy.sum(numpy.log(numpy.diagonal(Rzcho[0]))) * 2
                tmp = scipy.linalg.cho_solve(Rzcho, l3[i], check_finite=False)
                logpy -= 0.5 * (ld + numpy.trace(tmp))

        return logpy

    def eval_logp_y_val_grad(self, particles, y, t):
        """
        Evaluate value and gradient of sum log p(y_t|x_t)

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - y (array-like): measurement
         - t (float): time stamp

        Returns: ((float), (array-like))
        """

        N = len(particles)
        logpy = 0.0
        lpy_grad = numpy.zeros(self.params.shape)
        (y, Cz, hz, Rz, _, _, Rz_identical) = self.get_meas_dynamics_int(particles, y, t)
        (C_grad, h_grad, R_grad) = self.get_meas_dynamics_grad(particles=particles, y=y, t=t)
        if (C_grad is None and h_grad is None and R_grad is None):
            logpy = self.eval_logp_y(particles, y, t)
        else:
            if (R_grad is None):
                R_grad = N * (numpy.zeros((len(self.params), len(y), len(y))),)

            (_xil, zl, Pl) = self.get_states(particles)

            (l3, l3_grad) = self.calc_l3_grad(y, zl, Pl, Cz, hz, C_grad, h_grad)

            if (Rz_identical):
                Rzcho = scipy.linalg.cho_factor(Rz[0], check_finite=False)
                # (_tmp, ld) = numpy.linalg.slogdet(Rz[0])
                ld = numpy.sum(numpy.log(numpy.diagonal(Rzcho[0]))) * 2
                for i in range(N):
                    tmp = scipy.linalg.cho_solve(Rzcho, l3[i], check_finite=False)
                    logpy -= 0.5 * (ld + numpy.trace(tmp))
                    for j in range(len(self.params)):
                        lpy_grad[j] -= 0.5 * mlnlg_compute.compute_logprod_derivative(Rzcho, R_grad[i][j],
                                                                      l3[i], l3_grad[i][j])
            else:
                for i in range(N):
                    Rzcho = scipy.linalg.cho_factor(Rzcho, check_finite=False)
                    # (_tmp, ld) = numpy.linalg.slogdet(Rz[i])
                    ld = numpy.sum(numpy.log(numpy.diagonal(Rzcho[0]))) * 2
                    tmp = scipy.linalg.cho_solve(Rzcho, l3[i])
                    logpy -= 0.5 * (ld + numpy.trace(tmp))
                    for j in range(len(self.params)):
                        lpy_grad[j] -= 0.5 * mlnlg_compute.compute_logprod_derivative(Rzcho, R_grad[i][j],
                                                                      l3[i], l3_grad[i][j])

        return (logpy, lpy_grad)


class MixedNLGaussianSampledInitialGaussian(MixedNLGaussianSampled):
    def __init__(self, xi0, z0, Pxi0=None, Pz0=None, **kwargs):

                # No uncertainty in initial state
        self.xi0 = numpy.copy(xi0).reshape((-1, 1))
        if (Pxi0 is None):
            self.Pxi0 = numpy.zeros((len(self.xi0), len(self.xi0)))
        else:
            self.Pxi0 = numpy.copy((Pxi0))
        if (Pz0 is None):
            self.Pz0 = numpy.zeros((len(self.z0), len(self.z0)))
        else:
            self.Pz0 = numpy.copy((Pz0))
        self.z0 = numpy.copy(z0).reshape((-1, 1))
        self.Pz0 = numpy.copy(Pz0)
        super(MixedNLGaussianSampledInitialGaussian, self).__init__(lxi=len(self.xi0),
                                                             lz=len(self.z0),
                                                             **kwargs)

    def create_initial_estimate(self, N):
        """Sample particles from initial distribution

        Args:
         - N (int): Number of particles to sample

        Returns:
         (array-like) with first dimension = N, model specific representation
         of all particles """
        dim = self.lxi + self.kf.lz + self.kf.lz ** 2
        particles = numpy.empty((N, dim))

        for i in range(N):
            particles[i, 0:self.lxi] = numpy.random.multivariate_normal(self.xi0.ravel(), self.Pxi0)
            particles[i, self.lxi:(self.lxi + self.kf.lz)] = numpy.copy(self.z0).ravel()
            particles[i, (self.lxi + self.kf.lz):] = numpy.copy(self.Pz0).ravel()
        return particles

    def get_rb_initial(self, xi0):
        """
        Default implementation has no dependence on xi, override if needed

        Calculate estimate of initial state for linear state condition on the
        nonlinear estimate

        Args:
         - xi0 (array-like): Initial xi states

        Returns:
         (z,P): z is a list of all inital mean values, P is a list of covariance
         matrices
        """

        N = len(xi0)
        z_list = numpy.repeat(self.z0.reshape((1, self.kf.lz, 1)), N, 0)
        P_list = numpy.repeat(self.Pz0.reshape((1, self.kf.lz, self.kf.lz)), N, 0)
        return (z_list, P_list)

    def cond_sampled_initial(self, part, t):
        xi = part[:, :self.lxi]
        (z, P) = self.get_rb_initial(xi)
        particles = numpy.zeros_like(part)
        self.set_states(particles, xi, z, P)
        return particles

    def get_rb_initial_grad(self, xi0):
        """
        Default implementation has no dependence on xi, override if needed

        Calculate gradient estimate of initial state for linear state condition on the
        nonlinear estimate

        Args:
         - xi0 (array-like): Initial xi states

        Returns:
         (z,P): z is a list of element-wise gradients for the inital mean values,
         P is a list of element-wise gradients for the covariance matrices
        """
        N = len(xi0)
        return (N * (numpy.zeros((N, len(self.params), self.kf.lz, 1)),),
                N * (numpy.zeros((N, len(self.params), self.kf.lz, self.kf.lz)),))

    def eval_logp_xi0(self, xil):
        """
        Evaluate logprob of the initial non-linear state eta,
        default implementation assumes all are equal, override this
        if another behavior is desired

        Args:
         - xil (list): Initial xi states
        """
        N = len(xil)
        res = numpy.empty((N,))
        Pchol = scipy.linalg.cho_factor(self.Pxi0, check_finite=False)
        for i in range(N):
            res[i] = kalman.lognormpdf_cho(xil[i] - self.xi0, Pchol)
        return res

    def get_xi_intitial_grad(self, N):
        """
        Calculate gradient of initial xi values (mean and covariance)

        Args:
         - N (int): number of particles
        """
        return (N * (numpy.zeros((len(self.params), self.lxi, 1)),),
                N * (numpy.zeros((len(self.params), self.lxi, self.lxi)),))

    def eval_logp_xi0_grad(self, xil):
        """
        Evaluate logprob of the initial non-linear state eta,
        default implementation assumes all are equal, override this
        if another behavior is desired

        Args:
         - xil (list): Initial xi states
        """
        N = len(xil)
        (xi0_grad, Pxi0_grad) = self.get_xi_intitial_grad(N)
        lpxi0_grad = numpy.zeros(self.params.shape)
        Pxi0cho = scipy.linalg.cho_factor(self.Pxi0, check_finite=False)
        for i in range(N):
            tmp = xil[i] - self.xi0
            l0 = tmp.dot(tmp.T)
            for j in range(len(self.params)):
                tmp2 = tmp.dot(xi0_grad[i][j].T)
                l0_grad = -tmp2 - tmp2.T
                lpxi0_grad[j] -= 0.5 * mlnlg_compute.compute_logprod_derivative(Pxi0cho, Pxi0_grad[i][j], l0, l0_grad)

        return lpxi0_grad

def factor_psd(A):
    """ internal helper function """
    (U, s, V) = numpy.linalg.svd(A)
    return U.dot(numpy.diag(numpy.sqrt(s)))

class MixedNLGaussianMarginalized(MixedNLGaussianSampled):
    """ This class implements a fully marginalized smoother for
        mixed linear/nonlinear models, in contrast to the MixedNLGaussian class
        it never samples the linear states.

        This is somewhat slower, and doesn't readily admit using
        rejection sampling, it is up to the end user which method is best for
        their particular problem """

    def logp_xnext_max(self, particles, u, t):
        raise NotImplementedError("MixedNLGaussianMarginalized doesn't support rejection sampling")

    def calc_prop1(self, particles, next_part, u, t):
        """ internal helper function """
        M = len(particles)
        lxi = self.lxi
        lz = self.kf.lz
        OHind = lxi
        OHlen = lz * lz
        LHind = lxi + OHlen
        LHlen = lz

        xinl = next_part[:, :lxi]
        OHnl = next_part[:, OHind:OHind + OHlen].reshape((M, lz, lz))
        LHnl = next_part[:, LHind:LHind + LHlen].reshape((M, lz, 1))

        logZ = numpy.zeros(M)
        Omega = numpy.zeros_like(OHnl)
        Lambda = numpy.zeros_like(LHnl)

        (Az, fz, Qz, _, _, _) = self.get_lin_pred_dynamics_int(particles=particles,
                                                               u=u, t=t)
        (Axi, fxi, Qxi, _, _, _) = self.get_nonlin_pred_dynamics_int(particles=particles,
                                                                     u=u, t=t)

        for j in range(M):
            F = factor_psd(Qz[j])
            m = LHnl[j] - OHnl[j].dot(fz[j])
            Mt = F.T.dot(OHnl[j]).dot(F) + numpy.eye(lz)
            Tau_t = 0.0
            xidiff = xinl[j] - fxi[j]
            Tau_t += xidiff.T.dot(numpy.linalg.solve(Qxi[j], xidiff))
            Tau_t += fz[j].T.dot(OHnl[j]).dot(fz[j])
            Tau_t += -2.0 * LHnl[j].T.dot(fz[j])
            tmp = F.T.dot(m)
            Tau_t += tmp.T.dot(numpy.linalg.solve(Mt, tmp))

            tmp = F.dot(numpy.linalg.solve(Mt, F.T))

            Omega[j] = (Az[j].T.dot(OHnl[j] - OHnl[j].dot(tmp).dot(OHnl[j])).dot(Az[j]) +
                        Axi[j].T.dot(numpy.linalg.solve(Qxi[j], Axi[j])))

            Lambda[j] = (Az[j].T.dot(numpy.eye(lz) - OHnl[j].dot(tmp)).dot(m) +
                         Axi[j].T.dot(numpy.linalg.solve(Qxi[j], xidiff)))

            logZ[j] = -0.5 * (numpy.linalg.slogdet(Mt)[1] + numpy.linalg.slogdet(Qxi[j])[1] + Tau_t)

        return (logZ, Omega, Lambda)


    def calc_prop3(self, particles, Omega, Lambda, u, t):
        """ internal helper function """
        M = len(particles)
        eta = numpy.zeros(M)
        L = numpy.zeros((M, self.kf.lz, self.kf.lz))
        (_, zl, Pl) = self.get_states(particles)

        for j in range(M):
            Gamma = factor_psd(Pl[j])
            L[j] = Gamma.T.dot(Omega[j]).dot(Gamma) + numpy.eye(self.kf.lz)
            tmp = Gamma.T.dot(Lambda[j] - Omega[j].dot(zl[j]))
            eta[j] = (zl[j].T.dot(Omega[j]).dot(zl[j]) -
                       2.0 * Lambda[j].T.dot(zl[j]) -
                       tmp.T.dot(numpy.linalg.solve(L[j], tmp)))
        return (eta, L)

    def logp_xnext_full(self, part, past_trajs, pind,
                        future_trajs, find, ut, yt, tt, cur_ind):
    #def logp_xnext_full(self, particles, future_trajs, ut, yt, tt):
        """
        Return the log-pdf value for the entire future trajectory.
        Useful for non-markovian modeles, that result from e.g
        marginalized state-space models.

        Default implemention just calls logp_xnext which is enough for
        Markovian models

        Args:

         - part  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - past_trajs: array of trajectory step objects from previous time-steps,
           last index is step just before the current
         - pind (array-like): index of the ancestor of each particle in part
         - future_trajs (array-like): particle estimate for {t+1:T}
         - find (array-like): index in future_trajs corresponding to each
           particle in part
         - ut (array-like): input signals for {0:T}
         - yt (array-like): measurements for {0:T}
         - tt (array-like): time stamps for {0:T}
         - cur_ind (int): index of current timestep (in ut, yt and tt)

        Returns:
         (array-like) with first dimension = N, logp(x_{t+1:T}|x_t^i)
        """

        N = len(part)

        lpx = numpy.empty(N)
        # (_, zl, Pl) = self.get_states(particles)

        (logZ, Omega, Lambda) = self.calc_prop1(part, future_trajs[0].pa.part[find],
                                                ut[cur_ind], tt[cur_ind])
        (eta, L) = self.calc_prop3(part, Omega, Lambda, ut[cur_ind], tt[cur_ind])

        for i in range(N):
            lpx[i] = logZ[i] - 0.5 * (numpy.linalg.slogdet(L[i])[1] + eta[i])


        return lpx

    def sample_smooth(self, part, ptraj, anc, future_trajs, find, ut, yt, tt, cur_ind):
        """
        Calculate statistics needed when evaluating the logp_xnext_full for
        the marginalized trajectory

        Args:

         - part  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - ptraj: array of trajectory step objects from previous time-steps,
           last index is step just before the current
         - anc (array-like): index of the ancestor of each particle in part
         - future_trajs (array-like): particle estimate for {t+1:T}
         - find (array-like): index in future_trajs corresponding to each
           particle in part
         - ut (array-like): input signals for {0:T}
         - yt (array-like): measurements for {0:T}
         - tt (array-like): time stamps for {0:T}
         - cur_ind (int): index of current timestep (in ut, yt and tt)

        Returns:
         (array-like) with first dimension = N
        """
        M = len(part)
        lxi = self.lxi
        lz = self.kf.lz

        if (future_trajs is not None):
            (_, Omega, Lambda) = self.calc_prop1(part, future_trajs[0].pa.part[find],
                                                 ut[cur_ind], tt[cur_ind])

        OHind = lxi
        OHlen = lz * lz

        LHlen = lz
        LHind = lxi + OHlen
        res = numpy.zeros((M, lxi + OHlen + LHlen))

        (_, Cz, hz, Rz, _, _, _) = self.get_meas_dynamics_int(particles=part, y=yt[cur_ind], t=tt[cur_ind])

        res[:, :lxi] = part[:, :lxi]

        if (future_trajs is not None):
            for j in range(M):
                res[j, OHind:OHind + OHlen] = Omega[j].ravel()
                res[j, LHind:] = Lambda[j].ravel()

        for j in range(M):
            if (Cz is not None and Cz[j] is not None):
                tmp = numpy.linalg.solve(Rz[j], Cz[j])
                res[j, OHind:OHind + OHlen] += (Cz[j].T.dot(tmp)).ravel()
            if (yt is not None and yt[cur_ind] is not None):
                res[j, LHind:] += (tmp.T.dot(numpy.asarray(yt[cur_ind]).reshape((-1, 1)) -
                                             hz[j])).ravel()

        return res


class MixedNLGaussianMarginalizedInitialGaussian(MixedNLGaussianMarginalized,
                                                 MixedNLGaussianSampledInitialGaussian):
    pass
