"""Model definition for base class for Linear Time-varying systems
@author: Jerker Nordh
"""
from pyparticleest.interfaces import FFBSi, ParticleFiltering

try:
    import pyparticleest.utils.ckalman as kalman
    import pyparticleest.utils.cmlnlg_compute as mlnlg_compute
except ImportError:
    print("Falling back to pure python implementaton, expect horrible performance")
    import pyparticleest.utils.kalman as kalman
    import pyparticleest.utils.mlnlg_compute as mlnlg_compute

import numpy
import scipy.linalg

from builtins import range


class LTV(FFBSi, ParticleFiltering):
    """
    Base class for particles of the type linear time varying with additive gaussian noise.

    Implement this type of system by extending this class and provide the methods for returning
    the system matrices at each time instant

    z_{t+1} = A*z_t + f + v, v ~ N(0, Q)
    y_t = C*z_t + h + e, e ~ N(0,R)

    Args:
     - z0: Initial mean value of the state estimate
     - P0: Coviariance of initial z estimate
     - A (array-like): A matrix (if constant)
     - C (array-like): C matrix (if constant)
     - Q (array-like): Q matrix (if constant)
     - R (array-like): R matrix (if constant)
     - f (array-like): f vector (if constant)
     - h (array-like): h vector (if constant)
     - params (array-like): model parameters (if any)
    """

    def __init__(self, z0, P0, A=None, C=None, Q=None,
                 R=None, f=None, h=None, params=None, **kwargs):
        self.z0 = numpy.copy(z0).reshape((-1, 1))
        self.P0 = numpy.copy(P0)
        if (f is None):
            f = numpy.zeros_like(self.z0)
        self.kf = kalman.KalmanSmoother(lz=len(self.z0),
                                        A=A, C=C,
                                        Q=Q, R=R,
                                        f_k=f, h_k=h)
        super(LTV, self).__init__(**kwargs)

    def create_initial_estimate(self, N):
        """Sample particles from initial distribution

        Args:
         - N (int): Number of particles to sample, since the estimate is
           deterministic there is no reason for N > 1

        Returns:
         (array-like) with first dimension = N, model specific representation
         of all particles """

        if (N > 1):
            print("N > 1 redundamt for LTV system (N={0})".format(N),)
        lz = len(self.z0)
        dim = lz + lz * lz
        particles = numpy.empty((N, dim))

        for i in range(N):
            particles[i, :lz] = numpy.copy(self.z0).ravel()
            particles[i, lz:] = numpy.copy(self.P0).ravel()
        return particles

    def set_states(self, particles, z_list, P_list):
        """
        Set the estimate of the states

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - z_list (list): list of mean values for z for each particle
         - P_list (list): list of covariance matrices for z for each particle
        """
        lz = len(self.z0)
        N = len(particles)
        for i in range(N):
            particles[i, :lz] = z_list[i].ravel()
            lzP = lz + lz * lz
            particles[i, lz:lzP] = P_list[i].ravel()

    def get_states(self, particles):
        """
        Return the estimates contained in the particles array

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)

        Returns
            (zl, Pl):
             - zl: list of mean values for z
             - Pl: list of covariance matrices for z
        """
        N = len(particles)
        zl = list()
        Pl = list()
        lz = len(self.z0)
        for i in range(N):
            zl.append(particles[i, :lz].reshape(-1, 1))
            lzP = lz + lz * lz
            Pl.append(particles[i, lz:lzP].reshape(self.P0.shape))

        return (zl, Pl)

    def get_pred_dynamics(self, u, t):
        """
        Return matrices describing affine relation of next
        nonlinear state conditioned on the current time and input signal

        z_{t+1} = A*z_t + f + v, v ~ N(0, Q)

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - u (array-like): input signal
         - t (float): time stamp

        Returns:
         (A, f, Q) where each element is a list
         with the corresponding matrix for each particle. None indicates
         that the matrix is identical for all particles and the value stored
         in this class should be used instead
        """
        return (None, None, None)

    def update(self, particles, u, t, noise):
        """ Propagate estimate forward in time

        Args:

         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - u (array-like):  input signal
         - t (float): time-stamp
         - noise: Unused for this type of model

        Returns:
         (array-like) with first dimension = N, particle estimate at time t+1
        """
        # Update linear estimate with data from measurement of next non-linear
        # state
        (zl, Pl) = self.get_states(particles)
        (A, f, Q) = self.get_pred_dynamics(u=u, t=t)
        self.kf.set_dynamics(A=A, Q=Q, f_k=f)
        for i in range(len(zl)):
            # Predict z_{t+1}
            (zl[i], Pl[i]) = self.kf.predict(zl[i], Pl[i])

        # Predict next states conditioned on eta_next
        self.set_states(particles, zl, Pl)
        return particles

    def get_meas_dynamics(self, y, t):
        """
        Return matrices describing affine relation of measurement and current
        state estimates

        y_t = C*z_t + h + e, e ~ N(0,R)

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - y (array-like): measurement
         - t (float): time stamp

        Returns:
         (y, C, h, R): y is a preprocessed measurement, the rest are lists
         with the corresponding matrix for each particle. None indicates
         that the matrix is identical for all particles and the value stored
         in this class should be used instead
        """
        return (y, None, None, None)

    def measure(self, particles, y, t):
        """
        Return the log-pdf value of the measurement and update the statistics
        for the states

        Args:

         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - y (array-like):  measurement
         - t (float): time-stamp

        Returns:
         (array-like) with first dimension = N, logp(y|x^i)
        """

        (zl, Pl) = self.get_states(particles)
        (y, C, h, R) = self.get_meas_dynamics(y=y, t=t)
        self.kf.set_dynamics(C=C, R=R, h_k=h)
        lyz = numpy.empty((len(particles)))
        for i in range(len(zl)):
            # Predict z_{t+1}
            lyz[i] = self.kf.measure(y, zl[i], Pl[i])

        self.set_states(particles, zl, Pl)
        return lyz

    def logp_xnext(self, particles, next_part, u, t):
        """
        Return the log-pdf value for the possible future state 'next'
        given input u.

        Always returns zeros since all particles are always equivalent for this
        type of model

        Args:

         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - next_part: Unused
         - u: Unused
         - t: Unused

        Returns:
         (array-like) with first dimension = N, numpu.zeros((N,))
        """
        # Not needed for Linear Gaussian models, always return 0 (all particles will be identical anyhow)
        N = len(particles)
        return numpy.zeros((N,))

    def sample_process_noise(self, particles, u, t):
        """
        There is no need to sample noise for this type of model

        Args:

         - particles: Unused
         - next_part: Unused
         - u: Unused
         - t: Unused

        Returns:
         None
        """
        return None

    def sample_smooth(self, part, ptraj, anc, future_trajs, find, ut, yt, tt, cur_ind):
        """
        Update sufficient statistics based on the future states

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

        (zl, Pl) = self.get_states(part)
        M = len(part)
        lz = len(self.z0)
        lzP = lz + lz * lz
        res = numpy.empty((M, lz + 2 * lz ** 2))
        for j in range(M):
            if (future_trajs is not None):
                zn = future_trajs[0].pa.part[j, :lz].reshape((lz, 1))
                Pn = future_trajs[0].pa.part[j, lz:lzP].reshape((lz, lz))
                (A, f, Q) = self.get_pred_dynamics(u=ut[0], t=tt[0])
                self.kf.set_dynamics(A=A, Q=Q, f_k=f)
                (zs, Ps, Ms) = self.kf.smooth(zl[0], Pl[0], zn, Pn, self.kf.A, self.kf.f_k, self.kf.Q)
            else:
                zs = zl[j]
                Ps = Pl[j]
                Ms = numpy.zeros_like(Ps)
            res[j] = numpy.hstack((zs.ravel(), Ps.ravel(), Ms.ravel()))

        return res

    def fwd_peak_density(self, u, t):
        """
        No need for rejections sampling for this type of model, always returns
        0.0 since all particles are equivalent

        Args:
         - u: Unused
         - t: Unused

        Returns
         (float) 0.0
        """
        return 0.0

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
        (zl, Pl) = self.get_states(particles)
        lpz0 = numpy.empty(N)
        for i in range(N):
            l1 = self.calc_l1(zl[i], Pl[i], self.z0, self.P0)
            (_tmp, ld) = numpy.linalg.slogdet(self.P0)
            tmp = numpy.linalg.solve(self.P0, l1)
            lpz0[i] = -0.5 * (ld + numpy.trace(tmp))
        return lpz0

    def eval_logp_x0_val_grad(self, particles, t):
        """
        Evaluate gradient of sum log p(x_0)

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - t (float): time stamp
        """
        # Calculate l1 according to (19a)
        N = len(particles)
        lparam = len(self.params)
        lpz0_grad = numpy.zeros(lparam)
        (zl, Pl) = self.get_states(particles)
        (z0_grad, P0_grad) = self.get_initial_grad()
        if (z0_grad is None and P0_grad is None):
            lpz0 = self.eval_logp_x0(particles, t)
        else:
            lpz0 = 0.0
            P0cho = scipy.linalg.cho_factor(self.P0)
            ld = numpy.sum(numpy.log(numpy.diagonal(P0cho[0]))) * 2
            for i in range(N):
                (l1, l1_grad) = self.calc_l1_grad(zl[i], Pl[i], self.z0, self.P0, z0_grad)
                tmp = scipy.linalg.cho_solve(P0cho, l1)
                lpz0 += -0.5 * (ld + numpy.trace(tmp))
                for j in range(len(self.params)):
                    lpz0_grad[j] -= 0.5 * mlnlg_compute.compute_logprod_derivative(P0cho, P0_grad[j], l1, l1_grad[j])
        return (lpz0, lpz0_grad)

    def eval_logp_xnext(self, particles, x_next, u, t):
        """
        Evaluate log p(x_{t+1}|x_t)

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - x_next (array-like): future states
         - t (float): time stamp

        Returns: (array-like)
        """
        # Calculate l2 according to (16)
        N = len(particles)
        (zl, Pl) = self.get_states(particles)
        (zn, Pn) = self.get_states(x_next)
        (A, f, Q) = self.get_pred_dynamics(u=u, t=t)
        self.kf.set_dynamics(A=A, Q=Q, f_k=f)
        self.t = t
        lpxn = numpy.empty(N)

        for k in range(N):
            lz = len(self.z0)
            lzP = lz + lz * lz
            Mz = particles[k][lzP:].reshape((lz, lz))
            (l2, _A, _M_ext, _predict_err) = self.calc_l2(zn[k], Pn[k], zl[k], Pl[k], self.kf.A, self.kf.f_k, Mz)
            (_tmp, ld) = numpy.linalg.slogdet(self.kf.Q)
            tmp = numpy.linalg.solve(self.kf.Q, l2)
            lpxn[k] = -0.5 * (ld + numpy.trace(tmp))

        return lpxn

    def eval_logp_xnext_val_grad(self, particles, x_next, u, t):
        """
        Evaluate value and gradient of log p(x_{t+1}|x_t)

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - x_next (array-like): future states
         - t (float): time stamp

        Returns: ((array-like), (array-like))
        """
        # Calculate l2 according to (16)
        N = len(particles)
        lparam = len(self.params)
        (zl, Pl) = self.get_states(particles)
        (zn, Pn) = self.get_states(x_next)
        (A, f, Q) = self.get_pred_dynamics(u=u, t=t)
        (A_grad, f_grad, Q_grad) = self.get_pred_dynamics_grad(u=u, t=t)
        lpxn_grad = numpy.zeros(lparam)
        if (A_grad is None and f_grad is None and Q_grad is None):
            lpxn = self.eval_logp_xnext(particles, x_next, u, t)
        else:
            self.kf.set_dynamics(A=A, Q=Q, f_k=f)
            lpxn = 0.0
            Qcho = scipy.linalg.cho_factor(self.kf.Q, check_finite=False)
            ld = numpy.sum(numpy.log(numpy.diagonal(Qcho[0]))) * 2

            if (Q_grad is None):
                Q_grad = numpy.zeros(
                    (len(self.params), self.kf.lz, self.kf.lz))

            for k in range(N):
                lz = len(self.z0)
                lzP = lz + lz * lz
                Mz = particles[k][lzP:].reshape((lz, lz))
                (l2, l2_grad) = self.calc_l2_grad(zn[k], Pn[k], zl[k], Pl[k], self.kf.A, self.kf.f_k, Mz, A_grad, f_grad)
                tmp = scipy.linalg.cho_solve(Qcho, l2)
                lpxn += -0.5 * (ld + numpy.trace(tmp))

                for j in range(len(self.params)):
                    lpxn_grad[j] -= 0.5 * mlnlg_compute.compute_logprod_derivative(Qcho, Q_grad[j], l2, l2_grad[j])

        return (lpxn, lpxn_grad)

    def eval_logp_y(self, particles, y, t):
        """
        Evaluate value of log p(y_t|x_t)

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - y (array-like): measurement
         - t (float): time stamp

        Returns: (array-like)
        """
        N = len(particles)
        self.t = t
        (y, C, h, R) = self.get_meas_dynamics(y=y, t=t)
        self.kf.set_dynamics(C=C, R=R, h_k=h)
        (zl, Pl) = self.get_states(particles)
        logpy = numpy.empty(N)
        for i in range(N):
            # Calculate l3 according to (19b)
            l3 = self.calc_l3(y, zl[i], Pl[i])
            (_tmp, ld) = numpy.linalg.slogdet(self.kf.R)
            tmp = numpy.linalg.solve(self.kf.R, l3)
            logpy[i] = -0.5 * (ld + numpy.trace(tmp))

        return logpy

    def eval_logp_y_val_grad(self, particles, y, t):
        """
        Evaluate value and gradient of log p(y_t|x_t)

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - y (array-like): measurement
         - t (float): time stamp

        Returns: ((array-like), (array-like))
        """
        N = len(particles)
        lparam = len(self.params)
        (y, C, h, R) = self.get_meas_dynamics(y=y, t=t)
        (C_grad, h_grad, R_grad) = self.get_meas_dynamics_grad(y=y, t=t)
        logpy_grad = numpy.zeros(lparam)
        if (C_grad is None and h_grad is None and R_grad is None):
            logpy = self.eval_logp_y(particles, y, t)
        else:

            self.kf.set_dynamics(C=C, R=R, h_k=h)
            Rcho = scipy.linalg.cho_factor(self.kf.R, check_finite=False)
            ld = numpy.sum(numpy.log(numpy.diagonal(Rcho[0]))) * 2
            (zl, Pl) = self.get_states(particles)
            logpy = 0.0

            if (R_grad is None):
                R_grad = numpy.zeros((len(self.params), len(y), len(y)))

            for i in range(N):
                # Calculate l3 according to (19b)
                (l3, l3_grad) = self.calc_l3_grad(y, zl[i], Pl[i])
                tmp = scipy.linalg.cho_solve(Rcho, l3)
                logpy += -0.5 * (ld + numpy.trace(tmp))

                for j in range(len(self.params)):
                    logpy_grad[j] -= 0.5 * mlnlg_compute.compute_logprod_derivative(
                        Rcho, R_grad[j], l3, l3_grad[j])

        return (logpy, logpy_grad)

    def get_pred_dynamics_grad(self, u, t):
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

    def get_meas_dynamics_grad(self, y, t):
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

    def get_initial_grad(self):
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
        lparam = len(self.params)
        return (numpy.zeros((lparam, self.kf.lz, 1)),
                numpy.zeros((lparam, self.kf.lz, self.kf.lz)))

    def calc_l1(self, z, P, z0, P0):
        """ internal helper function """
        z0_diff = z - z0
        l1 = z0_diff.dot(z0_diff.T) + P
        return l1

    def calc_l1_grad(self, z, P, z0, P0, z0_grad):
        """ internal helper function """
        lparams = len(self.params)
        z0_diff = z - z0
        l1 = z0_diff.dot(z0_diff.T) + P
        l1_diff = numpy.zeros((lparams, self.kf.lz, self.kf.lz))
        if (z0_grad is not None):
            for j in range(lparams):
                tmp = -z0_grad[j].dot(z0_diff.T)
                l1_diff[j] += tmp + tmp.T
        return (l1, l1_diff)

    def calc_l2(self, zn, Pn, z, P, A, f, M):
        """ internal helper function """
        predict_err = zn - f - A.dot(z)
        AM = A.dot(M)
        l2 = predict_err.dot(predict_err.T)
        l2 += Pn + A.dot(P).dot(A.T) - AM.T - AM
        return (l2, A, M, predict_err)

    def calc_l2_grad(self, zn, Pn, z, P, A, f, M, A_grad, f_grad):
        """ internal helper function """
        lparam = len(self.params)
        predict_err = zn - f - A.dot(z)
        AM = A.dot(M)
        l2 = predict_err.dot(predict_err.T)
        l2 += Pn + A.dot(P).dot(A.T) - AM.T - AM
        l2_grad = numpy.zeros((lparam, self.kf.lz, self.kf.lz))
        if (f_grad is not None):
            for j in range(lparam):
                tmp = -f_grad[j].dot(predict_err.T)
                l2_grad[j] += tmp + tmp.T
        if (A_grad is not None):
            for j in range(lparam):
                tmp = -A_grad[j].dot(z).dot(predict_err.T)
                l2_grad[j] += tmp + tmp.T
                tmp = A_grad[j].dot(P).dot(A.T)
                l2_grad[j] += tmp + tmp.T
                tmp = -A_grad[j].dot(M)
                l2_grad[j] += tmp + tmp.T
        return (l2, l2_grad)

    def calc_l3(self, y, z, P):
        """ internal helper function """
        meas_diff = self.kf.measurement_diff(y.reshape((-1, 1)),
                                             z,
                                             C=self.kf.C,
                                             h_k=self.kf.h_k)
        l3 = meas_diff.dot(meas_diff.T)
        l3 += self.kf.C.dot(P).dot(self.kf.C.T)
        return l3

    def calc_l3_grad(self, y, z, P, C_grad, h_grad):
        """ internal helper function """
        lparam = len(self.params)
        meas_diff = self.kf.measurement_diff(y.reshape((-1, 1)),
                                             z,
                                             C=self.kf.C,
                                             h_k=self.kf.h_k)
        l3 = meas_diff.dot(meas_diff.T)
        l3 += self.kf.C.dot(P).dot(self.kf.C.T)
        l3_grad = numpy.zeros((lparam, len(y), len(y)))
        if (h_grad is not None):
            for j in range(lparam):
                tmp = -h_grad[j].dot(meas_diff)
                l3_grad[j] += tmp + tmp.T
        if (C_grad is not None):
            for j in range(lparam):
                tmp = -C_grad[j].dot(z).dot(meas_diff)
                l3_grad[j] += tmp + tmp.T
                tmp = C_grad[j].dot(P).dot(self.kf.C)
                l3_grad[j] += tmp + tmp.T
        return (l3, l3_grad)
