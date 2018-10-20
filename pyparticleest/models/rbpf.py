""" Model definition for base class for Rao-Blackwellized models


@author: Jerker Nordh
"""

import abc
import pyparticleest.interfaces as interfaces
from pyparticleest.filter import ParticleApproximation, TrajectoryStep

try:
    import pyparticleest.utils.ckalman as kalman
except ImportError:
    print("Falling back to pure python implementaton, expect horrible performance")
    import pyparticleest.utils.kalman as kalman
import numpy

from builtins import range


class RBPFBase(interfaces.ParticleFiltering):
    """
    Base class for Rao-Blackwellized models

    Args:
     - lz (int): Dimension of linear subsystem
     - Az (array-like): Transition matrix for linear states (if constant)
     - fz (array-like): affine term for linear states (if constant)
     - Qz (array-like): Covariance of process noise for linear states
       (if constant)
     - C (array-like): Measurement dynamic for linear states (if constant)
     - hz (array-like): Affine measurement term for linear states (if constant)
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, lz, Az=None, fz=None, Qz=None,
                 C=None, hz=None, R=None, **kwargs):

        self.kf = kalman.KalmanSmoother(lz, A=Az, C=C,
                                        Q=Qz, R=R,
                                        f_k=fz, h_k=hz)
        super(RBPFBase, self).__init__(**kwargs)

    def set_dynamics(self, Az=None, C=None, Qz=None, R=None, fz=None, hz=None):
        """
        Change the dynamics for linear subsystem

        Args:
         - lz (int): Dimension of linear subsystem
         - Az (array-like): Transition matrix for linear states (if constant)
         - fz (array-like): affine term for linear states (if constant)
         - Qz (array-like): Covariance of process noise for linear states
           (if constant)
         - C (array-like): Measurement dynamic for linear states (if constant)
         - hz (array-like): Affine measurement term for linear states (if constant)
        """
        return self.kf.set_dynamics(Az, C, Qz, R, fz, hz)

    def get_nonlin_pred_dynamics(self, particles, u, t):
        """
        Return matrices describing affine relation of next
        nonlinear state conditioned on current nonlinear state

        xi_{t+1]} = A_xi(xi) * z_t + f_xi(xi) + v_xi, v_xi ~ N(0,Q_xi(xi))

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - u (array-like): input signal
         - t (float): time stamp

        Returns:
         (A_xi, f_xi, Q_xi) where each element is a list
         with the corresponding matrix for each particle. None indicates
         that the matrix is identical for all particles and the value stored
         in this class should be used instead
        """
        return (None, None, None)

    def get_nonlin_pred_dynamics_int(self, particles, u, t):
        """
        Helper class for calculating dynamics for nonlinear state

        xi_{t+1]} = A_xi(xi) * z_t + f_xi(xi) + v_xi, v_xi ~ N(0,Q_xi(xi))

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - u (array-like): input signal
         - t (float): time stamp

        Returns:
         (A_xi, f_xi, Q_xi, Axi_identical, fxi_identical, Qxi_identical)
         where the first three element is are lists with the corresponding
         matrix for each particle. The last three are boolean to indicate if
         all the matrices are identical to allow for more efficient computions
        """
        (Axi, fxi, Qxi) = self.get_nonlin_pred_dynamics(particles, u=u, t=t)
        N = len(particles)
        Axi_identical = False
        fxi_identical = False
        Qxi_identical = False
        # This is probably not so nice performance-wise, but will
        # work initially to profile where the bottlenecks are.
        if (Axi is None):
            Axi = N * (self.Axi,)
            Axi_identical = True
        if (fxi is None):
            fxi = N * (self.fxi,)
            fxi_identical = True
        if (Qxi is None):
            Qxi = N * (self.Qxi,)
            Qxi_identical = True
        return (Axi, fxi, Qxi, Axi_identical, fxi_identical, Qxi_identical)

    def get_lin_pred_dynamics(self, particles, u, t):
        """
        Return matrices describing affine relation of next
        nonlinear state conditioned on current nonlinear state

        \z_{t+1]} = A_z(xi) * z_t + f_z(xi) + v_z, v_z ~ N(0,Q_z(xi))

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - u (array-like): input signal
         - t (float): time stamp

        Returns:
         (A_z, f_z, Q_z) where each element is a list
         with the corresponding matrix for each particle. None indicates
         that the matrix is identical for all particles and the value stored
         in this class should be used instead
        """
        return (None, None, None)

    def get_lin_pred_dynamics_int(self, particles, u, t):
        """
        Helper class for calculating dynamics for linear state

        \z_{t+1]} = A_z(xi) * z_t + f_z(xi) + v_z, v_z ~ N(0,Q_z(xi))

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - u (array-like): input signal
         - t (float): time stamp

        Returns:
         (A_z, f_z, Q_z, Az_identical, fz_identical, Qz_identical)
         where the first three element is are lists with the corresponding
         matrix for each particle. The last three are boolean to indicate if
         all the matrices are identical to allow for more efficient computations
        """
        N = len(particles)
        (Az, fz, Qz) = self.get_lin_pred_dynamics(particles, u=u, t=t)
        Az_identical = False
        fz_identical = False
        Qz_identical = False
        if (Az is None):
            # Az=numpy.repeat(self.kf.A[numpy.newaxis,:,:], N, axis=0)
            Az = N * (self.kf.A,)
            Az_identical = True
        if (fz is None):
            # fz=numpy.repeat(self.kf.f_k[numpy.newaxis,:,:], N, axis=0)
            fz = N * (self.kf.f_k,)
            fz_identical = True
        if (Qz is None):
            # Qz=numpy.repeat(self.kf.Q[numpy.newaxis,:,:], N, axis=0)
            Qz = N * (self.kf.Q,)
            Qz_identical = True

        return (Az, fz, Qz, Az_identical, fz_identical, Qz_identical)

    def get_meas_dynamics(self, particles, y, t):
        """
        Return matrices describing affine relation of measurement and current
        state estimates

        \y_t+1 = C(xi) * z_t + h_z(xi) + e_z, e_z ~ N(0,R(xi))

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - y (array-like): measurement
         - t (float): time stamp

        Returns:
         (y, C_z, h_z, R_z): y is a preprocessed measurement, the rest are lists
         with the corresponding matrix for each particle. None indicates
         that the matrix is identical for all particles and the value stored
         in this class should be used instead
        """
        return (y, None, None, None)

    def get_meas_dynamics_int(self, particles, y, t):
        """
        Helper class for calculating measurement dynamics

        \y_t+1 = C(xi) * z_t + h_z(xi) + e_z, e_z ~ N(0,R(xi))

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - y (array-like): measurement
         - t (float): time stamp

        Returns:
         (y, C, h_z, R_z, C_identical, hz_identical, Rz_identical)
         y is a preprocessed measurement, the next three element are lists with
         the corresponding matrix for each particle. The last three are boolean
         to indicate if all the matrices are identical to allow for more
         efficient computations
        """
        N = len(particles)
        (y, Cz, hz, Rz) = self.get_meas_dynamics(particles=particles, y=y, t=t)
        Cz_identical = False
        hz_identical = False
        Rz_identical = False
        if (Cz is None):
            if (self.kf.C is None and hz is not None):
                Cz = N * (numpy.zeros((len(hz[0]), self.kf.lz)),)
            else:
                Cz = N * (self.kf.C,)
            # Cz=N*(self.kf.C,)
            Cz_identical = True
        if (hz is None):
            hz = N * (self.kf.h_k,)
            hz_identical = True
        if (Rz is None):
            Rz = N * (self.kf.R,)
            Rz_identical = True
        return (y, Cz, hz, Rz, Cz_identical, hz_identical, Rz_identical)

# This is not implemented
#    def get_condlin_meas_dynamics(self, y, xi_next, particles):
#        return (y, None, None, None)

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

        # Calc (xi_{t+1} | xi_t, z_t, y_t)
        xin = self.calc_xi_next(particles=particles, u=u, t=t, noise=noise)
        # Calc (z_{t+1} | xi_{t+1}, y_t)
        self.cond_predict(particles=particles, xi_next=xin, u=u, t=t)
        return particles

    def cond_predict_single_step(self, part, past_trajs, pind, future_parts, find, ut, yt, tt, cur_ind):
        """
        Calculate estimates of the next time step using particle 'part', conditioned
        on the non-linear parts of the first step of the future trajectory.

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
        xin = future_parts[find, :self.lxi]
        particles = numpy.copy(part)
        self.cond_predict(particles=particles, xi_next=xin, u=ut[cur_ind], t=tt[cur_ind])
        return particles

    def cond_predict(self, particles, xi_next, u, t):
        """
        Calculate estimate of z_{t+1} given information of xi_{t+1}

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - xi_next (array-like): next nonlinear state
         - u (array-like): input signal
         - t (float): time stamp
        """
        # Calc (z_t | xi_{t+1}, y_t)
        self.meas_xi_next(particles=particles, xi_next=xi_next, u=u, t=t)
        # Compensate for noise correlation
        (Az, fz, Qz) = self.calc_cond_dynamics(particles=particles, xi_next=xi_next, u=u, t=t)
        (_, zl, Pl) = self.get_states(particles)
        # Predict next states conditioned on xi_next
        for i in range(len(zl)):
            # Predict z_{t+1}
            (zl[i], Pl[i]) = self.kf.predict_full(z=zl[i], P=Pl[i], A=Az[i], f_k=fz[i], Q=Qz[i])

        self.set_states(particles, xi_next, zl, Pl)


class RBPSBase(RBPFBase, interfaces.FFBSiRS):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_rb_initial(self, xi_initial):
        """
        Calculate estimate of initial state for linear state condition on the
        nonlinear estimate

        Args:
         - xi_initial (array-like): Initial xi states

        Returns:
         (z,P): z is a list of all inital mean values, P is a list of covariance
         matrices
        """
        pass

    def post_smoothing(self, st):
        """
        Kalman smoothing of the linear states conditioned on the non-linear
        trajetory

        Args:
         - st (SmoothTrajectory): Smoothed estimate (with post processing step)

        Returns:
         (array-like): Smoothed estimate with sufficient statistics for linear
         states
        """
        T = len(st.traj)
        M = len(st.traj[0].pa.part)

        lx_filt = self.lxi + self.kf.lz + self.kf.lz ** 2
        # Allocate extra space for Mz
        lx = lx_filt + self.kf.lz ** 2

        straj = numpy.empty((T,), dtype=object)

        # Forward filtering
        ftraj = self.pre_mhips_pass(st)

        particles = numpy.zeros((M, lx))
        particles[:, :lx_filt] = ftraj[-1].pa.part

        straj[-1] = TrajectoryStep(ParticleApproximation(particles),
                                   ftraj[-1].ancestors)

        # Backward smoothing
        for i in reversed(range(T - 1)):
            (xin, zn, Pn) = self.get_states(straj[i + 1].pa.part)
            particles = numpy.zeros((M, lx))
            particles[:, :lx_filt] = ftraj[i].pa.part
            straj[i] = TrajectoryStep(ParticleApproximation(particles),
                                      ftraj[i].ancestors)

            # Condition on future nonlinear state
            self.meas_xi_next(particles, xin, u=st.u[i], t=st.t[i])
            (xi, z, P) = self.get_states(particles)
            (Al, fl, Ql) = self.calc_cond_dynamics(
                particles, xin, u=st.u[i], t=st.t[i])
            # Update distribution for linear states
            for j in range(M):
                (zs, Ps, Ms) = self.kf.smooth(z[j], P[j], zn[j], Pn[j],
                                              Al[j], fl[j], Ql[j])
                self.set_states(straj[i].pa.part[j:j + 1, :],
                                xi[j], zs[numpy.newaxis], Ps[numpy.newaxis])
                self.set_Mz(straj[i].pa.part[j:j + 1, :], Ms[numpy.newaxis])

        return straj

    def pre_mhips_pass(self, st):
        """
        Calculated sufficient statistics for the filtering problem.
        Used to make sure all particles are in the expected state when using
        MHIPS/MHBP

        Args:
         - st (SmoothTrajectory): Smoothed estimate

        Returns:
         (array-like): Filtered estimate with sufficient statistics for linear
         states
        """
        T = len(st.traj)
        M = len(st.traj[0].pa.part)

        lx = self.lxi + self.kf.lz + self.kf.lz ** 2

        straj = numpy.empty((T,), dtype=object)

        particles = numpy.empty((M, lx))
        #(xil, _zl, _Pl) = self.get_states(particles)
        xil = st.traj[0].pa.part[:, :self.lxi].reshape((M, self.lxi, 1))

        (z0, P0) = self.get_rb_initial(xil)
        self.set_states(particles, xil, z0, P0)

        for i in range(T - 1):
            if (st.y[i] is not None):
                self.measure(particles, y=st.y[i], t=st.t[i])

            straj[i] = TrajectoryStep(ParticleApproximation(particles),
                                      st.traj[i].ancestors)

            #(xin, _zn, _Pn) = self.get_states(st.traj[i + 1])
            xin = st.traj[i + 1].pa.part[:, :self.lxi].reshape((M, self.lxi, 1))
            self.cond_predict(particles, xin, u=st.u[i], t=st.t[i])

        if (st.y[-1] is not None):
            self.measure(particles, y=st.y[-1], t=st.t[-1])

        straj[-1] = TrajectoryStep(ParticleApproximation(particles),
                                   st.traj[-1].ancestors)
        return straj

    def set_states(self, particles, xi_list, z_list, P_list):
        """
        Set the estimate of the states states

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - xi_list (list): list of xi values for each particle
         - z_list (list): list of mean values for z for each particle
         - P_list (list): list of covariance matrices for z for each particle
        """
        N = len(particles)
        zend = self.lxi + self.kf.lz
        Pend = zend + self.kf.lz ** 2

        particles[:, :self.lxi] = xi_list.reshape((N, self.lxi))
        particles[:, self.lxi:zend] = z_list.reshape((N, self.kf.lz))
        particles[:, zend:Pend] = P_list.reshape((N, self.kf.lz ** 2))

    def get_states(self, particles):
        """
        Return the estimates contained in the particles array

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)

        Returns
            (xil, zl, Pl):
             - xil: list of xi values
             - zl: list of mean values for z
             - Pl: list of covariance matrices for z
        """
        N = len(particles)
        zend = self.lxi + self.kf.lz
        Pend = zend + self.kf.lz ** 2

        xil = particles[:, :self.lxi, numpy.newaxis]
        zl = particles[:, self.lxi:zend, numpy.newaxis]
        Pl = particles[:, zend:Pend].reshape((N, self.kf.lz, self.kf.lz))

        return (xil, zl, Pl)

    def get_Mz(self, smooth_particles):
        """
        Return the cross covariance of z_t and z_t+1 at time t

        Args:
         - smooth_particles (array-like): smoothed particle estimates

        Returns
         (arrau-like): Array of covariance matrices, first dimenson indexs the
         particles
        """
        N = len(smooth_particles)
        zend = self.lxi + self.kf.lz
        Pend = zend + self.kf.lz ** 2
        Mend = Pend + self.kf.lz ** 2

        Mz = smooth_particles[:, Pend:Mend].reshape((N, self.kf.lz, self.kf.lz))
        return Mz

    def set_Mz(self, smooth_particles, Mz):
        """
        Set the cross covariance estimate for z_t and z_t+1 at time t

        Args:
         - smooth_particles (array-like): smoothed particle estimates
         - Mz (array-like): Array of covariance matrices, first dimenson indexs the
           particles
        """
        N = len(smooth_particles)
        zend = self.lxi + self.kf.lz
        Pend = zend + self.kf.lz ** 2
        Mend = Pend + self.kf.lz ** 2

        smooth_particles[:, Pend:Mend] = Mz.reshape((N, self.kf.lz ** 2))
