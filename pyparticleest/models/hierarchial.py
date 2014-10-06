""" Model definition for base class for hierarchical systems 

@author: Jerker Nordh
"""
import abc
import pyparticleest.utils.kalman as kalman
from pyparticleest.interfaces import FFBSiRS
from pyparticleest.models.rbpf import RBPSBase
import numpy
import copy
import math

class HierarchicalBase(RBPSBase):
    """ Base class for Rao-Blackwellization of hierarchical models """
    __metaclass__ = abc.ABCMeta

    def __init__(self, len_xi, len_z, **kwargs):
        """ len_xi - number of nonlinear states
            len_z - number of linear states """
        self.lxi = len_xi
        super(HierarchicalBase, self).__init__(lz=len_z, **kwargs)

    def measure(self, particles, y, t):
        """ Return the log-pdf value of the measurement """

        lyxi = self.measure_nonlin(particles, y, t)
        (xil, zl, Pl) = self.get_states(particles)
        N = len(particles)
        (y, Cz, hz, Rz) = self.get_lin_meas_dynamics(particles, y, t)
        if (Cz == None):
            Cz = numpy.repeat(self.kf.C[numpy.newaxis, :, :], N, axis=0)
        if (hz == None):
            hz = numpy.repeat(self.kf.h_k[numpy.newaxis, :, :], N, axis=0)
        if (Rz == None):
            Rz = numpy.repeat(self.kf.R[numpy.newaxis, :, :], N, axis=0)

        lyz = numpy.empty_like(lyxi)
        for i in xrange(len(zl)):
            lyz[i] = self.kf.measure_full(numpy.asarray(y).reshape((-1, 1)),
                                          zl[i], Pl[i], Cz[i], hz[i], Rz[i])

        self.set_states(particles, xil, zl, Pl)
        return lyxi + lyz

    def calc_cond_dynamics(self, particles, xi_next, u, t):
        """ Calculates the linear dynamics for each particles """
        (Az, fz, Qz, _, _, _) = self.get_lin_pred_dynamics_int(particles, u, t)
        return (Az, fz, Qz)

    def meas_xi_next(self, particles, xi_next, u, t):
        # There is no information in the next nonlinear state about the
        # current linear states for this class of models
        return particles

    def logp_xnext(self, particles, next_part, u, t):
        """ Return the log-pdf value for the possible future state 'next' given input u """
        N = len(particles)
        Nn = len(next_part)
        if (N > 1 and Nn == 1):
            next_part = numpy.repeat(next_part, N, 0)

        lpz = numpy.empty(N)

        (Az, fz, Qz, _, _, _) = self.get_lin_pred_dynamics_int(particles, u, t)
        (_xil, zl, Pl) = self.get_states(particles)
        zln = numpy.empty_like(zl)
        Pln = numpy.empty_like(Pl)

        lpxi = self.logp_xnext_xi(particles, next_part[:, :self.lxi], u, t).ravel()

        for i in xrange(N):

            # Predict z_{t+1}
            (zln[i], Pln[i]) = self.kf.predict_full(zl[i], Pl[i], Az[i], fz[i], Qz[i])

            lpz[i] = kalman.lognormpdf(next_part[i][self.lxi:(self.lxi + self.kf.lz)].reshape((-1, 1)) - zln[i], Pln[i])

        return lpxi + lpz

    def sample_smooth(self, particles, future_trajs, ut, yt, tt):
        """ Update ev. Rao-Blackwellized states conditioned on "next_part" """
        M = len(particles)
        res = numpy.zeros((M, self.lxi + self.kf.lz + 2 * self.kf.lz ** 2))
        for j in range(M):
            part = numpy.copy(particles[j:j + 1])
            (xil, zl, Pl) = self.get_states(part,)
            if (future_trajs != None):
                (A, f, Q, _, _, _) = self.get_lin_pred_dynamics_int(part,
                                                                    ut[0],
                                                                    tt[0])
                # Measure the sampled next state,
                self.kf.measure_full(future_trajs[0, j, self.lxi:(self.lxi + self.kf.lz)].reshape((-1, 1)),
                                     zl[0], Pl[0], C=A[0], h_k=f[0], R=Q[0])

            xi = copy.copy(xil[0]).ravel()
            # Sample the linear variables, the full conditional density
            # is recovred later in the post_smoothing step
            z = numpy.random.multivariate_normal(zl[0].ravel(), Pl[0]).ravel()
            res[j, :(self.lxi + self.kf.lz)] = numpy.hstack((xi, z))
        return res

    @abc.abstractmethod
    def logp_xnext_xi(self, particles, next_xi, u, t):
        """ Evaluate the log-probability of the next nonlinear state """
        pass

    @abc.abstractmethod
    def calc_xi_next(self, particles, u, t, noise):
        """ Calculate the next nonlinear state given the input and noise
            realisation """
        pass

    @abc.abstractmethod
    def measure_nonlin(self, particles, y, t):
        """ Measurement probability for the nonlinear parts of the measurement
            equations """
        pass


class HierarchicalRSBase(HierarchicalBase, FFBSiRS):
    def __init__(self, **kwargs):
        super(HierarchicalRSBase, self).__init__(**kwargs)

    def logp_xnext_max(self, particles, u, t):
        """ Calculate maximum value of the logp_xnext function, used for
            rejection sampling """
        N = len(particles)
        lpxi = self.logp_xnext_xi_max(particles, u, t)
        (Az, _fz, Qz, _, _, _) = self.get_lin_pred_dynamics_int(particles, u, t)
        lpz = numpy.empty_like(lpxi)
        (_xil, _zl, Pl) = self.get_states(particles)
        nx = len(Qz[0])
        for i in xrange(N):
            # Predict z_{t+1}
            Pn = Az[i].dot(Pl[i]).dot(Az[i].T) + Qz[i]
            lpz[i] = -0.5 * nx * math.log(2 * math.pi) + numpy.linalg.slogdet(Pn)[1]
        lpmax = numpy.max(lpxi + lpz)
        return lpmax

    @abc.abstractmethod
    def logp_xnext_xi_max(self, particles, u, t):
        """ Maximum for nonlinear part of the logp_xnext, called from 
            logp_xnext_max"""
        pass
