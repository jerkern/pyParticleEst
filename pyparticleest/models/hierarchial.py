""" Collection of functions and classes used for Particle Filtering/Smoothing """
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
        self.len_xi = len_xi
        super(HierarchicalBase, self).__init__(lz=len_z, **kwargs)
        # Sore z0, P0 needed for default implementation of 
        # get_z0_initial and get_grad_z0_initial

    def update(self, particles, u, t, noise):
        """ Update estimate using noise as input """
        xin = self.calc_xi_next(particles, u, t, noise)
        # Update linear estimate with data from measurement of next non-linear
        # state 
        (_xil, zl, Pl) = self.get_states(particles)
        (Az, fz, Qz, _, _, _) = self.get_lin_pred_dynamics_int(particles, u, t)
        for i in xrange(len(zl)):
            # Predict z_{t+1}
            (zl[i], Pl[i]) = self.kf.predict_full(zl[i], Pl[i], Az[i], fz[i], Qz[i])
        
        # Predict next states conditioned on eta_next
        self.set_states(particles, xin, zl, Pl)
    
    def measure(self, particles, y, t):
        """ Return the log-pdf value of the measurement """
        
        lyxi = self.measure_nonlin(particles, y, t)
        (xil, zl, Pl) = self.get_states(particles)
        N = len(particles)
        (y, Cz, hz, Rz) = self.get_lin_meas_dynamics(particles, y, t)
        if (Cz == None):
            Cz=numpy.repeat(self.kf.C[numpy.newaxis,:,:], N, axis=0)
            #Cz=N*(self.kf.C,)
        if (hz == None):
            hz=numpy.repeat(self.kf.h_k[numpy.newaxis,:,:], N, axis=0)
            #hz=N*(self.kf.h_k,)
        if (Rz == None):
            Rz=numpy.repeat(self.kf.R[numpy.newaxis,:,:], N, axis=0)
            #Rz=N*(self.kf.R,)
            
        lyz = numpy.empty_like(lyxi)
        for i in xrange(len(zl)):
            # Predict z_{t+1}
            lyz[i] = self.kf.measure_full(numpy.asarray(y).reshape((-1,1)),
                                          zl[i], Pl[i], Cz[i], hz[i], Rz[i])
        
        self.set_states(particles, xil, zl, Pl)
        return lyxi + lyz
    
    def logp_xnext(self, particles, next_part, u, t):
        """ Return the log-pdf value for the possible future state 'next' given input u """
        N = len(particles)
        Nn = len(next_part)
        if (N > 1 and Nn == 1):
            next_part = numpy.repeat(next_part, N, 0)
        
        #lpxi = numpy.empty(N)
        lpz = numpy.empty(N)

        (Az, fz, Qz, _, _, _) = self.get_lin_pred_dynamics_int(particles, u, t)
        (_xil, zl, Pl) = self.get_states(particles)
        zln = numpy.empty_like(zl)
        Pln = numpy.empty_like(Pl)
        
        lpxi = self.logp_xnext_xi(particles, next_part[:,:self.len_xi], u, t).ravel()
        
        for i in xrange(N):
            
            # Predict z_{t+1}
            (zln[i], Pln[i]) = self.kf.predict_full(zl[i], Pl[i], Az[i], fz[i], Qz[i])
            
            lpz[i] = kalman.lognormpdf(next_part[i][self.len_xi:].reshape((-1,1))-zln[i], Pln[i])
            
        #mul = numpy.repeat(next_part[1].reshape((-1,1)),N,axis=0)
        #mul = N*(next_part[:self.len_xi].reshape((-1,1)),)
        #lpz = kalman.lognormpdf_vec(zln, mul, Pln)
        #lpz = kalman.lognormpdf_jit(zl, mul, Pl)
        return lpxi + lpz
    
    def sample_smooth(self, particles, next_part, u, y, t):
        """ Update ev. Rao-Blackwellized states conditioned on "next_part" """
        M = len(particles)
        res = numpy.empty((M,self.len_xi+self.kf.lz,1))
        for j in range(M):
            part = numpy.copy(particles[j])
            (xil, zl, Pl) = self.get_states([part,])
            if (next_part != None):
                (A, f, Q, _, _, _) = self.get_lin_pred_dynamics_int([part,], u, t)
                self.kf.measure_full(next_part[j][self.len_xi:], zl[0], Pl[0],
                                     C=A[0], h_k=f[0], R=Q[0])

            xi = copy.copy(xil[0]).reshape((1,-1,1))
            z = numpy.random.multivariate_normal(zl[0].ravel(), Pl[0]).reshape((1,-1,1))
            res[j] = numpy.hstack((xi, z))
        return res

    @abc.abstractmethod
    def logp_xnext_xi(self, particles, next_xi, u, t):
        pass
    
    @abc.abstractmethod
    def calc_xi_next(self, particles, u, t, noise):
        pass
    
    @abc.abstractmethod
    def measure_nonlin(self, particles, y, t):
        pass
    
    @abc.abstractmethod
    def set_states(self, particles, xi_list, z_list, P_list):
        """ Set the estimate of the Rao-Blackwellized states """
        pass
 
    @abc.abstractmethod
    def get_states(self, particles):
        """ Return the estimate of the Rao-Blackwellized states.
            Must return thrre variables, the first a list containing all the xi,
            the second a list of all the expected values, 
            the third a list of the corresponding covariance matrices"""
        pass
    
    def post_smoothing(self, st):
        """ Kalman smoothing of the linear states conditioned on the non-linear
            trajetory """
        
        T = st.traj.shape[0]
        M = st.traj.shape[1]
        st.Mz = numpy.empty((T-1,M), dtype=numpy.ndarray)
        particles = self.create_initial_estimate(M)
        for j in xrange(M):
            (z0, P0) = self.get_rb_initial([st.traj[0][j][0],])
            self.set_states(particles[j:j+1], (st.traj[0][j][0],), (z0,), (P0,))
        
        T = len(st.traj)
        straj = numpy.empty((T, M), dtype=object)
        
        for i in xrange(T-1):
            if (st.y[i] != None):
                self.measure(particles, y=st.y[i], t=st.t[i])
            for j in xrange(M):
                (_xil, zl, Pl) = self.get_states(particles[j:j+1])
                self.set_states(particles[j:j+1], st.traj[i][j][0], zl, Pl)
            straj[i] = particles

            particles = copy.deepcopy(particles)
            (Az, fz, Qz, _, _, _) = self.get_lin_pred_dynamics_int(particles, u=st.u[i], t=st.t[i])
            (_xil, zl, Pl) = self.get_states(particles)
            for j in xrange(M):
                self.kf.predict_full(zl[j], Pl[j], Az[j], fz[j], Qz[j])
                self.set_states(particles[j:j+1], st.traj[i+1][j:j+1][0], zl[j:j+1], Pl[j:j+1])
            
        if (st.y[-1] != None):
            self.measure(particles, y=st.y[-1], t=st.t[-1])
        
        
        for j in xrange(M):
            (_xil, zl, Pl) = self.get_states(particles[j:j+1])
            self.set_states(particles[j:j+1], st.traj[-1][j][0], zl, Pl)
        straj[-1] = particles
        
        # Backward smoothing
        for i in reversed(xrange(T-1)):
            (_xin, zn, Pn) = self.get_states(straj[i+1])
            (xi, z, P) = self.get_states(straj[i])
            (Az, fz, Qz, _, _, _) = self.get_lin_pred_dynamics_int(straj[i], u=st.u[i], t=st.t[i])
            for j in xrange(M):
                (zs, Ps, Ms) = self.kf.smooth(z[j], P[j], zn[j], Pn[j], Az[j], fz[j], Qz[j])
                self.set_states(straj[i][j:j+1], xi[j], (zs,), (Ps,))
                st.Mz[i,j] = Ms
                
        return straj
    
class HierarchicalRSBase(HierarchicalBase,FFBSiRS):
    def __init__(self, **kwargs):
        super(HierarchicalRSBase, self).__init__(**kwargs)
        
    def logp_xnext_max(self, particles, u, t):
        N = len(particles)
        lpxi = self.logp_xnext_xi_max(particles, u, t)
        (Az, _fz, Qz, _, _, _) = self.get_lin_pred_dynamics_int(particles, u, t)
        lpz = numpy.empty_like(lpxi)
        (_xil, _zl, Pl) = self.get_states(particles)
        nx = len(Qz[0])
        for i in xrange(N):
            # Predict z_{t+1}
            Pn=Az[i].dot(Pl[i]).dot(Az[i].T) + Qz[i]
            lpz[i] = -0.5*nx*math.log(2*math.pi)+numpy.linalg.slogdet(Pn)[1]
        lpmax = numpy.max(lpxi+lpz)
        return lpmax
    
    @abc.abstractmethod
    def logp_xnext_xi_max(self, particles, u, t):
        pass
