""" Collection of functions and classes used for Particle Filtering/Smoothing """
import abc
import kalman
import numpy
import copy
import math
# This was slower than kalman.lognormpdf
#from scipy.stats import multivariate_normal

class ParticleFilteringInterface(object):
    """ Base class for particles to be used with particle filtering """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def create_initial_estimate(self, N):
        """ Sample N particle from initial distribution """
        return
     
    @abc.abstractmethod
    def sample_process_noise(self, particles, u):
        """ Return process noise for input u """
        return
    
    @abc.abstractmethod
    def update(self, particles, u, noise):
        """ Update estimate using 'data' as input """
        return
    
    @abc.abstractmethod    
    def measure(self, particles, y):
        """ Return the log-pdf value of the measurement """
        return
    
    def copy_ind(self, particles, new_ind):
        N = len(new_ind)
        new_part = numpy.empty(N, type(particles[0]))
        for k in range(numpy.shape(new_ind)[0]):
            new_part[k] = copy.copy(particles[new_ind[k]])
        return new_part
    
class FFBSiInterface(ParticleFilteringInterface):
    """ Base class for particles to be used with particle smoothing """
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def next_pdf(self, particles, next_cpart, u=None):
        """ Return the log-pdf value for the possible future state 'next' given input u """
        pass
    
    @abc.abstractmethod
    def sample_smooth(self, particles, next_part, u=None):
        """ Update ev. Rao-Blackwellized states conditioned on "next_part" """
        pass

class FFBSiRSInterface(FFBSiInterface):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def next_pdf_max(self, particles, u=None):
        """ Return the log-pdf value for the possible future state 'next' given input u """
        pass
    
class RBPFBase(ParticleFilteringInterface):
    """ Base class for Rao-Blackwellized particles """
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, lz, Az=None, fz=None, Qz=None,
                 C=None ,hz=None, R=None, t0=0):
        
        self.kf = kalman.KalmanSmoother(lz, A=Az, C=C, 
                                        Q=Qz, R=R,
                                        f_k=fz, h_k=hz)
        
        self.t = t0
        
    def set_dynamics(self, Az=None, C=None, Qz=None, R=None, fz=None, hz=None):
        return self.kf.set_dynamics(Az, C, Qz, R, fz, hz)
    
    def get_nonlin_pred_dynamics(self, particles, u):
        """ Return matrices describing affine relation of next
            nonlinear state conditioned on current linear state
            
            xi_{t+1]} = A_xi * z_t + f_xi + v_xi, v_xi ~ N(0,Q_xi)
            
            Return (A_xi, f_xi, Q_xi) where each element is a list
            with the corresponding matrix for each particle. None indicates
            that the matrix is identical for all particles and the value stored
            in this class should be used instead
            """
        return (None, None, None)
    
    def get_nonlin_pred_dynamics_int(self, particles, u):
        (Axi, fxi, Qxi) = self.get_nonlin_pred_dynamics(particles, u)
        N = len(particles)
        Axi_identical = False
        fxi_identical = False
        Qxi_identical = False
        # This is probably not so nice performance-wise, but will
        # work initially to profile where the bottlenecks are.
        if (Axi == None):
            Axi=N*(self.Axi,)
            Axi_identical = True
        if (fxi == None):
            fxi=N*(self.fxi,)
            fxi_identical = True
        if (Qxi == None):
            Qxi= N*(self.Qxi,)
            Qxi_identical = True
        return (Axi, fxi, Qxi, Axi_identical, fxi_identical, Qxi_identical)
    
    def get_condlin_pred_dynamics(self, u, xi_next, particles):
        """ Return matrices describing affine relation of next
            nonlinear state conditioned on current linear state
            
            z_{t+1]} = A_z * z_t + f_z + v_z, v_z ~ N(0,Q_z)
            
            conditioned on the value of xi_{t+1}. 
            (Not the same as the dynamics unconditioned on xi_{t+1})
            when for example there is a noise correlation between the 
            linear and nonlinear state dynamics) 
            """
        return (None, None, None)
    
    def get_lin_pred_dynamics(self, particles, u):
        """ Return matrices describing affine relation of next
            nonlinear state conditioned on current linear state
            
            \z_{t+1]} = A_z * z_t + f_z + v_z, v_z ~ N(0,Q_z)
            
            conditioned on the value of xi_{t+1}. 
            (Not the same as the dynamics unconditioned on xi_{t+1})
            when for example there is a noise correlation between the 
            linear and nonlinear state dynamics) 
            """
        return (None, None, None)
    
    def get_lin_pred_dynamics_int(self, particles, u):
        N = len(particles)
        (Az, fz, Qz) = self.get_lin_pred_dynamics(particles, u)
        Az_identical = False
        fz_identical = False
        Qz_identical = False
        if (Az == None):
            #Az=numpy.repeat(self.kf.A[numpy.newaxis,:,:], N, axis=0)
            Az=N*(self.kf.A,)
            Az_identical = True
        if (fz == None):
            #fz=numpy.repeat(self.kf.f_k[numpy.newaxis,:,:], N, axis=0)
            fz=N*(self.kf.f_k,)
            fz_identical = True
        if (Qz == None):
            #Qz=numpy.repeat(self.kf.Q[numpy.newaxis,:,:], N, axis=0)
            Qz=N*(self.kf.Q,)
            Qz_identical = True
            
        return (Az, fz, Qz, Az_identical, fz_identical, Qz_identical)
    
    def get_meas_dynamics(self, particles, y):
        return (y, None, None, None)
    
    def get_meas_dynamics_int(self, particles, y):
        N=len(particles)
        (y, Cz, hz, Rz) = self.get_meas_dynamics(particles=particles, y=y)
        Cz_identical = False
        hz_identical = False
        Rz_identical = False
        if (Cz == None):
#            if (self.kf.C == None):
#                Cz=N*(numpy.zeros((len(y), self.kf.lz)))
#            else:
#                Cz=N*(self.kf.C,)
            Cz=N*(self.kf.C,)
            Cz_identical = True
        if (hz == None):
            hz=N*(self.kf.h_k,)
            hz_identical = True
        if (Rz == None):
            Rz=N*(self.kf.R,)
            Rz_identical = True
        return (y, Cz, hz, Rz, Cz_identical, hz_identical, Rz_identical)
    
# This is not implemented  
#    def get_condlin_meas_dynamics(self, y, xi_next, particles):
#        return (y, None, None, None)
    
    def update(self, particles, u, noise):
        """ Update estimate using noise as input """
        # Calc (xi_{t+1} | xi_t, z_t, y_t)
        xin = self.calc_xi_next(particles=particles, noise=noise, u=u)
        # Calc (z_t | xi_{t+1}, y_t)
        self.meas_xi_next(particles=particles, xi_next=xin, u=u)
        # Calc (z_{t+1} | xi_{t+1}, y_t)
        self.cond_predict(particles=particles, xi_next=xin, u=u)
        
        (_xil, zl, Pl) = self.get_states(particles)
        self.set_states(particles, xin, zl, Pl)
        self.t = self.t + 1.0


    
class RBPSBase(RBPFBase, FFBSiInterface):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def get_rb_initial(self, xi_initial):
        pass
    
    def post_smoothing(self, st):
        """ Kalman smoothing of the linear states conditioned on the non-linear
            trajetory """
        T = st.traj.shape[0]
        M = st.traj.shape[1]

        particles = numpy.copy(st.traj[0])
        (xil, _zl, _Pl) = self.get_states(particles)
        (z0, P0) = self.get_rb_initial(xil)
        self.set_states(particles, xil, z0, P0)

        T = len(st.traj)
        
        for i in xrange(T-1):
            self.t = st.t[i]
            if (st.y[i] != None):
                self.measure(particles, st.y[i])
            (xin, _zn, _Pn) = self.get_states(st.traj[i+1])
            self.meas_xi_next(particles, xin, st.u[i])
            st.traj[i] = particles

            particles = numpy.copy(particles)
            self.cond_predict(particles, xin, st.u[i])
            (_xil, zl, Pl) = self.get_states(particles)
            self.set_states(particles, xin, zl, Pl)
            
        if (st.y[-1] != None):
            self.measure(particles, st.y[-1])
        
        
        (_xil, zl, Pl) = self.get_states(particles)
        (xin, zn, Pn) = self.get_states(st.traj[-1])
        self.set_states(particles, xin, zl, Pl)
        st.traj[-1] = particles
        
        # Backward smoothing
        for i in reversed(xrange(T-1)):
            self.t = st.t[i]
            (xin, zn, Pn) = self.get_states(st.traj[i+1])
            (xi, z, P) = self.get_states(st.traj[i])
            (Al, fl, Ql) = self.calc_cond_dynamics(st.traj[i], xin, st.u[i])
            for j in xrange(M):
                (zs, Ps, Ms) = self.kf.smooth(z[j], P[j], zn[j], Pn[j],
                                              Al[j], fl[j], Ql[j])
                self.set_states(st.traj[i][j:j+1], xi[j], (zs,), (Ps,))
                self.set_Mz(st.traj[i][j:j+1], (Ms,))

