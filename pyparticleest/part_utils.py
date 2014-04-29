""" Collection of functions and classes used for Particle Filtering/Smoothing """
import abc
import numpy
import copy
import math
import pyximport
pyximport.install(inplace=True)
import kalman
# This was slower than kalman.lognormpdf
#from scipy.stats import multivariate_normal

class ParticleFilteringInterface(object):
    """ Base class for particles to be used with particle filtering """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def create_initial_estimate(self, N):
        """ Sample N particle from initial distribution """
        pass
     
    @abc.abstractmethod
    def sample_process_noise(self, particles, u, t):
        """ Return process noise for input u """
        pass
    
    @abc.abstractmethod
    def update(self, particles, u, t, noise):
        """ Update estimate using 'data' as input """
        pass
    
    @abc.abstractmethod    
    def measure(self, particles, y, t):
        """ Return the log-pdf value of the measurement """
        pass
    
    def copy_ind(self, particles, new_ind=None):
        if (new_ind != None):
            N = len(new_ind)
        else:
            N = len(particles)
            new_ind = range(N)
        new_part = numpy.empty(N, type(particles[0]))
        for k in range(numpy.shape(new_ind)[0]):
            new_part[k] = copy.copy(particles[new_ind[k]])
        return new_part
    
class AuxiliaryParticleFilteringInterface(object):
    """ Base class for particles to be used with particle filtering """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def eval_1st_stage_weights(self, particles, u, y, t):
        """ Evaluate "first stage weights" for the auxiliary particle filter.
            (log-probability of measurement using some propagated statistic, such
            as the mean, for the future state) """
        pass
    
class FFBSiInterface(ParticleFilteringInterface):
    """ Base class for particles to be used with particle smoothing """
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def next_pdf(self, particles, next_cpart, u, t):
        """ Return the log-pdf value for the possible future state 'next' given input u """
        pass
    
    @abc.abstractmethod
    def sample_smooth(self, particles, next_part, u, t):
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
                 C=None ,hz=None, R=None):
        
        self.kf = kalman.KalmanSmoother(lz, A=Az, C=C, 
                                        Q=Qz, R=R,
                                        f_k=fz, h_k=hz)
        
    def set_dynamics(self, Az=None, C=None, Qz=None, R=None, fz=None, hz=None):
        return self.kf.set_dynamics(Az, C, Qz, R, fz, hz)
    
    def get_nonlin_pred_dynamics(self, particles, u, t):
        """ Return matrices describing affine relation of next
            nonlinear state conditioned on current linear state
            
            xi_{t+1]} = A_xi * z_t + f_xi + v_xi, v_xi ~ N(0,Q_xi)
            
            Return (A_xi, f_xi, Q_xi) where each element is a list
            with the corresponding matrix for each particle. None indicates
            that the matrix is identical for all particles and the value stored
            in this class should be used instead
            """
        return (None, None, None)
    
    def get_nonlin_pred_dynamics_int(self, particles, u, t):
        (Axi, fxi, Qxi) = self.get_nonlin_pred_dynamics(particles, u=u, t=t)
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
    
    def get_lin_pred_dynamics(self, particles, u, t):
        """ Return matrices describing affine relation of next
            nonlinear state conditioned on current linear state
            
            \z_{t+1]} = A_z * z_t + f_z + v_z, v_z ~ N(0,Q_z)
            
            conditioned on the value of xi_{t+1}. 
            (Not the same as the dynamics unconditioned on xi_{t+1})
            when for example there is a noise correlation between the 
            linear and nonlinear state dynamics) 
            """
        return (None, None, None)
    
    def get_lin_pred_dynamics_int(self, particles, u, t):
        N = len(particles)
        (Az, fz, Qz) = self.get_lin_pred_dynamics(particles, u=u, t=t)
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
    
    def get_meas_dynamics(self, particles, y, t):
        return (y, None, None, None)
    
    def get_meas_dynamics_int(self, particles, y, t):
        N=len(particles)
        (y, Cz, hz, Rz) = self.get_meas_dynamics(particles=particles, y=y, t=t)
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
    
    def update(self, particles, u, t, noise):
        """ Update estimate using noise as input """
        # Calc (xi_{t+1} | xi_t, z_t, y_t)
        xin = self.calc_xi_next(particles=particles, u=u, t=t, noise=noise)
        # Calc (z_{t+1} | xi_{t+1}, y_t)
        self.cond_predict(particles=particles, xi_next=xin, u=u, t=t)


    
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

        for i in xrange(T-1):
            if (st.y[i] != None):
                self.measure(particles, y=st.y[i], t=st.t[i])
            (xin, _zn, _Pn) = self.get_states(st.traj[i+1])
            #self.meas_xi_next(particles, xin, u=st.u[i], t=st.t[i])
            st.traj[i] = particles

            particles = numpy.copy(particles)
            self.cond_predict(particles, xin, u=st.u[i], t=st.t[i])
            
        if (st.y[-1] != None):
            self.measure(particles, y=st.y[-1], t=st.t[-1])
        
        st.traj[-1] = particles
        
        # Backward smoothing
        for i in reversed(xrange(T-1)):
            (xin, zn, Pn) = self.get_states(st.traj[i+1])
            particles = st.traj[i]
            self.meas_xi_next(particles, xin, u=st.u[i], t=st.t[i])
            (xi, z, P) = self.get_states(particles)
            (Al, fl, Ql) = self.calc_cond_dynamics(particles, xin, u=st.u[i], t=st.t[i])
            for j in xrange(M):
                
                (zs, Ps, Ms) = self.kf.smooth(z[j], P[j], zn[j], Pn[j],
                                              Al[j], fl[j], Ql[j])
                self.set_states(st.traj[i][j:j+1], xi[j], zs[numpy.newaxis], Ps[numpy.newaxis])
                self.set_Mz(st.traj[i][j:j+1], Ms[numpy.newaxis])

