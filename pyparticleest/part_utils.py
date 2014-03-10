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
    def sample_process_noise(self, u, particles):
        """ Return process noise for input u """
        return
    
    @abc.abstractmethod
    def update(self, u, noise, particles):
        """ Update estimate using 'data' as input """
        return
    
    @abc.abstractmethod    
    def measure(self, y, particles):
        """ Return the log-pdf value of the measurement """
        return
    
class FFBSiInterface(ParticleFilteringInterface):
    """ Base class for particles to be used with particle smoothing """
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def next_pdf(self, next_cpart, u, particles):
        """ Return the log-pdf value for the possible future state 'next' given input u """
        pass
    
    @abc.abstractmethod
    def sample_smooth(self, next_part, particles):
        """ Update ev. Rao-Blackwellized states conditioned on "next_part" """
        pass

class FFBSiRSInterface(FFBSiInterface):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def next_pdf_max(self, u, particles):
        """ Return the log-pdf value for the possible future state 'next' given input u """
        pass

class HierarchicalBase(ParticleFilteringInterface):
    """ Base class for Rao-Blackwellization of hierarchical models """
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, Az=None, fz=None, Qz=None,
                 C=None ,hz=None, R=None, t0=0):
        
        self.kf = kalman.KalmanFilter(A=Az, C=C, 
                                        Q=Qz, R=R,
                                        f_k=fz, h_k=hz)
        
        # Sore z0, P0 needed for default implementation of 
        # get_z0_initial and get_grad_z0_initial
        self.t = t0
    
    def get_lin_pred_dynamics(self, u, particles):
        """ Return matrices describing affine relation of next
            nonlinear state conditioned on current linear state
            
            \z_{t+1]} = A_z * z_t + f_z + v_z, v_z ~ N(0,Q_z)
            
            conditioned on the value of xi_{t+1}. 
            (Not the same as the dynamics unconditioned on xi_{t+1})
            when for example there is a noise correlation between the 
            linear and nonlinear state dynamics) 
            """
        return (None, None, None)
    
    def get_lin_meas_dynamics(self, y, particles):
        return (y, None, None, None)
    
    def update(self, u, noise, particles):
        """ Update estimate using noise as input """
        xin = self.calc_xi_next(u, noise, particles)
        # Update linear estimate with data from measurement of next non-linear
        # state 
        (_xil, zl, Pl) = self.get_states(particles)
        N = len(particles)
        (Az, fz, Qz) = self.get_lin_pred_dynamics(u, particles)
        if (Az == None):
            Az=numpy.repeat(self.kf.A[numpy.newaxis,:,:], N, axis=0)
            #Az=N*(self.kf.A,)
        if (fz == None):
            fz=numpy.repeat(self.kf.f_k[numpy.newaxis,:,:], N, axis=0)
            #fz=N*(self.kf.f_k,)
        if (Qz == None):
            Qz=numpy.repeat(self.kf.Q[numpy.newaxis,:,:], N, axis=0)
            #Qz=N*(self.kf.Q,)
        for i in xrange(len(zl)):
            # Predict z_{t+1}
            (zl[i], Pl[i]) = self.kf.predict_full(zl[i], Pl[i], Az[i], fz[i], Qz[i])
        
        # Predict next states conditioned on eta_next
        self.set_states(particles, xin, zl, Pl)
        self.t = self.t + 1.0
    
    def measure(self, y, particles):
        """ Return the log-pdf value of the measurement """
        
        lyxi = self.measure_nonlin(y, particles)
        (_xil, zl, Pl) = self.get_states(particles)
        N = len(particles)
        (y, Cz, hz, Rz) = self.get_lin_meas_dynamics(y, particles)
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
            lyz[i] = self.kf.measure_full(y, zl[i], Pl[i], Cz[i], hz[i], Rz[i])
        
        return lyxi + lyz
    
    def next_pdf(self, next_cpart, u, particles):
        """ Return the log-pdf value for the possible future state 'next' given input u """
        N = len(particles)
        lpxi = self.next_pdf_xi(next_cpart[0], u, particles).ravel()
        (Az, fz, Qz) = self.get_lin_pred_dynamics(u, particles)
        lpz = numpy.empty_like(lpxi)
        (xil, zl, Pl) = self.get_states(particles)
        zln = numpy.empty_like(zl)
        Pln = numpy.empty_like(Pl)
        if (Az == None):
            Az=numpy.repeat(self.kf.A[numpy.newaxis,:,:], N, axis=0)
            #Az=N*(self.kf.A,)
        if (fz == None):
            fz=numpy.repeat(self.kf.f_k[numpy.newaxis,:,:], N, axis=0)
            #fz=N*(self.kf.f_k,)
        if (Qz == None):
            Qz=numpy.repeat(self.kf.Q[numpy.newaxis,:,:], N, axis=0)
            #Qz=N*(self.kf.Q,)
        for i in xrange(len(zl)):
            # Predict z_{t+1}
            (zln[i], Pln[i]) = self.kf.predict_full(zl[i], Pl[i], Az[i], fz[i], Qz[i])
            #lpz[i] = kalman.lognormpdf(next_cpart[1].reshape((-1,1)), zl[i], Pl[i])
        #mul = numpy.repeat(next_cpart[1].reshape((-1,1)),N,axis=0)
        mul = N*(next_cpart[1].reshape((-1,1)),)
        lpz = kalman.lognormpdf_vec(zln, mul, Pln)
        #lpz = kalman.lognormpdf_jit(zl, mul, Pl)
        return lpxi + lpz
    
    def sample_smooth(self, next_part, u, particle):
        """ Update ev. Rao-Blackwellized states conditioned on "next_part" """
        (xil, zl, Pl) = self.get_states(particle)
        if (next_part != None):
            (Az, fz, Qz) = self.get_lin_pred_dynamics(u, particle)
            if (Az == None):
                Az=numpy.repeat(self.kf.A[numpy.newaxis,:,:], 1, axis=0)
                #Az=(self.kf.A,)
            if (fz == None):
                fz=numpy.repeat(self.kf.f_k[numpy.newaxis,:,:], 1, axis=0)
                #fz=(self.kf.f_k,)
            if (Qz == None):
                Qz=numpy.repeat(self.kf.Q[numpy.newaxis,:,:], 1, axis=0)
                #Qz=(self.kf.Q,)
            self.kf.measure_full(next_part[1].reshape((-1,1)), zl[0], Pl[0], C=Az[0], h_k=fz[0], R=Qz[0])
        
        xi = copy.copy(xil[0])
        z = numpy.random.multivariate_normal(zl[0].ravel(), Pl[0])
            
        return (xi, z)

    @abc.abstractmethod
    def next_pdf_xi(self, next_xi, u, particles):
        pass
    
    @abc.abstractmethod
    def calc_xi_next(self, u, noise, particles):
        pass
    
    @abc.abstractmethod
    def measure_nonlin(self, y, particles):
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
    
class HierarchicalRSBase(HierarchicalBase,FFBSiRSInterface):
    def __init__(self, **kwargs):
        super(HierarchicalRSBase, self).__init__(**kwargs)
        
    def next_pdf_max(self, u, particles):
        N = len(particles)
        lpxi = self.next_pdf_xi_max(u, particles)
        (Az, _fz, Qz) = self.get_lin_pred_dynamics(u, particles)
        lpz = numpy.empty_like(lpxi)
        (_xil, _zl, Pl) = self.get_states(particles)
        if (Az == None):
            #Az=numpy.repeat(self.kf.A[numpy.newaxis,:,:], N, axis=0)
            Az=N*(self.kf.A,)
        if (Qz == None):
            #Qz=numpy.repeat(self.kf.Q[numpy.newaxis,:,:], N, axis=0)
            Qz=N*(self.kf.Q,)
        nx = len(Qz[0])
        for i in xrange(N):
            # Predict z_{t+1}
            Pn=Az[i].dot(Pl[i]).dot(Az[i].T) + Qz[i]
            lpz[i] = -0.5*nx*math.log(2*math.pi)+numpy.linalg.slogdet(Pn)[1]
        lpmax = numpy.max(lpxi+lpz)
        return lpmax
    
    @abc.abstractmethod
    def next_pdf_xi_max(self, u, particles):
        pass

class RBPFBase(HierarchicalBase):
    """ Base class for Rao-Blackwellized particles """
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, Axi=None, fxi=None, Qxi=None,
                 **kwargs):
        
        self.Axi = numpy.copy(Axi)
        self.fxi = numpy.copy(fxi)
        self.Qxi = numpy.copy(Qxi)
        
        super(RBPFBase, self).__init__(**kwargs)
    
    def get_nonlin_pred_dynamics(self, u, particles):
        """ Return matrices describing affine relation of next
            nonlinear state conditioned on current linear state
            
            \xi_{t+1]} = A_xi * z_t + f_xi + v_xi, v_xi ~ N(0,Q_xi)
            
            Return (A_xi, f_xi, Q_xi) where each element is a list
            with the corresponding matrix for each particle. None indicates
            that the matrix is identical for all particles and the value stored
            in this class should be used instead
            """
        return (None, None, None)
    
    def get_condlin_pred_dynamics(self, u, xi_next, particles):
        """ Return matrices describing affine relation of next
            nonlinear state conditioned on current linear state
            
            \z_{t+1]} = A_z * z_t + f_z + v_z, v_z ~ N(0,Q_z)
            
            conditioned on the value of xi_{t+1}. 
            (Not the same as the dynamics unconditioned on xi_{t+1})
            when for example there is a noise correlation between the 
            linear and nonlinear state dynamics) 
            """
        return (None, None, None)
    
# This is not implemented  
#    def get_condlin_meas_dynamics(self, y, xi_next, particles):
#        return (y, None, None, None)
    
    def update(self, u, noise, particles):
        """ Update estimate using noise as input """
        xin = self.calc_xi_next(u, noise, particles)
        N = len(particles)
        # Update linear estimate with data from measurement of next non-linear
        # state 
        (zl, Pl) = self.get_lin_states(particles)
        (Ax, fx, Qx) = self.get_nonlin_pred_dynamics(particles, u)
        
        # This is probably not so nice performance-wise, but will
        # work initially to profile where the bottlenecks are.
        if (Ax == None):
            Ax=N*(self.Ax,)
        if (fx == None):
            fx=N*(self.fx,)
        if (Qx == None):
            Qx= N*(self.Qx)
        for i in xrange(N):
            self.kf.measure_full(zl[i], Pl[i], xin, fx[i], Ax[i], Qx[i])
        
        (Az, fz, Qz) = self.get_condlin_pred_dynamics(u, xin, particles)
        for i in xrange(len(zl)):
            # Predict z_{t+1}
            self.kf.predict_full(zl[i], Pl[i], Az[i], fz[i], Qz[i])
            pass
        
        # Predict next states conditioned on eta_next
        self.set_states(particles, xin, zl, Pl)
        self.t = self.t + 1.0
    
    def measure(self, y, particles):
        """ Return the log-pdf value of the measurement """
        N = len(particles)
        return self.kf.measure(y=y)
    
    @abc.abstractmethod
    def calc_xi_next(self, particles, u, noise):
        pass
    
    @abc.abstractmethod
    def meas_eta_next(self, eta_next):
        """ Update linear estimate using observation 
            of next non-linear state """
        pass

    @abc.abstractmethod
    def set_lin_states(self, particles, z_list, P_list):
        """ Set the estimate of the Rao-Blackwellized states """
        pass
 
    @abc.abstractmethod
    def get_lin_states(self, particles):
        """ Return the estimate of the Rao-Blackwellized states.
            Must return two variables, the first a list containing all the
            expected values, the second a list of the corresponding covariance
            matrices"""
        pass

    def cond_predict(self, eta_next=None):
        """ Predict linear-guassian states z_{t+1|t} conditioned on eta_{t+1} """
        (z, P) = self.kf.predict()
        return (z.reshape((-1,1)), P)

    def clin_update(self, next_part=None):
        self.kf.time_update()
    
    def prep_measure(self, y):
        """ Pre-processing of measurement y """
        return y
    


    def prep_update(self, u):
        """ Update dynamics with u as input """
        pass
    

           
        


    


#    
#class RBPSBase(RBPFBase, ParticleSmoothingInterface):
#    __metaclass__ = abc.ABCMeta
#    
##    def __init__(self, z0, P0, 
##                 Az=None, Bz=None, C=None,
##                  Qz=None, R=None, f_k=None, h_k=None):
##        super(RBPSBase,self).__init__(z0=z0, P0=P0, Az=Az, C=C,
##                                      Qz=Qz, R=R, f_k=f_k, h_k=h_k)
#        
#    def clin_measure(self, y, next_part=None):
#        """ Kalman measurement of the linear states conditioned on the non-linear trajectory estimate """
#        self.kf.measure(y)
#
#    def clin_smooth(self, next_part):
#        """ Kalman smoothing of the linear states conditioned on the next particles linear states
#            Before calling this method clin_dynamics should be called to update the dynamics
#            according to the conditioning on the non-linear trajectory """
#        tmp = (next_part.get_lin_est())
#        self.kf.smooth(tmp[0], tmp[1])
#
#

