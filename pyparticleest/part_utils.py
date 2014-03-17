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
    def next_pdf(self, particles, next_cpart, u=None):
        """ Return the log-pdf value for the possible future state 'next' given input u """
        pass
    
    @abc.abstractmethod
    def sample_smooth(self, particles, next_part):
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
    
    def __init__(self, Az=None, fz=None, Qz=None,
                 C=None ,hz=None, R=None, t0=0):
        
        self.kf = kalman.KalmanSmoother(A=Az, C=C, 
                                        Q=Qz, R=R,
                                        f_k=fz, h_k=hz)
        
        # Sore z0, P0 needed for default implementation of 
        # get_z0_initial and get_grad_z0_initial
        self.t = t0
    
    def get_nonlin_pred_dynamics(self, u, particles):
        """ Return matrices describing affine relation of next
            nonlinear state conditioned on current linear state
            
            xi_{t+1]} = A_xi * z_t + f_xi + v_xi, v_xi ~ N(0,Q_xi)
            
            Return (A_xi, f_xi, Q_xi) where each element is a list
            with the corresponding matrix for each particle. None indicates
            that the matrix is identical for all particles and the value stored
            in this class should be used instead
            """
        return (None, None, None)
    
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
    
    def get_meas_dynamics(self, y, particles):
        return (y, None, None, None)
    
# This is not implemented  
#    def get_condlin_meas_dynamics(self, y, xi_next, particles):
#        return (y, None, None, None)
    
    def update(self, u, noise, particles):
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
    
#    def __init__(self, z0, P0, 
#                 Az=None, Bz=None, C=None,
#                  Qz=None, R=None, f_k=None, h_k=None):
#        super(RBPSBase,self).__init__(z0=z0, P0=P0, Az=Az, C=C,
#                                      Qz=Qz, R=R, f_k=f_k, h_k=h_k)
    @abc.abstractmethod
    def get_rb_initial(self, xi_initial):
        pass    
    
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



class HierarchicalBase(RBPFBase):
    """ Base class for Rao-Blackwellization of hierarchical models """
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        
        super(HierarchicalBase, self).__init__(**kwargs)
        # Sore z0, P0 needed for default implementation of 
        # get_z0_initial and get_grad_z0_initial

    def update(self, u, noise, particles):
        """ Update estimate using noise as input """
        xin = self.calc_xi_next(particles, u, noise)
        # Update linear estimate with data from measurement of next non-linear
        # state 
        (_xil, zl, Pl) = self.get_states(particles)
        N = len(particles)
        (Az, fz, Qz) = self.get_lin_pred_dynamics(particles, u)
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
    
    def next_pdf(self, particles, next_cpart, u=None):
        """ Return the log-pdf value for the possible future state 'next' given input u """
        N = len(particles)
        lpxi = self.next_pdf_xi(particles, next_cpart[0], u).ravel()
        (Az, fz, Qz) = self.get_lin_pred_dynamics(particles, u)
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
    
    def sample_smooth(self, particle, next_part, u):
        """ Update ev. Rao-Blackwellized states conditioned on "next_part" """
        (xil, zl, Pl) = self.get_states(particle)
        if (next_part != None):
            (Az, fz, Qz) = self.get_lin_pred_dynamics(particle, u)
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
    def next_pdf_xi(self, particles, next_xi, u):
        pass
    
    @abc.abstractmethod
    def calc_xi_next(self, particles, u, noise):
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
        
    def next_pdf_max(self, particles, u=None):
        N = len(particles)
        lpxi = self.next_pdf_xi_max(particles, u)
        (Az, _fz, Qz) = self.get_lin_pred_dynamics(particles, u)
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
    def next_pdf_xi_max(self, particles, u=None):
        pass



class MixedNLGaussian(RBPSBase):
    """ Base class for particles of the type mixed linear/non-linear with additive gaussian noise.
    
        Implement this type of system by extending this class and provide the methods for returning 
        the system matrices at each time instant  """
    def __init__(self, Az=None, C=None, Qz=None, R=None, fz=None,
                 Axi=None, Qxi=None, Qxiz=None, fxi=None, h=None, params=None, t0=0):
        self.Axi = numpy.copy(Axi)
        self.fxi = numpy.copy(fxi)
        self.Qxi = numpy.copy(Qxi)
        self.Qxiz = numpy.copy(Qxiz)
        return super(MixedNLGaussian, self).__init__(Az=Az, C=C, 
                                              Qz=Qz, R=R,
                                              hz=h, fz=fz,
                                              t0=t0)

    def sample_process_noise(self, particles, u=None): 
        """ Return sampled process noise for the non-linear states """
        (Axi, fxi, Qxi) = self.get_nonlin_pred_dynamics(particles, u)
        (_xil, zl, Pl) = self.get_states(particles)
        N = len(particles)
        # This is probably not so nice performance-wise, but will
        # work initially to profile where the bottlenecks are.
        if (Axi == None):
            Axi=N*(self.Axi,)
        if (fxi == None):
            fxi=N*(self.fxi,)
        if (Qxi == None):
            Qxi= N*(self.Qxi,)
                    
        dim=len(_xil[0])
        noise = numpy.empty((N,dim))
        zeros = numpy.zeros(dim)
        for i in xrange(N):
            Sigma = Qxi[i] + Axi[i].dot(Pl[i]).dot(Axi[i].T)
            noise[i] =  numpy.random.multivariate_normal(zeros, Sigma).ravel()    
        return noise

    def calc_xi_next(self, particles, noise, u=None):
        """ Update non-linear state using sampled noise,
        # the noise term here includes the uncertainty from z """
        xi_pred = self.pred_xi(particles=particles, u=u)
        xi_next = xi_pred + noise
   
        return xi_next

    def pred_xi(self, particles, u=None):
        N = len(particles)
        (Axi, fxi, _Qxi) = self.get_nonlin_pred_dynamics(particles=particles, u=u)
        (xil, zl, _Pl) = self.get_states(particles)
        dim=len(xil[0])
        xi_next = numpy.empty((N,dim))
        # This is probably not so nice performance-wise, but will
        # work initially to profile where the bottlenecks are.
        if (Axi == None):
            Axi=N*(self.Axi,)
        if (fxi == None):
            fxi=N*(self.fxi,)
        for i in xrange(N):
            xi_next[i] =  Axi[i].dot(zl[i]) + fxi[i]
        return xi_next
    
    def meas_xi_next(self, particles, xi_next, u=None):
        """ Update estimate using observation of next state """
        # This is what is sometimes called "the second measurement update"
        # for Rao-Blackwellized particle filters
        
        N = len(particles)
        (xil, zl, Pl) = self.get_states(particles)
        (Axi, fxi, Qxi) = self.get_nonlin_pred_dynamics(particles=particles, u=u)
        if (Axi == None):
            Axi=N*(self.Axi,)
        if (fxi == None):
            fxi=N*(self.fxi,)
        if (Qxi == None):
            Qxi= N*(self.Qxi,)
        for i in xrange(len(zl)):
            self.kf.measure_full(y=xi_next[i], z=zl[i], P=Pl[i], C=Axi[i], h_k=fxi[i], R=Qxi[i])
        
        # Predict next states conditioned on eta_next
        self.set_states(particles, xil, zl, Pl)
    
    def get_cross_covariance(self, particles, u):
        N = len(particles)
        Qxiz = N*(self.Qxiz,)
        return Qxiz
    
    def calc_cond_dynamics(self, particles, xi_next, u=None):
        #Compensate for noise correlation
        N = len(particles)
        #(xil, zl, Pl) = self.get_states(particles)
        (Az, fz, Qz) = self.get_lin_pred_dynamics(particles=particles, u=u)
        (Axi, fxi, Qxi) = self.get_nonlin_pred_dynamics(particles=particles, u=u)
        Qxiz = self.get_cross_covariance(particles=particles, u=u)
        
        if (Axi == None):
            Axi = N*(self.Axi,)
        if (fxi == None):
            fxi = N*(self.fxi,)
        if (Qxi == None):
            Qxi = N*(self.Qxi,)
        if (Az == None):
            Az = N*(self.kf.A,)
        if (fz == None):
            fz = N*(self.kf.f_k,)
        if (Qz == None):
            Qz = N*(self.kf.Q,)
        
        Acond = list()
        fcond = list()
        Qcond = list()
        
        for i in xrange(N):
            #TODO linalg.solve instead?
            tmp = Qxiz[i].T.dot(numpy.linalg.inv(Qxi[i]))
            Acond.append(Az[i] - tmp.dot(Axi[i]))
            #Acond.append(Az[i])
            fcond.append(fz[i] +  tmp.dot(xi_next[i] - fxi[i]))
            # TODO, shouldn't Qz be affected? Why wasn't it before?
            Qcond.append(Qz[i])
        
        return (Acond, fcond, Qcond)
    
    def cond_predict(self, particles, xi_next, u=None):
        #Compensate for noise correlation
        (Az, fz, Qz) = self.calc_cond_dynamics(particles=particles, xi_next=xi_next, u=u)
        N = len(particles)
        (xil, zl, Pl) = self.get_states(particles)
        for i in xrange(len(zl)):
            (zl[i], Pl[i]) = self.kf.predict_full(z=zl[i], P=Pl[i], A=Az[i], f_k=fz[i], Q=Qz[i])
        
        # Predict next states conditioned on eta_next
        self.set_states(particles, xil, zl, Pl)
        
    def measure(self, particles, y):
        """ Return the log-pdf value of the measurement """
        
        (xil, zl, Pl) = self.get_states(particles)
        N = len(particles)
        (y, Cz, hz, Rz) = self.get_meas_dynamics(y=y, particles=particles)
        if (Cz == None):
            Cz=N*(self.kf.C,)
        if (hz == None):
            hz=N*(self.kf.h_k,)
        if (Rz == None):
            Rz=N*(self.kf.R,)
            
        lyz = numpy.empty(N)
        for i in xrange(len(zl)):
            # Predict z_{t+1}
            lyz[i] = self.kf.measure_full(y=y, z=zl[i], P=Pl[i], C=Cz[i], h_k=hz[i], R=Rz[i])
            
        self.set_states(particles, xil, zl, Pl)
        return lyz

    def next_pdf_max(self, particles, u=None):
        """ Implements the fwd_peak_density function for MixedNLGaussian models """
        N = len(particles)
        pmax = numpy.empty(N)
        (Az, fz, Qz) = self.get_lin_pred_dynamics(particles=particles, u=u)
        (Axi, fxi, Qxi) = self.get_nonlin_pred_dynamics(particles=particles, u=u)
        Qxiz = self.get_cross_covariance(particles=particles, u=u)
        (xil, zl, Pl) = self.get_states(particles)
        if (Axi == None):
            Axi = N*(self.Axi,)
        if (fxi == None):
            fxi = N*(self.fxi,)
        if (Qxi == None):
            Qxi = N*(self.Qxi,)
        if (Az == None):
            Az = N*(self.kf.A,)
        if (fz == None):
            fz = N*(self.kf.f_k,)
        if (Qz == None):
            Qz = N*(self.kf.Q,)
        dim=len(xil[0])+len(zl[0])
        zeros = numpy.zeros((dim,1))
        for i in xrange(N):
            A = numpy.vstack((Axi[i], Az[i]))
            Q = numpy.vstack((numpy.hstack((Qxi[i], Qxiz[i])),
                              numpy.hstack((Qxiz[i].T, Qz[i]))))
            self.Sigma = Q + A.dot(Pl[i]).dot(A.T)
            pmax[i] = kalman.lognormpdf(zeros, zeros, self.Sigma)
        
        return numpy.max(pmax)
        
    def next_pdf(self, particles, next_part, u=None):
        """ Implements the next_pdf function for MixedNLGaussian models """
        
        N = len(particles)
        (Az, fz, Qz) = self.get_lin_pred_dynamics(particles=particles, u=u)
        (Axi, fxi, Qxi) = self.get_nonlin_pred_dynamics(particles=particles, u=u)
        Qxiz = self.get_cross_covariance(particles=particles, u=u)
        (xil, zl, Pl) = self.get_states(particles)
        if (Axi == None):
            Axi = N*(self.Axi,)
        if (fxi == None):
            fxi = N*(self.fxi,)
        if (Qxi == None):
            Qxi = N*(self.Qxi,)
        if (Az == None):
            Az = N*(self.kf.A,)
        if (fz == None):
            fz = N*(self.kf.f_k,)
        if (Qz == None):
            Qz = N*(self.kf.Q,)
        
        lpx = numpy.empty(N)
        x_next = numpy.vstack(next_part)
#        #z_diff= next_part.sampled_z - self.kf.predict()[0]
#        z_diff= next_part.sampled_z - self.cond_predict(eta_est)[0]
        
        for i in xrange(N):
            A = numpy.vstack((Axi[i], Az[i]))
            f = numpy.vstack((fxi[i], fz[i]))
            Q = numpy.vstack((numpy.hstack((Qxi[i], Qxiz[i])),
                              numpy.hstack((Qxiz[i].T, Qz[i]))))
            xp = f + A.dot(zl[i])
            Sigma = A.dot(Pl[i]).dot(A.T) + Q
            lpx[i] = kalman.lognormpdf(x_next,mu=xp,S=Sigma)
        
        return lpx
    
    def sample_smooth(self, particles, next_part, u=None):
        """ Implements the sample_smooth function for MixedNLGaussian models """
        part = numpy.copy(particles[0])
        (xil, zl, Pl) = self.get_states([part,])
        if (next_part != None):
            self.meas_xi_next([part,], next_part[0])
            (Acond, fcond, Qcond) = self.calc_cond_dynamics([part,], next_part[0], u)
            (xil, zl, Pl) = self.get_states([part,])
            self.kf.measure_full(next_part[1], zl[0], Pl[0],
                                 C=Acond[0], h_k=fcond[0], R=Qcond[0])

        xi = copy.copy(xil[0]).reshape((-1,1))
        z = numpy.random.multivariate_normal(zl[0].ravel(), Pl[0]).reshape((-1,1))
            
        return (xi, z)
#    
#    def measure(self, y):
#        y=numpy.reshape(y, (-1,1))
#        return super(MixedNLGaussian, self).measure(y)
    
#    def eval_1st_stage_weight(self, u,y):
##        eta_old = copy.deepcopy(self.get_nonlin_state())
##        lin_old = copy.deepcopy(self.get_lin_est())
##        t_old = self.t
#        self.prep_update(u)
#        noise = numpy.zeros_like(self.eta)
#        self.update(u, noise)
#        
#        yn = self.prep_measure(y)
#        logpy = self.measure(yn)
#        
#        # Restore state
##        self.set_lin_est(lin_old)
##        self.set_nonlin_state(eta_old)
##        self.t = t_old
#        
#        return logpy
        
    def get_nonlin_state(self, particles):
        return self.eta

    def set_nonlin_state(self, particles):
        self.eta = numpy.copy(inp)
