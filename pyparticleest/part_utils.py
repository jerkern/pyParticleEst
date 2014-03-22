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
        # This is probably not so nice performance-wise, but will
        # work initially to profile where the bottlenecks are.
        if (Axi == None):
            Axi=N*(self.Axi,)
        if (fxi == None):
            fxi=N*(self.fxi,)
        if (Qxi == None):
            Qxi= N*(self.Qxi,)
        return (Axi, fxi, Qxi)
    
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
        if (Az == None):
            #Az=numpy.repeat(self.kf.A[numpy.newaxis,:,:], N, axis=0)
            Az=N*(self.kf.A,)
        if (fz == None):
            #fz=numpy.repeat(self.kf.f_k[numpy.newaxis,:,:], N, axis=0)
            fz=N*(self.kf.f_k,)
        if (Qz == None):
            Qz=numpy.repeat(self.kf.Q[numpy.newaxis,:,:], N, axis=0)
            #Qz=N*(self.kf.Q,)
        return (Az, fz, Qz)
    
    def get_meas_dynamics(self, particles, y):
        return (y, None, None, None)
    
    def get_meas_dynamics_int(self, particles, y):
        N=len(particles)
        (y, Cz, hz, Rz) = self.get_meas_dynamics(particles=particles, y=y)
        if (Cz == None):
            Cz=N*(self.kf.C,)
        if (hz == None):
            hz=N*(self.kf.h_k,)
        if (Rz == None):
            Rz=N*(self.kf.R,)
        return (y, Cz, hz, Rz)
    
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

    def update(self, particles, u, noise):
        """ Update estimate using noise as input """
        xin = self.calc_xi_next(particles, u, noise)
        # Update linear estimate with data from measurement of next non-linear
        # state 
        (_xil, zl, Pl) = self.get_states(particles)
        (Az, fz, Qz) = self.get_lin_pred_dynamics_int(particles, u)
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
        (Az, fz, Qz) = self.get_lin_pred_dynamics_int(particles, u)
        lpz = numpy.empty_like(lpxi)
        (xil, zl, Pl) = self.get_states(particles)
        zln = numpy.empty_like(zl)
        Pln = numpy.empty_like(Pl)

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
            (Az, fz, Qz) = self.get_lin_pred_dynamics_int(particle, u)
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
        (Az, _fz, Qz) = self.get_lin_pred_dynamics_int(particles, u)
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
    def next_pdf_xi_max(self, particles, u=None):
        pass



class MixedNLGaussian(RBPSBase):
    """ Base class for particles of the type mixed linear/non-linear with additive gaussian noise.
    
        Implement this type of system by extending this class and provide the methods for returning 
        the system matrices at each time instant  """
    def __init__(self, Az=None, C=None, Qz=None, R=None, fz=None,
                 Axi=None, Qxi=None, Qxiz=None, fxi=None, h=None, params=None, t0=0):
        if (Axi != None):
            self.Axi = numpy.copy(Axi)
        else:
            self.Axi = None
        if (fxi != None):
            self.fxi = numpy.copy(fxi)
        else:
            self.fxi = None
        if (Qxi != None):
            self.Qxi = numpy.copy(Qxi)
        else:
            self.Qxi = None
        if (Qxiz != None):
            self.Qxiz = numpy.copy(Qxiz)
        else:
            self.Qxiz = None
        return super(MixedNLGaussian, self).__init__(Az=Az, C=C, 
                                              Qz=Qz, R=R,
                                              hz=h, fz=fz,
                                              t0=t0)

    def set_dynamics(self, Az=None, fz=None, Qz=None, R=None,
                     Axi=None, fxi=None, Qxi=None, Qxiz=None, 
                     C=None, h=None):
        super(MixedNLGaussian, self).set_dynamics(Az=Az, C=C, Qz=Qz, R=R, fz=fz,hz=h)

        if (Axi != None):
            self.Axi = numpy.copy(Axi)
        if (Az != None):
            self.Az = numpy.copy(Az)
        if (Qxi != None):
            self.Qxi = numpy.copy(Qxi)
        if (Qxiz != None):
            self.Qxiz = numpy.copy(Qxiz)
        if (Qz != None):
            self.Qz = numpy.copy(self.kf.Q)
        if (fz != None):
            self.fz = numpy.copy(self.kf.f_k)
        if (fxi != None):
            self.fxi = numpy.copy(fxi)

    def sample_process_noise(self, particles, u=None): 
        """ Return sampled process noise for the non-linear states """
        (Axi, fxi, Qxi) = self.get_nonlin_pred_dynamics_int(particles, u)
        (_xil, zl, Pl) = self.get_states(particles)
        N = len(particles)
        # This is probably not so nice performance-wise, but will
        # work initially to profile where the bottlenecks are.
                    
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
        (Axi, fxi, _Qxi) = self.get_nonlin_pred_dynamics_int(particles=particles, u=u)
        (xil, zl, _Pl) = self.get_states(particles)
        dim=len(xil[0])
        xi_next = numpy.empty((N,dim))
        # This is probably not so nice performance-wise, but will
        # work initially to profile where the bottlenecks are.
        for i in xrange(N):
            xi_next[i] =  Axi[i].dot(zl[i]) + fxi[i]
        return xi_next
    
    def meas_xi_next(self, particles, xi_next, u=None):
        """ Update estimate using observation of next state """
        # This is what is sometimes called "the second measurement update"
        # for Rao-Blackwellized particle filters
        
        N = len(particles)
        (xil, zl, Pl) = self.get_states(particles)
        (Axi, fxi, Qxi) = self.get_nonlin_pred_dynamics_int(particles=particles, u=u)
        for i in xrange(len(zl)):
            self.kf.measure_full(y=xi_next[i], z=zl[i], P=Pl[i], C=Axi[i], h_k=fxi[i], R=Qxi[i])
        
        # Predict next states conditioned on eta_next
        self.set_states(particles, xil, zl, Pl)
    
    def get_cross_covariance(self, particles, u):
        return None
    
    def calc_cond_dynamics(self, particles, xi_next, u=None):
        #Compensate for noise correlation
        N = len(particles)
        #(xil, zl, Pl) = self.get_states(particles)
        
        (Az, fz, Qz) = self.get_lin_pred_dynamics_int(particles=particles, u=u)

        Qxiz = self.get_cross_covariance(particles=particles, u=u)
        if (Qxiz == None and self.Qxiz == None):
            return (Az, fz, Qz)
        if (Qxiz == None):
            Qxiz = N*(self.Qxiz,)
        
        (Axi, fxi, Qxi) = self.get_nonlin_pred_dynamics_int(particles=particles, u=u)
        
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
        (y, Cz, hz, Rz) = self.get_meas_dynamics_int(particles=particles, y=y)
            
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
        (Az, fz, Qz) = self.get_lin_pred_dynamics_int(particles=particles, u=u)
        (Axi, fxi, Qxi) = self.get_nonlin_pred_dynamics_int(particles=particles, u=u)
        Qxiz = self.get_cross_covariance(particles=particles, u=u)
        (xil, zl, Pl) = self.get_states(particles)
        if (Qxiz == None):
            if (self.Qxiz == None):
                Qxiz = N*(numpy.zeros((Qxi[0].shape[0],Qz[0].shape[0])),)
            else:
                Qxiz = N*(self.Qxiz,)
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
        (Az, fz, Qz) = self.get_lin_pred_dynamics_int(particles=particles, u=u)
        (Axi, fxi, Qxi) = self.get_nonlin_pred_dynamics_int(particles=particles, u=u)
        Qxiz = self.get_cross_covariance(particles=particles, u=u)
        (xil, zl, Pl) = self.get_states(particles)
        if (Qxiz == None):
            if (self.Qxiz == None):
                Qxiz = N*(numpy.zeros((Qxi[0].shape[0],Qz[0].shape[0])),)
            else:
                Qxiz = N*(self.Qxiz,)
        
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
    
    def set_params(self, params):
        self.params = numpy.copy(params).reshape((-1,1))
    
    def calc_logprod_derivative(self, A, dA, B, dB):
        """ I = logdet(A)+Tr(inv(A)*B)
            dI/dx = Tr(inv(A)*(dA - dA*inv(A)*B + dB) """
            
        tmp = numpy.linalg.solve(A, B)
        tmp2 = dA + dB - dA.dot(tmp)
        return numpy.trace(numpy.linalg.solve(A,tmp2))



    def eval_logp_xi0(self, xil):
        """ Evaluate logprob of the initial non-linear state eta,
            default implementation assumes all are equal, override this
            if another behavior is desired """
        return numpy.zeros((len(xil)))
    
#    def eval_grad_eta0_logpdf(self, eta):
#        """ Evaluate logprob of the initial non-linear state eta,
#            default implementation assumes all are equal, override this
#            if another behavior is desired """
#        return numpy.zeros(self.params.shape)


    def calc_l1(self, z, P, z0, P0):
        z0_diff = z - z0
        l1 = z0_diff.dot(z0_diff.T) + P
        return l1
        
    def eval_logp_x0(self, particles):
        """ Calculate gradient of a term of the I1 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""    
            
        # Calculate l1 according to (19a)
        N = len(particles)
        (xil, zl, Pl) = self.get_states(particles)
        lpxi0 = self.eval_logp_xi0(xil)
        lpz0 = numpy.empty(N)
        for i in xrange(N):
            (z0, P0) = self.get_rb_initial([xil[i],])
            l1 = self.calc_l1(zl[i], Pl[i], z0, P0)
            (_tmp, ld) = numpy.linalg.slogdet(P0)
            tmp = numpy.linalg.solve(P0, l1)
            lpz0[i] = -0.5*(ld + numpy.trace(tmp))
        return lpxi0 + lpz0
    
#    def eval_logp_x0_grad(self, z0, P0, diff_z0, diff_P0):
#        # Calculate l1 according to (19a)
#        l1 = self.calc_l1(z0, P0)
#        
#        (_tmp, ld) = numpy.linalg.slogdet(self.Q)
#        tmp = numpy.linalg.solve(P0, l1)
#        val = -0.5*(ld + numpy.trace(tmp))
#        
#        grad = numpy.zeros(self.params.shape)
#        # Calculate gradient
#        for i in range(len(self.params)):
#
#            if (diff_z0 != None):
#                dl1 = -diff_z0[i].dot((self.kf.z-z0).T) - (self.kf.z-z0).dot(diff_z0[i].T)
#            else:
#                dl1 = numpy.zeros(l1.shape)
#        
#            if (diff_P0 != None): 
#                dP0 = diff_P0[i]
#            else:
#                dP0 = numpy.zeros(P0.shape)
#
#            grad[i] = -0.5*self.calc_logprod_derivative(P0, dP0, l1, dl1)
#        return (val + self.eval_eta0_logpdf(self.eta),
#                grad + self.eval_grad_eta0_logpdf(self.eta))
    
    
    def calc_l2(self, xin, zn, Pn, z, P, Axi, Az, f, M):
        xn = numpy.vstack((xin, zn))
        A = numpy.vstack((Axi, Az))
        predict_err = xn - f - A.dot(z)
        
        l2 = predict_err.dot(predict_err.T) +A.dot(P).dot(A.T)
        
        tmp = -Axi.dot(M)
        l2[len(xin):,:len(xin)] += tmp.T
        l2[:len(xin),len(xin):] += tmp
        
        tmp2 = Pn - M.T.dot(P) - Az.dot(M)        

        l2[len(xin):,len(xin):] += tmp2
        return (l2, predict_err)
    
       
    def eval_logp_xnext(self, particles, x_next, u, Mzl):
        """ Calculate gradient of a term of the I2 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""
        # Calculate l2 according to (16)
        N = len(particles)
        lpxn = numpy.empty(N)
        
        (_xi, z, P) = self.get_states(particles)
        (xin, zn, Pn) = self.get_states(x_next)
        
        (Az, fz, Qz) = self.get_lin_pred_dynamics_int(particles=particles, u=u)
        (Axi, fxi, Qxi) = self.get_nonlin_pred_dynamics_int(particles=particles, u=u)
        Qxiz = self.get_cross_covariance(particles=particles, u=u)
        if (Qxiz == None):
            if (self.Qxiz == None):
                Qxiz = N*(numpy.zeros((Qxi[0].shape[0],Qz[0].shape[0])),)
            else:
                Qxiz = N*(self.Qxiz,)
        
        for i in xrange(N):
            f = numpy.vstack((fxi[i], fz[i]))
            Q = numpy.vstack((numpy.hstack((Qxi[i], Qxiz[i])),
                              numpy.hstack((Qxiz[i].T, Qz[i]))))
            (l2, _pe) = self.calc_l2(xin[i], zn[i], Pn[i], z[i], P[i], Axi[i], Az[i], f, Mzl[i])
            (_tmp, ld) = numpy.linalg.slogdet(Q)
            tmp = numpy.linalg.solve(Q, l2)
            lpxn[i] = -0.5*(ld + numpy.trace(tmp))
      
        return lpxn

    def calc_diff_l2(self, x_next):
        
        (l2, predict_err) = self.calc_l2(x_next)
        A = self.A
        
        diff_l2 = list()
        
        for i in range(len(self.params)):
            diff_l2_i = numpy.zeros(l2.shape)
            grad_f = numpy.zeros((len(self.eta)+len(self.kf.z),1))
            if (self.grad_fe != None):
                grad_f[:len(self.eta)] = self.grad_fe[i]
            if (self.grad_fz != None):
                grad_f[len(self.eta):] = self.grad_fz[i]
                    
            grad_A = numpy.zeros(A.shape)
            if (self.grad_Ae != None):
                grad_A[:len(self.eta),:] = self.grad_Ae[i]
            if (self.grad_Az != None):
                grad_A[len(self.eta):,:] = self.grad_Az[i]
                    
            tmp = (grad_f + grad_A.dot(self.kf.z)).dot(predict_err.T)
            diff_l2_i = -tmp - tmp.T
            tmp = grad_A.dot(self.kf.P).dot(A.T)
            diff_l2_i += tmp + tmp.T
            tmp = -grad_A.dot(self.kf.M)
            diff_l2_i[:,len(self.eta):] +=  tmp
            diff_l2_i[len(self.eta):, :] += tmp.T
        
            diff_l2.append(diff_l2_i)
            
        return (l2,diff_l2)

#    def eval_logp_xnext_grad(self, x_next):
#        """ Calculate gradient of a term of the I2 integral approximation
#            as specified in [1].
#            The gradient is an array where each element is the derivative with 
#            respect to the corresponding parameter"""
#        # Calculate l2 according to (16)
#        (l2, diff_l2) = self.calc_diff_l2(x_next)
#      
#        (_tmp, ld) = numpy.linalg.slogdet(self.Q)
#        tmp = numpy.linalg.solve(self.Q, l2)
#        val = -0.5*(ld + numpy.trace(tmp))
#      
#        # Calculate gradient
#        grad = numpy.zeros(self.params.shape)
#        for i in range(len(self.params)):
#            
#            grad_Q = numpy.zeros(self.Q.shape)
#            
#            if (self.grad_Qe != None or 
#                self.grad_Qez != None or 
#                self.grad_Qz != None): 
#            
#                if (self.grad_Qe != None):
#                    grad_Q[:len(self.eta),:len(self.eta)] = self.grad_Qe[i]
#                if (self.grad_Qez != None):
#                    grad_Q[:len(self.eta),len(self.eta):] = self.grad_Qez[i]
#                    grad_Q[len(self.eta):,:len(self.eta)] = self.grad_Qez[i].T
#                if (self.grad_Qz != None):
#                    grad_Q[len(self.eta):, len(self.eta):] = self.grad_Qz[i]
#
#            
#                            
#            grad[i] = -0.5*self.calc_logprod_derivative(self.Q, grad_Q, l2, diff_l2[i])
#                
#        return (val, grad)
    
    def eval_logp_y(self, particles, y):
        """ Calculate a term of the I3 integral approximation
        and its gradient as specified in [1]"""
        N = len(particles)
        (y, Cz, hz, Rz) = self.get_meas_dynamics_int(particles, y)
        (xil, zl, Pl) = self.get_states(particles)
        logpy = numpy.empty(N)
        for i in xrange(N):
        # Calculate l3 according to (19b)
            meas_diff = self.kf.measurement_diff(y,z=zl[i], C=Cz[i], h_k=hz[i]) 
            l3 = meas_diff.dot(meas_diff.T)
            l3 += Cz[i].dot(Pl[i]).dot(Cz[i].T)
        
            (_tmp, ld) = numpy.linalg.slogdet(Rz[i])
            tmp = numpy.linalg.solve(Rz[i], l3)
            logpy[i] = -0.5*(ld + numpy.trace(tmp))

        return logpy

    def calc_l3(self, y):
        meas_diff = self.kf.measurement_diff(y,C=self.kf.C, h_k=self.kf.h_k) 
        l3 = meas_diff.dot(meas_diff.T)
        l3 += self.kf.C.dot(self.kf.P).dot(self.kf.C.T)
        return l3
        
#    def eval_logp_y_grad(self, y):
#        """ Calculate a term of the I3 integral approximation
#        and its gradient as specified in [1]"""
#        
#        # For later use
#        R = self.kf.R
#        grad_R = self.grad_R
#        # Calculate l3 according to (19b)
#        l3 = self.calc_l3(y)
#        
#        (_tmp, ld) = numpy.linalg.slogdet(R)
#        tmp = numpy.linalg.solve(R, l3)
#        val = -0.5*(ld + numpy.trace(tmp))
#
#        # Calculate gradient
#        grad = numpy.zeros(self.params.shape)
#        for i in range(len(self.params)):
#            
#            dl3 = numpy.zeros(l3.shape)
#            if (self.grad_C != None):
#                meas_diff = self.kf.measurement_diff(y,C=self.kf.C, h_k=self.kf.h_k) 
#                tmp2 = self.grad_C[i].dot(self.kf.P).dot(self.kf.C.T)
#                tmp = self.grad_C[i].dot(self.kf.z).dot(meas_diff.T)
#                if (self.grad_h != None):
#                    tmp += self.grad_h[i].dot(meas_diff.T)
#                dl3 += -tmp -tmp.T + tmp2 + tmp2.T
#
#            if (grad_R != None): 
#                dR = grad_R[i]
#            else:
#                dR = numpy.zeros(R.shape)
#
#            grad[i] = -0.5*self.calc_logprod_derivative(R, dR, l3, dl3)
#
#        return (val, grad)

#class LTV(part_utils.RBPSBase, param_est.ParamEstInterface):
class LTV(RBPSBase):
    """ Base class for particles of the type linear time varying with additive gaussian noise.

        Implement this type of system by extending this class and provide the methods for returning 
        the system matrices at each time instant  """
    def __init__(self, z0, P0, A=None, C=None, Q=None,
             R=None, f=None, h=None, params=None, t0=0):
        super(LTV, self).__init__(z0=z0, P0=P0,
                                  Az=A, C=C, 
                                  Qz=Q, R=R,
                                  h_k=h, f_k=f, t0=t0)
        
        self.z0 = numpy.copy(z0)
        self.P0 = numpy.copy(P0)
        self.t0 = t0

    def create_initial_estimate(self, N):
        particles = numpy.empty((N,), dtype=numpy.ndarray)
        lz = len(self.z0)
        dim = lz + len(self.P0.ravel())
        
        for i in xrange(N):
            particles[i] = numpy.empty(dim)
            particles[i][:lz] = numpy.copy(self.z0).ravel()
            particles[i][lz:] = numpy.copy(self.P0).ravel()  
        return particles
    
    def set_states(self, particles, _xi_list, z_list, P_list):
        """ Set the estimate of the Rao-Blackwellized states """
        lz = len(self.z0)
        N = len(particles)
        for i in xrange(N):
            particles[i][:lz] = z_list[i].ravel()
            particles[i][lz:] = P_list[i].ravel()
 
    def get_states(self, particles):
        """ Return the estimate of the Rao-Blackwellized states.
            Must return two variables, the first a list containing all the
            expected values, the second a list of the corresponding covariance
            matrices"""
        N = len(particles)
        xil = list()
        zl = list()
        Pl = list()
        N = len(particles)
        lz = len(self.z0)
        for i in xrange(N):
            xil.append(None)
            zl.append(particles[i][:lz].reshape(-1,1))
            Pl.append(particles[i][lz:].reshape(self.P0.shape))
        
        return (xil, zl, Pl)

    def update(self, particles, u, noise):
        """ Update estimate using noise as input """
        # Update linear estimate with data from measurement of next non-linear
        # state 
        (_xil, zl, Pl) = self.get_states(particles)
        (Az, fz, Qz) = self.get_lin_pred_dynamics_int(particles, u)
        for i in xrange(len(zl)):
            # Predict z_{t+1}
            (zl[i], Pl[i]) = self.kf.predict_full(zl[i], Pl[i], Az[i], fz[i], Qz[i])
        
        # Predict next states conditioned on eta_next
        self.set_states(particles, None, zl, Pl)
        self.t = self.t + 1.0
    
    def measure(self, y, particles):
        """ Return the log-pdf value of the measurement """
        
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
            
        lyz = numpy.empty((len(particles)))
        for i in xrange(len(zl)):
            # Predict z_{t+1}
            lyz[i] = self.kf.measure_full(y, zl[i], Pl[i], Cz[i], hz[i], Rz[i])
        
        return lyz
    
    def next_pdf(self, particles, next_cpart, u=None):
        """ Return the log-pdf value for the possible future state 'next' given input u """
        N = len(particles)
        return numpy.zeros((N,))
    
    def sample_smooth(self, particle, next_part, u):
        """ Update ev. Rao-Blackwellized states conditioned on "next_part" """
        (xil, zl, Pl) = self.get_states(particle)
        if (next_part != None):
            (Az, fz, Qz) = self.get_lin_pred_dynamics_int(particle, u)
            self.kf.measure_full(next_part[1].reshape((-1,1)), zl[0], Pl[0], C=Az[0], h_k=fz[0], R=Qz[0])
        
        xi = copy.copy(xil[0])
        z = numpy.random.multivariate_normal(zl[0].ravel(), Pl[0])
            
        return (None, z)

    def fwd_peak_density(self, u=None):
        return 0.0

    def calc_l1(self, z, P, z0, P0):
        z0_diff = z - z0
        l1 = z0_diff.dot(z0_diff.T) + P
        return l1

    def eval_logp_x0(self, particles):
        """ Calculate gradient of a term of the I1 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""    
        # Calculate l1 according to (19a)
        N = len(particles)
        (xil, zl, Pl) = self.get_states(particles)
        lpz0 = numpy.empty(N)
        for i in xrange(N):
            (z0, P0) = self.get_rb_initial([xil[i],])
            l1 = self.calc_l1(zl[i], Pl[i], z0, P0)
            (_tmp, ld) = numpy.linalg.slogdet(P0)
            tmp = numpy.linalg.solve(P0, l1)
            lpz0[i] = -0.5*(ld + numpy.trace(tmp))
        return lpz0
    
    def calc_l2(self, zn, Pn, z, P, A, f, M):
        predict_err = zn - f - A.dot(z)
        AM = A.dot(M)
        l2 = predict_err.dot(predict_err.T)
        l2 += Pn + A.dot(P).dot(A.T) - AM.T - AM
        return (l2, A, M, predict_err)    
        
    def eval_logp_xnext(self, x_next):
        """ Calculate gradient of a term of the I2 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""
        # Calculate l2 according to (16)    
        (xil, zl, Pl) = self.get_states(particles)
        
        (l2, A, M_ext, predict_err) = self.calc_l2(x_next)
      
        Q = self.kf.Q
        (_tmp, ld) = numpy.linalg.slogdet(Q)
        tmp = numpy.linalg.solve(Q, l2)
        val = -0.5*(ld + numpy.trace(tmp))
        
        z = self.kf.z
        P = self.kf.P
        
        # Calculate gradient
        grad = numpy.zeros(self.params.shape)
        for i in range(len(self.params)):
            tmp = numpy.zeros((len(l2), 1))
            
            grad_Q = numpy.zeros(Q.shape)
            
            if (self.grad_Q != None):
                grad_Q = self.grad_Q[i]

            diff_l2 = numpy.zeros(l2.shape)
            grad_f = numpy.zeros((len(z),1))
            if (self.grad_f != None):
                grad_f = self.grad_f[i]
                    
            grad_A = numpy.zeros(A.shape)
            if (self.grad_A != None):
                grad_A = self.grad_A[i]
                    
            tmp = (grad_f + grad_A.dot(z)).dot(predict_err.T)
            tmp2 = grad_A.dot(M_ext)
            tmp3 = grad_A.dot(P).dot(A.T)
            diff_l2 = -tmp - tmp.T -tmp2 - tmp2.T + tmp3 + tmp3.T
                
            grad[i] = -0.5*self.calc_logprod_derivative(Q, grad_Q, l2, diff_l2)
                
        return (val, grad)
    
    def calc_l3(self, y):
        P = self.kf.P
        # P = self.P_tN
        meas_diff = self.kf.measurement_diff(y,C=self.kf.C, h_k=self.kf.h_k) 
        l3 = meas_diff.dot(meas_diff.T)
        l3 += self.kf.C.dot(P).dot(self.kf.C.T)
        return l3
    
    def eval_logp_y(self, y):
        """ Calculate a term of the I3 integral approximation
        and its gradient as specified in [1]"""

        # For later use
        R = self.kf.R
        grad_R = self.grad_R
        # Calculate l3 according to (19b)
        l3 = self.calc_l3(y)
        (_tmp, ld) = numpy.linalg.slogdet(R)
        tmp = numpy.linalg.solve(R, l3)
        val = -0.5*(ld + numpy.trace(tmp))
                    
        # Calculate gradient
        grad = numpy.zeros(self.params.shape)
        for i in range(len(self.params)):
            
            dl3 = numpy.zeros(l3.shape)
            if (self.grad_C != None):
                meas_diff = self.kf.measurement_diff(y,C=self.kf.C, h_k=self.kf.h_k) 
                tmp2 = self.grad_C[i].dot(self.kf.P).dot(self.kf.C.T)
                tmp = self.grad_C[i].dot(self.kf.z).dot(meas_diff)
                if (self.grad_h != None):
                    tmp += self.grad_h[i]
                dl3 += -tmp -tmp.T + tmp2 + tmp2.T
        
            if (grad_R != None): 
                dR = grad_R[i]
            else:
                dR = numpy.zeros(R.shape)

            grad[i] = -0.5*self.calc_logprod_derivative(R, dR, l3, dl3)
                
        return (val, grad)