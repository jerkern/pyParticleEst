""" Class for mixed linear/non-linear models with additive gaussian noise """

import numpy
import math
import kalman
import part_utils

class MixedNLGaussianCollapsed(object):
    """ Stores collapsed sample of MixedNLGaussian object """
    def __init__(self, parent):
        self.eta = parent.get_nonlin_state().ravel()            
        (lin_est,lin_P) = parent.get_lin_est()
        tmpp = numpy.random.multivariate_normal(numpy.ravel(lin_est),lin_P)
        self.z = tmpp.ravel()   


class MixedNLGaussian(part_utils.ParticleSmoothingBaseRB):
    """ Base class for particles of the type mixed linear/non-linear with additive gaussian noise.
    
        Implement this type of system by extending this class and provide the methods for returning 
        the system matrices at each time instant  """
        
    def next_pdf(self, next, u):
        """ Implements the next_pdf function for MixedNLGaussian models """
        (lin_est,lin_P) = self.get_lin_est()
        z_mean = numpy.reshape(lin_est,(-1,1))
        lin_cov = self.get_lin_Q()
        nonlin_est = numpy.reshape(self.get_nonlin_state(),(-1,1))
        
        x = numpy.hstack((next.eta,next.z)).reshape((-1,1))
        A = self.get_lin_A() 
        B = self.get_lin_B()
        
        # TODO, handle non-linear dependence!
        lin_est = A.dot(z_mean) + B.dot(self.linear_input(u))
        st = numpy.vstack((nonlin_est,lin_est))
        Sigma = self.calc_sigma(lin_P)
        prob = self.normpdf(x,st,Sigma)
        return prob
    
    def sample_smooth(self, filt_traj, ind, next_cpart):
        """ Implements the sample_smooth function for MixedNLGaussian models """
        # Create sample particle
        cpart = self.collapse()
        # Extract needed input signal from trajectory  
        u = filt_traj[ind].u
        # Smooth linear states
        etaj = cpart.eta
        (lin_est,P) = self.get_lin_est()
        A_ext = self.get_full_A()
        lin_cov = self.get_lin_Q()
        Q = self.get_Q()
        QPinv = numpy.linalg.solve(Q+A_ext.dot(P.dot(A_ext.transpose())),
                                   numpy.eye(len(cpart.eta)+len(cpart.z)))
        Hi = P.dot(A_ext.transpose().dot(QPinv))
        self.PI = P-Hi.dot(A_ext.dot(P))
        z_mean = cpart.z
        
        B = self.get_lin_B()

        # TODO, handle non-linear dependence!
        lin_est = self.get_lin_A().dot(z_mean)
        if (u != None):
            lin_est += B.dot(self.linear_input(u)).ravel()
        ytmp = numpy.hstack((cpart.eta,lin_est))
        
        mu = z_mean + numpy.ravel(Hi.dot(numpy.hstack((next_cpart.eta,next_cpart.z)) - ytmp))
        
        cpart.z = numpy.random.multivariate_normal(mu,self.PI).ravel()

        return cpart
    
    def calc_sigma(self, P):
        cov = self.get_Q()
        A_ext = self.get_full_A()
        #B_ext = self.get_full_B()
        lin_P_ext = A_ext.dot(P.dot(A_ext.transpose()))
        Sigma = cov + lin_P_ext
        return Sigma
    
    def fwd_peak_density(self, u):
        """ Implements the fwd_peak_density function for MixedNLGaussian models """
        Sigma = self.calc_sigma(self.get_lin_est()[1])
        nx = len(Sigma)
        tmp = math.pow(2*math.pi, nx)*numpy.linalg.det(Sigma)
        return 1.0 / math.sqrt(tmp)
    
    def collapse(self):
        """ collapse the object by sampling all the states """
        return MixedNLGaussianCollapsed(self)
    
    def clin_update(self, u):
        """ Implements the clin_update function for MixedNLGaussian models """
        A = self.get_lin_A()
        B = self.get_lin_B()
        C = self.get_lin_C()
        Q = self.get_lin_Q()
        R = self.get_R()
        (x0, P) = self.get_lin_est()

        kf = kalman.KalmanFilter(A,B,C, numpy.reshape(x0,(-1,1)), P, Q, R)
        kf.time_update(u=u)
        
        return (kf.x_new.reshape((-1,1)), kf.P)
    
    def clin_measure(self, y):
        """ Implements the clin_measure function for MixedNLGaussian models """
        A = self.get_lin_A()
        B = self.get_lin_B()
        C = self.get_lin_C()
        Q = self.get_lin_Q()
        R = self.get_R()
        (x0, P) = self.get_lin_est()

        kf = kalman.KalmanFilter(A,B,C, numpy.reshape(x0,(-1,1)), P, Q, R)
        kf.meas_update(y)
        
        return (kf.x_new.reshape((-1,1)), kf.P)

    def clin_smooth(self, z_next, u):
        """ Implements the clin_smooth function for MixedNLGaussian models """
        A = self.get_lin_A()
        B = self.get_lin_B()
        C = self.get_lin_C()
        Q = self.get_lin_Q()
        R = self.get_R()
        (x0, P) = self.get_lin_est()

        kf = kalman.KalmanSmoother(A,B,C, numpy.reshape(x0,(-1,1)), P, Q, R)
        kf.smooth(z_next[0], z_next[1], u)
        
        return (kf.x_new.reshape((-1,1)), kf.P)

    
    def normpdf(self,x,mu,sigma):
        """ Calculate gaussian probability density of x, when x ~ N(mu,sigma) """
        Sinv = numpy.linalg.solve(sigma,numpy.eye(sigma.shape[0]))
        u = (x-mu).transpose().dot(Sinv)
        e = numpy.dot(u,x-mu)
        y = numpy.exp(-0.5*e) 
        return y

    def get_R(self):
        raise NotImplementedError( "Should have implemented this" )
    
    def get_lin_A(self):
        raise NotImplementedError( "Should have implemented this" )
    
    def get_lin_B(self):
        raise NotImplementedError( "Should have implemented this" )

    def get_lin_C(self):
        raise NotImplementedError( "Should have implemented this" )

    def get_nonlin_B(self):
        raise NotImplementedError( "Should have implemented this" )
    
    def get_full_B(self):
        raise NotImplementedError( "Should have implemented this" )

    def get_nonlin_A(self):
        raise NotImplementedError( "Should have implemented this" )
    
    def get_full_A(self):
        raise NotImplementedError( "Should have implemented this" )
    
    def get_lin_Q(self):
        raise NotImplementedError( "Should have implemented this" )
    
    def get_Q(self):
        raise NotImplementedError( "Should have implemented this" )
