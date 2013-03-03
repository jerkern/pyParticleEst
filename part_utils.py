""" Collection of functions and classes used for Particle Filtering/Smoothing """
import numpy
import math
import copy
import kalman
import PF
import PS

class ParticleFilteringBase(object):
    """ Base class for particles to be used with particle filtering """
    def sample_input_noise(self, u):
        """ Return a noise perturbed input vector u """
        raise NotImplementedError( "Should have implemented this" )
    
    def update(self, data):
        raise NotImplementedError( "Should have implemented this" )
        
    def measure(self, y):
        raise NotImplementedError( "Should have implemented this" )



class ParticleSmoothingBase(ParticleFilteringBase):
    """ Base class for particles to be used with particle smoothing """
    def next_pdf(self, next, u):
        """ Return the probability density value for the possible future state 'next' given input u """
        raise NotImplementedError( "Should have implemented this" )
    
    def sample_smooth(self, filt_traj, ind, next_cpart):
        """ Return a collapsed particle with the rao-blackwellized states sampled """
        raise NotImplementedError( "Should have implemented this" )
    
    def collapse(self):
        """ Return a sample of the particle where the rao-blackwellized states
        are drawn from the MVN that results from CLGSS structure """
        
        raise NotImplementedError( "Should have implemented this" )
    
class ParticleSmoothingBaseRB(ParticleSmoothingBase):
    
    def clin_update(self, u):
        """ Kalman update of the linear states conditioned on the non-linear trajectory estimate """
        raise NotImplementedError( "Should have implemented this" )
    
    def clin_measure(self, y):
        """ Kalman measuement of the linear states conditioned on the non-linear trajectory estimate """
        raise NotImplementedError( "Should have implemented this" )

    def clin_smooth(self, z_next, u):
        """ Kalman smoothing of the linear states conditioned on the next particles linear states """ 
        raise NotImplementedError( "Should have implemented this" )

    def set_nonlin_state(self, eta):
        """ Set the non-linear state estimates """
        raise NotImplementedError( "Should have implemented this" )
    
    def set_lin_est(self, lest):
        """ Set the estimate of the rao-blackwellized states """
        raise NotImplementedError( "Should have implemented this" )
 
    def linear_input(self, u):
        """ Extract the part of u affect the conditionally rao-blackwellized states """
        raise NotImplementedError( "Should have implemented this" )    
    
    
class MixedNLGaussianCollapsed(object):
    """ Stores collapsed sample of MixedNLGaussian object """
    def __init__(self, parent):
        self.eta = parent.get_nonlin_state().ravel()            
        (lin_est,lin_P) = parent.get_lin_est()
        tmpp = numpy.random.multivariate_normal(numpy.ravel(lin_est),lin_P)
        self.z = tmpp.ravel()   

class MixedNLGaussian(ParticleSmoothingBaseRB):
    """ Base class for particles of the type mixed linear/non-linear with gaussian noise """
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



