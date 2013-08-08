""" Class for mixed linear/non-linear models with additive gaussian noise """

import numpy
import kalman
import part_utils
import param_est

class MixedNLGaussianCollapsed(object):
    """ Stores collapsed sample of MixedNLGaussian object """
    def __init__(self, parent):
        self.eta = parent.get_nonlin_state().ravel()            
        tmpp = numpy.random.multivariate_normal(numpy.ravel(parent.kf.x_new),
                                                parent.kf.P)
        self.z = tmpp.ravel()   


class MixedNLGaussian(part_utils.RBPSBase, param_est.ParamEstInterface):
    """ Base class for particles of the type mixed linear/non-linear with additive gaussian noise.
    
        Implement this type of system by extending this class and provide the methods for returning 
        the system matrices at each time instant  """
    def __init__(self, z0=None, P0=None,  Az=None, C=None, Qz=None, R=None, fz=None,
                 e0=None, Ae=None, Qe=None, Qez=None, fe=None, h=None, params=None):
        super(MixedNLGaussian, self).__init__(z0=z0, P0=P0,
                                              Az=Az, C=C, 
                                              Qz=Qz, R=R)
        self.Ae = Ae
        self.Qe = Qe
        self.Qez = Qez
        self.eta = e0
        if (fe != None):
            self.fe = fe
        else:
            self.fe = numpy.zeros((len(self.eta), 1))
        if (fz != None):
            self.fz = fz
        else:
            self.fz = numpy.zeros((len(z0), 1))
        if (h != None):
            self.h = h
        else:
            self.h = numpy.zeros((self.kf.C.shape[0],1))
        
        self.sampled_z = None
        self.z_tN = None
        self.P_tN = None
        self.M_tN = None
        
        # Element-wise matrix derivates needed for parameters estimation
        self.grad_Ae = None
        self.grad_Qe = None
        self.grad_Qez = None
        self.grad_eta = None
        self.grad_fe = None
        self.grad_fz = None
        
    def update(self, u, noise):
        
        # Update non-linear state using sampled noise
        self.eta += self.fe + self.Ae.dot(self.kf.x_new) + noise
        
        # Handle linear/non-linear noise correlation
        Sigma_a = self.Qe + self.Ae.dot(self.kf.P).dot(self.Ae.T)
        Sigma_az = self.Qez + self.Ae.dot(self.kf.P).dot(self.kf.A.T)
        Sigma_z = self.kf.Q + self.kf.A.dot(self.kf.P).dot(self.kf.A.T)
        
        tmp = Sigma_az.T.dot(numpy.linalg.inv(Sigma_a))
        
        self.kf.x_new = self.fz + self.kf.A.dot(self.kf.x_new) + tmp.dot(noise)
        self.kf.P = Sigma_z - tmp.dot(Sigma_az)
        

    def sample_process_noise(self, u): 
        """ Return sampled process noise for the non-linear states """
        Sigma_a = self.Qe + self.Ae.dot(self.kf.P).dot(self.Ae.T)
        
        return numpy.random.multivariate_normal(numpy.zeros(len(self.   eta)), Sigma_a).reshape((-1,1))

        
    def next_pdf(self, next_part, u):
        """ Implements the next_pdf function for MixedNLGaussian models """
        lin_P = self.kf.P
        z_mean = numpy.reshape(self.kf.x_new,(-1,1))
        
        _lin_cov = self.kf.Q
        nonlin_est = numpy.reshape(self.get_nonlin_state(),(-1,1))
        
        x = numpy.vstack((next_part.eta,next_part.sampled_z)).reshape((-1,1))
        
        lin_est = self.kf.A.dot(z_mean)+self.fz
        st = numpy.vstack((nonlin_est,lin_est))
        Sigma = self.calc_sigma(lin_P)
        return kalman.lognormpdf(x,mu=st,S=Sigma)
    
    def get_Q(self):
        return numpy.vstack((numpy.hstack((self.Qe, self.Qez)),
                             numpy.hstack((self.Qez.T, self.kf.Q))))
    
    def set_dynamics(self, Az=None, fz=None, Qz=None, R=None,
                     Ae=None, fe=None, Qe=None, Qez=None, 
                     C=None, h=None):
        if (Az != None):
            self.kf.A = Az
        if (C != None):
            self.kf.C = C
        if (Qz != None):
            self.kf.Q = Qz
        if (R != None):
            self.kf.R = R
        if (Ae != None):
            self.Ae = Ae
        if (Qe != None):
            self.Qe = Qe
        if (Qez != None):
            self.Qez = Qez
        if (fe != None):
            self.fe = fe
        if (fz != None):
            self.fz = fz
            
    
    def calc_suff_stats(self, next_part):
        """ Implements the sample_smooth function for MixedNLGaussian models """
        # Create sample particle
        
        lin_est = self.kf.x_new
        P = self.kf.P
        A_ext = numpy.vstack((self.Ae, self.kf.A))
        _lin_cov = self.kf.Q
        Q = self.get_Q()
        W = A_ext.T.dot(numpy.linalg.inv(Q))
        el = len(self.eta)
        Wa = W[:,:el]
        Wz = W[:,el:]
        QPinv = numpy.linalg.inv(Q+A_ext.dot(P.dot(A_ext.transpose())))
        Sigma = P-P.dot(A_ext.transpose().dot(QPinv)).dot(A_ext.dot(P))
        c = Sigma.dot(Wa.dot(next_part.eta-self.fe)-Wz.dot(self.fz)+numpy.linalg.inv(P).dot(lin_est))
        
        z_tN = Sigma.dot(Wz.dot(next_part.z_tN))+c
        M_tN = Sigma.dot(Wz.dot(next_part.P_tN))
        P_tN = Sigma+M_tN.T.dot(Sigma)
        
        return (z_tN, P_tN, M_tN)
    
 
    
    def sample_smooth(self, next_part):
        """ Implements the sample_smooth function for MixedNLGaussian models """
        if (next_part != None):
            (self.z_tN, self.P_tN, self.M_tN) = self.calc_suff_stats(next_part)
        else:
            self.z_tN = self.kf.x_new
            self.P_tN = self.kf.P

        self.sampled_z = numpy.random.multivariate_normal(self.z_tN.ravel(),self.P_tN).ravel().reshape((-1,1))
    
    def calc_sigma(self, P):
        cov = self.get_Q()
        A_ext = numpy.vstack((self.Ae, self.kf.A))
        lin_P_ext = A_ext.dot(P.dot(A_ext.transpose()))
        Sigma = cov + lin_P_ext
        return Sigma
    
    def prep_clin_measure(self, y):
        return y
    
    def clin_measure(self, y):
        yl = self.prep_clin_measure(y)
        return super(MixedNLGaussian, self).clin_measure(yl)
    
    def fwd_peak_density(self, u):
        """ Implements the fwd_peak_density function for MixedNLGaussian models """
        Sigma = self.calc_sigma(self.kf.P)
        zero = numpy.zeros((Sigma.shape[0],1))
        return kalman.lognormpdf(zero, zero, Sigma)
    
    def set_lin_est(self, est):
        self.kf.x_new = est[0]
        self.kf.P = est[1]
        
    def get_lin_est(self):
        return (self.kf.x_new, self.kf.P)

    def get_nonlin_state(self):
        return self.eta

    def set_nonlin_state(self,inp):
        self.eta = inp

    def param_change(self, Az=None, fz=None, Qz=None, R=None,
                     Ae=None, fe=None, Qe=None, Qez=None, 
                     C=None, h=None, grad_Az=None, grad_fz=None, grad_Qz=None, grad_R=None,
                     grad_Ae=None, grad_fe=None, grad_Qe=None, grad_Qez=None, 
                     grad_C=None, grad_h=None):
        """ Update all values and gradients according to a new set of parameters  
        grad_* params are tuples where the i:th entry is the element-wise derivative with
        respect to the i:th parameter"""

        self.set_dynamics(Az=Az, fz=fz, Qz=Qz, R=R,
                     Ae=Ae, fe=fe, Qe=Qe, Qez=Qez, 
                     C=C, h=h)
        # Calculate values of all cached variables
        
        # Calcuate l1 according to (19a)
        self.z1 = None #FIXME
        tmp = self.kf.x_new - self.z1
        l1 = tmp.dot(tmp.T) + self.kf.P
        # Calculate l2 according to (16)
        tmp1 = x_next - f - self.A.dot(self.kf.x_new)
        zero_tmp = numpy.zeros((self.A.shape[0],self.Ae.shape[0]))
        A1 = numpy.hstack((zero_tmp, self.A))
        tmp2 = A1.dot(self.M_tN)
        l2 = tmp.dot(tmp.T) - self.A.dot(self.kf.P).dot(self.A.T) - tmp2.T -tmp2
        # Calculate l3 according to (19b)
        y = None #FIXME
        tmp = (y-self.h-self.kf.C.dot(self.kf.x_new))
        l3 = tmp.dot(tmp.T) + self.kf.C.dot(self.kf.P).dot(self.kf.C.T)
        
    @property
    def logp_xo(self):
        """ Calculate a term of the I1 integral as specified in [1]"""
        return self.cached_logp_xo
    
    @property
    def logp_xnext(self):
        """ Calculate a term of the I2 integral as specified in [1]"""
        return self.cached_logp_xnext
    
    @property
    def logp_y(self):
        """ Calculate a term of the I3 integral as specified in [1]"""
        return self.cached_logp_xnet

    @property
    def grad_logp_xo(self):
        """ Calculate gradient of a term of the I1 integral,
            as specified in [1]"""
        return self.cached_grad_logp_xo
    
    @property
    def grad_logp_xnext(self):
        """ Calculate gradient of a term of the I2 integral,
            as specified in [1]"""
        return self.cached_grad_logp_xnext
    
    @property
    def grad_logp_y(self):
        """ Calculate gradient of a term of the the I3 integral,
            as specified in [1]"""
        return self.cached_grad_logp_y