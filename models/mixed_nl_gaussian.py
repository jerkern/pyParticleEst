""" Class for mixed linear/non-linear models with additive gaussian noise """

import numpy
import kalman
import part_utils
import param_est

class MixedNLGaussianCollapsed(object):
    """ Stores collapsed sample of MixedNLGaussian object """
    def __init__(self, parent):
        self.eta = parent.get_nonlin_state().ravel()            
        tmpp = numpy.random.multivariate_normal(numpy.ravel(parent.kf.z),
                                                parent.kf.P)
        self.z = tmpp.ravel()   


class MixedNLGaussian(part_utils.RBPSBase, param_est.ParamEstInterface):
    """ Base class for particles of the type mixed linear/non-linear with additive gaussian noise.
    
        Implement this type of system by extending this class and provide the methods for returning 
        the system matrices at each time instant  """
    def __init__(self, z0, P0, e0, Az=None, C=None, Qz=None, R=None, fz=None,
                 Ae=None, Qe=None, Qez=None, fe=None, h=None, params=None):
        super(MixedNLGaussian, self).__init__(z0=z0, P0=P0,
                                              Az=Az, C=C, 
                                              Qz=Qz, R=R)
        self.Ae = Ae
        self.Qe = Qe
        
        self.eta = e0

        if (Qez != None):
            self.Qez = Qez
        else:
            self.Qez = numpy.zeros((len(e0),len(z0)))
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
        self.eta += self.fe + self.Ae.dot(self.kf.z) + noise
        
        # Handle linear/non-linear noise correlation
        Sigma_a = self.Qe + self.Ae.dot(self.kf.P).dot(self.Ae.T)
        Sigma_az = self.Qez + self.Ae.dot(self.kf.P).dot(self.kf.A.T)
        
        super(MixedNLGaussian, self).update(u, noise)

        # This is what is sometimes called "the second measurement update"
        # for Rao-Blackwellized particle filters
        tmp = Sigma_az.T.dot(numpy.linalg.inv(Sigma_a))
        self.kf.z += tmp.dot(noise)
        self.kf.P -= tmp.dot(Sigma_az)

    def clin_predict(self, next_part=None):
        noise = next_part.eta - self.fe - self.Ae.dot(self.kf.z)
        Sigma_a = self.Qe + self.Ae.dot(self.kf.P).dot(self.Ae.T)
        Sigma_az = self.Qez + self.Ae.dot(self.kf.P).dot(self.kf.A.T)
        
        (z, P) = super(MixedNLGaussian, self).clin_predict(next)
        
        # This is what is sometimes called "the second measurement update"
        # for Rao-Blackwellized particle filters
        tmp = Sigma_az.T.dot(numpy.linalg.inv(Sigma_a))
        z += tmp.dot(noise)
        P -= tmp.dot(Sigma_az)
        
        return (z, P)

    def sample_process_noise(self, u): 
        """ Return sampled process noise for the non-linear states """
        Sigma_a = self.Qe + self.Ae.dot(self.kf.P).dot(self.Ae.T)
        
        return numpy.random.multivariate_normal(numpy.zeros(len(self.   eta)), Sigma_a).reshape((-1,1))

        
    def next_pdf(self, next_part, u):
        """ Implements the next_pdf function for MixedNLGaussian models """
        lin_P = self.kf.P
        z_mean = numpy.reshape(self.kf.z,(-1,1))
        
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
        super(MixedNLGaussian, self).set_dynamics(Az=Az, C=C, Qz=Qz, R=R, f_k=fz,h_k=h)

        if (Ae != None):
            self.Ae = Ae
        if (Qe != None):
            self.Qe = Qe
        if (Qez != None):
            self.Qez = Qez
        if (fe != None):
            self.fe = fe
            
    
    def calc_suff_stats(self, next_part):
        """ Implements the sample_smooth function for MixedNLGaussian models """
        # Create sample particle
        
        lin_est = self.kf.z
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
            self.z_tN = self.kf.z
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
    
    def clin_measure(self, y, next_part=None):
                # This implementation doesn't handle correlation between measurement
        # and process noise (ie, we don't need to know the next state
        yl = self.prep_clin_measure(y)
        return super(MixedNLGaussian, self).clin_measure(yl)
    
    def measure(self, y):

        return super(MixedNLGaussian, self).measure(y)
    
    def fwd_peak_density(self, u):
        """ Implements the fwd_peak_density function for MixedNLGaussian models """
        Sigma = self.calc_sigma(self.kf.P)
        zero = numpy.zeros((Sigma.shape[0],1))
        return kalman.lognormpdf(zero, zero, Sigma)
    
    def get_nonlin_state(self):
        return self.eta

    def set_nonlin_state(self,inp):
        self.eta = inp

    def set_dynamics_gradient(self, grad_Az=None, grad_fz=None, grad_Qz=None, grad_R=None,
                     grad_Ae=None, grad_fe=None, grad_Qe=None, grad_Qez=None, 
                     grad_C=None, grad_h=None):
        pass

    def eval_logp_x1(self, z0, P0):
        """ Calculate a term of the I1 integral approximation
        and its gradient as specified in [1]"""
        
        # Calcuate l1 according to (19a)
        tmp = self.kf.z - z0
        l1 = tmp.dot(tmp.T) + self.kf.P

    def eval_logp_xnext(self, x_next):
        """ Calculate a term of the I2 integral approximation
        and its gradient as specified in [1]"""
        # Calculate l2 according to (16)
        f = numpy.vstack((self.fe, self.fz))
        tmp1 = x_next - f - self.A.dot(self.kf.z)
        zero_tmp = numpy.zeros((self.A.shape[0],self.Ae.shape[0]))
        A1 = numpy.hstack((zero_tmp, self.A))
        tmp2 = A1.dot(self.M_tN)
        l2 = tmp1.dot(tmp1.T) - self.A.dot(self.kf.P).dot(self.A.T) - tmp2.T -tmp2

    def eval_logp_y(self, y):
        """ Calculate a term of the I3 integral approximation
        and its gradient as specified in [1]"""

        # Calculate l3 according to (19b)
        tmp = self.kf.measurement_diff(y) 
        l3 = tmp.dot(tmp.T)
        l3 += self.kf.C.dot(self.kf.P).dot(self.kf.C.T)
        (tmp, ld) = numpy.linalg.slogdet(self.kf.R)
        tmp = numpy.linalg.solve(self.kf.R, l3)
        return ld + numpy.trace(tmp)