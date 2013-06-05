""" Class for mixed linear/non-linear models with additive gaussian noise """

import numpy
import kalman
import part_utils

class MixedNLGaussianCollapsed(object):
    """ Stores collapsed sample of MixedNLGaussian object """
    def __init__(self, parent):
        self.eta = parent.get_nonlin_state().ravel()            
        tmpp = numpy.random.multivariate_normal(numpy.ravel(parent.kf.x_new),
                                                parent.kf.P)
        self.z = tmpp.ravel()   


class MixedNLGaussian(part_utils.RBPSBase):
    """ Base class for particles of the type mixed linear/non-linear with additive gaussian noise.
    
        Implement this type of system by extending this class and provide the methods for returning 
        the system matrices at each time instant  """
    def __init__(self, z0=None, P0=None, e0=None, Az=None, Bz=None, C=None, D=None, Qz=None, R=None,
                 Ae=None, Be=None, Qe=None, Qez=None):
        super(MixedNLGaussian, self).__init__(z0=z0, P0=P0,
                                              Az=Az, Bz=Bz,C=C, D=D, 
                                              Qz=Qz, R=R)
        self.Ae = Ae
        self.Be = Be
        self.Qe = Qe
        self.Qez = Qez
        self.eta = e0
            
    def update(self, data=None):
        super(MixedNLGaussian, self).update(data)
        
    def next_pdf(self, next_cpart, u):
        """ Implements the next_pdf function for MixedNLGaussian models """
        lin_P = self.kf.P
        z_mean = numpy.reshape(self.kf.x_new,(-1,1))
        
        _lin_cov = self.kf.Q
        nonlin_est = numpy.reshape(self.get_nonlin_state(),(-1,1))
        
        x = numpy.hstack((next_cpart.eta,next_cpart.z)).reshape((-1,1))
        A = self.kf.A
        
        # TODO, handle non-linear dependence!
        lin_est = A.dot(z_mean)
        if (u != None):
            lin_est += self.kf.B.dot(u)
        st = numpy.vstack((nonlin_est,lin_est))
        Sigma = self.calc_sigma(lin_P)
        return kalman.lognormpdf(x,mu=st,S=Sigma)
    
    def get_Q(self):
        return numpy.vstack((numpy.hstack((self.Qe, self.Qez)),
                             numpy.hstack((self.Qez.T, self.kf.Q))))
    
    def set_dynamics(self, Az=None, Bz=None, C=None, D=None, Qz=None, R=None,
                     Ae=None, Be=None, Qe=None, Qez=None):
        if (Az != None):
            self.kf.A = Az
        if (Bz != None):
            self.kf.B = Bz
        if (C != None):
            self.kf.C = C
        if (D != None):
            self.kf.D = D
        if (Qz != None):
            self.kf.Q = Qz
        if (R != None):
            self.kf.R = R
        if (Ae != None):
            self.Ae = Ae
        if (Be != None):
            self.Be = Be
        if (Qe != None):
            self.Qe = Qe
        if (Qez != None):
            self.Qez = Qez
            
    
    def sample_smooth(self, filt_traj, ind, next_cpart):
        """ Implements the sample_smooth function for MixedNLGaussian models """
        # Create sample particle
        cpart = self.collapse()
        # Extract needed input signal from trajectory  
        u = filt_traj[ind].u
        # Smooth linear states
        _etaj = cpart.eta
        
        _lin_est = self.kf.x_new
        P = self.kf.P
        A_ext = numpy.vstack((self.Ae, self.kf.A))
        _lin_cov = self.kf.Q
        Q = self.get_Q()
        QPinv = numpy.linalg.solve(Q+A_ext.dot(P.dot(A_ext.transpose())),
                                   numpy.eye(len(cpart.eta)+len(cpart.z)))
        Hi = P.dot(A_ext.transpose().dot(QPinv))
        PI = P-Hi.dot(A_ext.dot(P))
        z_mean = cpart.z
        
        B = self.kf.B

        # TODO, handle non-linear dependence!
        lin_est = self.kf.A.dot(z_mean)
        if (u != None):
            lin_est += B.dot(self.linear_input(u)).ravel()
        ytmp = numpy.hstack((cpart.eta,lin_est))
        
        mu = z_mean + numpy.ravel(Hi.dot(numpy.hstack((next_cpart.eta,next_cpart.z)) - ytmp))
        
        cpart.z = numpy.random.multivariate_normal(mu,PI).ravel()

        return cpart
    
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
    
    def collapse(self):
        """ collapse the object by sampling all the states """
        return MixedNLGaussianCollapsed(self)

    @classmethod
    def from_collapsed(cls, collapsed):
        pass

    def set_lin_est(self, est):
        self.kf.x_new = est[0]
        self.kf.P = est[1]
        
    def get_lin_est(self):
        return (self.kf.x_new, self.kf.P)
