""" Class for hierarchial models"""

import abc
import numpy
import kalman
import part_utils

class Hierarchial(part_utils.RBPSBase):
    """ Base class for particles of the type hierarchial which have a linear/guassian system conditioned on the
        non-linear states, with no effect from the linear states on the non-linear """

    def __init__(self, z0, P0, e0, Az=None, C=None, Qz=None, 
                 R=None, fz=None, h=None, params=None):
        super(Hierarchial, self).__init__(z0=z0, P0=P0,
                                              Az=Az, C=C, 
                                              Qz=Qz, R=R,
                                              h_k=h, f_k=fz)
       
        # Sore z0, P0 needed for default implementation of 
        # get_z0_initial and get_grad_z0_initial
        self.z0 = numpy.copy(z0)
        self.P0 = numpy.copy(P0)
        
        self.eta = numpy.copy(numpy.asarray(e0).reshape((-1,1)))

        self.sampled_z = None
        
    def meas_eta_next(self, eta_next):
        """ Update estimate using observation of next state """
        return
    
    def calc_cond_dynamics(self, eta_next):
        return
    
    def cond_dynamics(self, eta_next):
        """ Condition dynamics on future state 'eta_next'. """
        return

    def cond_predict(self, eta_next=None):
        return self.kf.predict()

    @abc.abstractmethod
    def next_eta_pdf(self, next_part, u):
        pass

    def next_pdf(self, next_part, u):
        """ Implements the next_pdf function for Hierarchial models """
        
        log_etapdf = self.next_eta_pdf(next_part, u)
        (zn, Pn) = self.kf.predict()
        return log_etapdf + kalman.lognormpdf(next_part.sampled_z,mu=zn,S=Pn)
    

    def sample_smooth(self, next_part):
        """ Implements the sample_smooth function for MixedNLGaussian models """
        if (next_part != None):
            self.kf.smooth(next_part.kf.z, next_part.kf.P)

        self.sampled_z = numpy.random.multivariate_normal(self.kf.z.ravel(),self.kf.P).ravel().reshape((-1,1))
    
    @abc.abstractmethod
    def split_measure(self, y):
        """ Split measurement into linear/non-liner part """
        pass
    
    @abc.abstractmethod
    def measure_nl(self, y_nl):
        pass

    def clin_measure(self, y, next_part=None):
        (_y_nl, y_l) = self.split_measure(y)
        self.kf.measure(y_l)

    def measure(self, y):
        # Split measurement into linear/non-liner part
        (y_nl, y_l) = self.split_measure(y)
        logp_nl = 0.0
        logp_l = 0.0
        
        # Non-linear measure
        if (y_nl != None):
            logp_nl = self.measure_nl(y_nl)
        # Linear measure
        if (y_l != None):
            logp_l = super(Hierarchial, self).measure(y_l)
        return logp_nl +logp_l 
    
    @abc.abstractmethod
    def fwd_peak_density_eta(self, u):
        pass
    
    def fwd_peak_density(self, u):
        """ Implements the fwd_peak_density function for MixedNLGaussian models """
        (zn, Pn) = self.kf.predict()
        return kalman.lognormpdf(zn, zn, Pn) + self.fwd_peak_density_eta(u)
    
    def get_nonlin_state(self):
        return self.eta

    def set_nonlin_state(self,inp):
        self.eta = numpy.copy(inp)
