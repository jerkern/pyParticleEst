""" Class for mixed linear/non-linear models with additive gaussian noise """

import numpy
import kalman
import part_utils
import param_est


#class LTV(part_utils.RBPSBase, param_est.ParamEstInterface):
class LTV(part_utils.RBPSBase):
    """ Base class for particles of the type linear time varying with additive gaussian noise.

        Implement this type of system by extending this class and provide the methods for returning 
        the system matrices at each time instant  """
    def __init__(self, z0, P0, A=None, C=None, Q=None,
             R=None, f=None, h=None, params=None, t0=0):
        super(LTV, self).__init__(z0=z0, P0=P0,
                                  Az=A, C=C, 
                                  Qz=Q, R=R,
                                  h_k=h, f_k=f)
        
        self.z0 = numpy.copy(z0)
        self.P0 = numpy.copy(P0)
        self.t0 = t0
        self.t = self.t0

    def calc_next_eta(self, u, noise):
        """ Update non-linear state using sampled noise,
        # the noise term here includes the uncertainty from z """
        return self.t+1
    
    def get_nonlin_state(self):
        return self.t

    def set_nonlin_state(self,inp):
        self.t = numpy.copy(inp)

    def meas_eta_next(self, eta_next):
        return 0
    
    def cond_dynamics(self, eta_next):
        return

    def sample_process_noise(self, u):
        """ Return process noise for input u """
        return None

    def next_pdf(self, next_cpart, u):
        """ Return the log-pdf value for the possible future state 'next' given input u """
        return (next_cpart.t == self.t+1)
    
    def sample_smooth(self, next_part):
        """ Update ev. Rao-Blackwellized states conditioned on "next_part" """
        return
    
    def fwd_peak_density(self, u=None):
        return 1.0
    
    def get_z0_initial(self):
        return (self.z0, self.P0)