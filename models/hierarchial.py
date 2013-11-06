""" Class for hierarchial models"""

import numpy
import kalman
import part_utils

class Hierarchial(part_utils.RBPSBase):
    """ Base class for particles of the type hierarchial which have a linear/guassian system conditioned on the
        non-linear states, with no effect from the linear states on the non-linear """

    def __init__(self, z0, P0, e0, Az=None, C=None, Qz=None, R=None, fz=None,
                 Ae=None, Qe=None, Qez=None, fe=None, h=None, params=None):
        super(Hierarchial, self).__init__(z0=z0, P0=P0,
                                              Az=Az, C=C, 
                                              Qz=Qz, R=R,
                                              h_k=h, f_k=fz)
        self.Ae = Ae
        self.Qe = Qe
        # Store a copy of these variables, needed in clin_dynamics
        self.Qz = numpy.copy(self.kf.Q)
        self.fz = numpy.copy(self.kf.f_k)
        
        # Sore z0, P0 needed for default implementation of 
        # get_z0_initial and get_grad_z0_initial
        self.z0 = numpy.copy(z0)
        self.P0 = numpy.copy(P0)
        
        self.eta = numpy.copy(e0.reshape((-1,1)))

        if (Qez != None):
            self.Qez = Qez
        else:
            self.Qez = numpy.zeros((len(e0),len(z0)))
        if (fe != None):
            self.fe = fe
        else:
            self.fe = numpy.zeros((len(self.eta), 1))

        
        self.sampled_z = None
        self.z_tN = None
        self.P_tN = None
        self.M_tN = None
        
        self.params = None
        # Lists of element-wise derivatives, e.g self.grad_Az[0] is the 
        # element-wise derivative of Az with respect to the first parameter
        self.grad_Az = None
        self.grad_fz = None
        self.grad_Ae = None
        self.grad_fe = None
        self.grad_Qe = None
        self.grad_Qez = None 
        self.grad_Qz = None
        self.grad_C = None
        self.grad_h = None
        self.grad_R = None
        
    def set_dynamics(self, Az=None, fz=None, Qz=None, R=None,
                     Ae=None, fe=None, Qe=None, Qez=None, 
                     C=None, h=None):
        super(MixedNLGaussian, self).set_dynamics(Az=Az, C=C, Qz=Qz, R=R, f_k=fz,h_k=h)

        if (Ae != None):
            self.Ae = numpy.copy(Ae)
        if (Qe != None):
            self.Qe = numpy.copy(Qe)
        if (Qez != None):
            self.Qez = numpy.copy(Qez)
        if (Qz != None):
            self.Qz = numpy.copy(self.kf.Q)
        if (fz != None):
            self.fz = numpy.copy(self.kf.f_k)
        if (fe != None):
            self.fe = numpy.copy(fe)
            
    def calc_next_eta(self, u, noise):
        """ Update non-linear state using sampled noise,
        # the noise term here includes the uncertainty from z """
        noise = numpy.reshape(noise, (-1,1))
        eta = self.pred_eta() + noise
        return eta

    def pred_eta(self):
        return self.fe + self.Ae.dot(self.kf.z)
    
    def meas_eta_next(self, eta_next):
        """ Update estimate using observation of next state """
        # This is what is sometimes called "the second measurement update"
        # for Rao-Blackwellized particle filters
        return self.kf.measure_full(y=eta_next, h_k=self.fe,
                                    C=self.Ae, R=self.Qe)
    
    def calc_cond_dynamics(self, eta_next):
        #Compensate for noise correlation
        tmp = self.Qez.T.dot(numpy.linalg.inv(self.Qe))
        A_cond = self.kf.A - tmp.dot(self.Ae)
        offset = tmp.dot(eta_next - self.fe)
        return (A_cond, self.fz + offset)
    
    def cond_dynamics(self, eta_next):
        """ Condition dynamics on future state 'eta_next'. """
        (Az, fz) = self.calc_cond_dynamics(eta_next)
        self.kf.set_dynamics(f_k=fz, A=Az)

    def cond_predict(self, eta_next=None):
        #Compensate for noise correlation
        (Az, fz) = self.calc_cond_dynamics(eta_next)
        return self.kf.predict_full(f_k=fz, A=Az, Q=self.Qz)



    def sample_process_noise(self, u=None): 
        """ Return sampled process noise for the non-linear states """
        Sigma_a = self.Qe + self.Ae.dot(self.kf.P).dot(self.Ae.T)
        
        return numpy.random.multivariate_normal(numpy.zeros(len(self.eta)), Sigma_a).reshape((-1,1))

        
    def next_pdf(self, next_part, u):
        """ Implements the next_pdf function for MixedNLGaussian models """
        
        #nonlin_est = numpy.reshape(self.get_nonlin_state(),(-1,1))
        eta_est = self.pred_eta()
        x_next = numpy.vstack((next_part.eta,next_part.sampled_z)).reshape((-1,1))
        
        z_est = self.kf.predict()[0]
        x_est = numpy.vstack((eta_est,z_est))
        Sigma = self.calc_sigma()
        return kalman.lognormpdf(x_next,mu=x_est,S=Sigma)
    

    def sample_smooth(self, next_part):
        """ Implements the sample_smooth function for MixedNLGaussian models """
        if (next_part != None):
            (self.z_tN, self.P_tN, self.M_tN) = self.calc_suff_stats(next_part)
        else:
            self.z_tN = self.kf.z
            self.P_tN = self.kf.P

        self.sampled_z = numpy.random.multivariate_normal(self.z_tN.ravel(),self.P_tN).ravel().reshape((-1,1))
    
    def calc_sigma(self):
        cov = self.get_Q()
        A_ext = numpy.vstack((self.Ae, self.kf.A))
        lin_P_ext = A_ext.dot(self.kf.P.dot(A_ext.transpose()))
        Sigma = cov + lin_P_ext
        return Sigma

    def measure(self, y):
        y=numpy.reshape(y, (-1,1))
        return super(MixedNLGaussian, self).measure(y)
    
    def fwd_peak_density(self, u):
        """ Implements the fwd_peak_density function for MixedNLGaussian models """
        Sigma = self.calc_sigma()
        zero = numpy.zeros((Sigma.shape[0],1))
        return kalman.lognormpdf(zero, zero, Sigma)
    
    def get_nonlin_state(self):
        return self.eta

    def set_nonlin_state(self,inp):
        self.eta = numpy.copy(inp)

    def set_dynamics_gradient(self, grad_Az=None, grad_fz=None, grad_Qz=None, grad_R=None,
                     grad_Ae=None, grad_fe=None, grad_Qe=None, grad_Qez=None, 
                     grad_C=None, grad_h=None):
        """ Lists of element-wise derivatives """
        if (grad_Az != None):
            self.grad_Az = grad_Az
        if (grad_fz !=None):
            self.grad_fz = grad_fz
        if (grad_Ae != None):
            self.grad_Ae = grad_Ae
        if (grad_fe != None):
            self.grad_fe = grad_fe
        if (grad_Qe != None):
            self.grad_Qe = grad_Qe
        if (grad_Qez != None):
            self.grad_Qez = grad_Qez 
        if (grad_Qz != None):
            self.grad_Qz = grad_Qz
        if (grad_C != None):
            self.grad_C = grad_C
        if (grad_h != None):
            self.grad_h = grad_h
        if (grad_R != None):
            self.grad_R = grad_R
    
    def set_params(self, params):
        self.params = numpy.copy(params).reshape((-1,1))
    