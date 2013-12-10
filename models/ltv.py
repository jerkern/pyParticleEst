""" Class for mixed linear/non-linear models with additive gaussian noise """

import numpy
import part_utils
import param_est


#class LTV(part_utils.RBPSBase, param_est.ParamEstInterface):
class LTV(part_utils.RBPSBase, param_est.ParamEstInterface):
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
        
        self.params = None
        self.grad_A = None
        self.grad_f = None
        self.grad_Q = None
        self.grad_C = None
        self.grad_h = None
        self.grad_R = None
        
        self.z_tN = None
        self.P_tN = None
        self.M_tN = None

    def calc_next_eta(self, u, noise):
        """ Update non-linear state using sampled noise,
        # the noise term here includes the uncertainty from z """
        return self.t+1
    
    def get_nonlin_state(self):
        return self.t

    def set_nonlin_state(self,inp):
        self.t = numpy.copy(inp)
        
    def set_dynamics(self, A=None, C=None, Q=None, R=None, f_k=None, h_k=None):
        super(LTV, self).set_dynamics(Az=A, C=C, Qz=Q, R=R, f_k=f_k, h_k=h_k)
        
    def set_dynamics_gradient(self, grad_A=None, grad_f=None, grad_Q=None, grad_R=None,
                              grad_Qez=None, grad_C=None, grad_h=None):
        """ Lists of element-wise derivatives """
        if (grad_A != None):
            self.grad_A = grad_A
        if (grad_f !=None):
            self.grad_f = grad_f
        if (grad_Q != None):
            self.grad_Q = grad_Q
        if (grad_C != None):
            self.grad_C = grad_C
        if (grad_h != None):
            self.grad_h = grad_h
        if (grad_R != None):
            self.grad_R = grad_R
            
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
    
    def set_params(self, params):
        self.params = numpy.copy(params).reshape((-1,1))
    
    def calc_logprod_derivative(self, A, dA, B, dB):
        """ I = logdet(A)+Tr(inv(A)*B)
            dI/dx = Tr(inv(A)*(dA - dA*inv(A)*B + dB) """
            
        tmp = numpy.linalg.solve(A, B)
        tmp2 = dA + dB - dA.dot(tmp)
        return numpy.trace(numpy.linalg.solve(A,tmp2))

    def calc_l1(self, z0, P0):
        z = self.kf.z
        P = self.kf.P
        #z = self.z_tN
        #P = self.P_tN
        
        z0_diff = z - z0
        l1 = z0_diff.dot(z0_diff.T) + P
        return l1

    # Default implementation, for when initial state is independent 
    # of parameters    
    def get_grad_z0_initial(self):
        grad_z0 = []
        grad_P0 = []
        for _i in range(len(self.params)):
            grad_z0.append(numpy.zeros(self.z0.shape))
            grad_P0.append(numpy.zeros(self.P0.shape))
            
        return (grad_z0, grad_P0)
        
    def eval_logp_x0(self, z0, P0, diff_z0, diff_P0):
        """ Calculate gradient of a term of the I1 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""    
            
        # Calculate l1 according to (19a)
        l1 = self.calc_l1(z0, P0)
        
        Q = self.kf.Q
        (_tmp, ld) = numpy.linalg.slogdet(Q)
        tmp = numpy.linalg.solve(P0, l1)
        val = -0.5*(ld + numpy.trace(tmp)) 
       
        grad = numpy.zeros(self.params.shape)
        
        z = self.kf.z
        #z = self.z_tN
        
        # Calculate gradient
        for i in range(len(self.params)):

            if (diff_z0 != None):
                dl1 = -diff_z0[i].dot((z-z0).T) - (z-z0).dot(diff_z0[i].T)
            else:
                dl1 = numpy.zeros(l1.shape)
        
            if (diff_P0 != None): 
                dP0 = diff_P0[i]
            else:
                dP0 = numpy.zeros(P0.shape)

            grad[i] = -0.5*self.calc_logprod_derivative(P0, dP0, l1, dl1)
        return (val, grad)
    
    
    def calc_l2(self, x_next):
        z = self.kf.z
        zn = x_next.kf.z
        P = self.kf.P
        Pn = x_next.kf.P
        #z = self.z_tN
        #zn = x_next.z_tN
        #P = self.P_tN
        #Pn = x_next.P_tN
        
        f = self.kf.f_k
        # TODO this A could have been changed!
        A = self.kf.A
        predict_err = zn - f - A.dot(z)
        M = self.kf.M
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