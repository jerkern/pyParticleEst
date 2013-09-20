""" Class for mixed linear/non-linear models with additive gaussian noise """

import numpy
import kalman
import part_utils
import param_est


class MixedNLGaussian(part_utils.RBPSBase, param_est.ParamEstInterface):
    """ Base class for particles of the type mixed linear/non-linear with additive gaussian noise.
    
        Implement this type of system by extending this class and provide the methods for returning 
        the system matrices at each time instant  """
    def __init__(self, z0, P0, e0, Az=None, C=None, Qz=None, R=None, fz=None,
                 Ae=None, Qe=None, Qez=None, fe=None, h=None, params=None):
        super(MixedNLGaussian, self).__init__(z0=z0, P0=P0,
                                              Az=Az, C=C, 
                                              Qz=Qz, R=R,
                                              h_k=h, f_k=fz)
        self.Ae = Ae
        self.Qe = Qe
        # Store a copy of these variables, needed in clin_dynamics
        self.Qz = numpy.copy(self.kf.Q)
        self.fz = numpy.copy(self.kf.f_k)
        
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
#   TODO     tmp = self.Qez.T.dot(numpy.linalg.inv(self.Qe))
        tmp = numpy.zeros((len(self.kf.z),len(self.eta)))
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
    
    def get_Q(self):
        return numpy.vstack((numpy.hstack((self.Qe, self.Qez)),
                             numpy.hstack((self.Qez.T, self.Qz))))
    

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
        c = Sigma.dot(Wa.dot(next_part.eta-self.fe)-Wz.dot(self.kf.f_k)+numpy.linalg.inv(P).dot(lin_est))
       
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
    
    def eta0_logpdf(self, eta):
        """ Evaluate logprob of the initial non-linear state eta,
            default implementation assumes all are equal, override this
            if another behavior is desired """
        return 0.0

    def calc_logprod_derivative(self, A, dA, B, dB):
        """ I = logdet(A)+Tr(inv(A)*B)
            dI/dx = Tr(inv(A)*(dA - dA*inv(A)*B + dB) """
            
        tmp = numpy.linalg.solve(A, B)
        tmp2 = dA + dB - dA.dot(tmp)
        return numpy.trace(numpy.linalg.solve(A,tmp2))

    def eval_logp_x0(self, z0, P0, diff_z0, diff_P0):
        """ Calculate a term of the I1 integral approximation
        and its gradient as specified in [1]"""
        
        # Calculate l1 according to (19a)
        z0_diff = self.kf.z - z0
        l1 = z0_diff.dot(z0_diff.T) + self.kf.P
       
        Q = self.get_Q()
        (_tmp, ld) = numpy.linalg.slogdet(Q)
        tmp = numpy.linalg.solve(P0, l1)
        val = -0.5*(ld + numpy.trace(tmp)) + self.eta0_logpdf(self.eta)
        grad = numpy.zeros(self.params.shape)
        # Calculate gradient
        for i in range(len(self.params)):

            if (diff_z0 != None):
                dl1 = -diff_z0[i].dot((self.kf.z-z0).T) - (self.kf.z-z0).dot(diff_z0[i].T)
            else:
                dl1 = numpy.zeros(l1.shape)
        
            if (diff_P0 != None): 
                dP0 = diff_P0[i]
            else:
                dP0 = numpy.zeros(P0.shape)
                tmp = numpy.zeros((len(l1), 1))

            # TODO, prob. of eta0 can also depend on parameters!
            grad[i] = -0.5*self.calc_logprod_derivative(P0, dP0, l1, dl1)
        return (val, grad)

            
        
    def eval_logp_xnext(self, x_next):
        """ Calculate a term of the I2 integral approximation
        and its gradient as specified in [1]"""
        # Calculate l2 according to (16)
            
        x_kplus = numpy.vstack((x_next.eta, x_next.kf.z))
        f = numpy.vstack((self.fe, self.fz))
        # TODO this A could have been changed!
        A = numpy.vstack((self.Ae, self.kf.A))
        predict_err = x_kplus - f - A.dot(self.kf.z)
        zero_tmp = numpy.zeros((len(self.kf.z),len(self.eta)))
        M_ext = numpy.hstack((zero_tmp, self.M_tN))
        AM_ext = A.dot(M_ext)
        l2 = predict_err.dot(predict_err.T)
        l2 += A.dot(self.kf.P).dot(A.T) - AM_ext.T -AM_ext
      
        Q = self.get_Q()
        (_tmp, ld) = numpy.linalg.slogdet(Q)
        tmp = numpy.linalg.solve(Q, l2)
        val = -0.5*(ld + numpy.trace(tmp))
        
        # Calculate gradient
        grad = numpy.zeros(self.params.shape)
        for i in range(len(self.params)):
            tmp = numpy.zeros((len(l2), 1))
            
            grad_Q = numpy.zeros(Q.shape)
            
            if (self.grad_Qe != None or 
                self.grad_Qez != None or 
                self.grad_Qz != None): 
            
                if (self.grad_Qe != None):
                    grad_Q[:len(self.eta),:len(self.eta)] = self.grad_Qe[i]
                if (self.grad_Qez != None):
                    grad_Q[:len(self.eta),len(self.eta):] = self.grad_Qez[i]
                    grad_Q[len(self.eta):,:len(self.eta)] = self.grad_Qez[i].T
                if (self.grad_Qz != None):
                    grad_Q[len(self.eta):, len(self.eta):] = self.grad_Qz[i]


            diff_l2 = numpy.zeros(l2.shape)
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
            tmp2 = grad_A.dot(M_ext)
            tmp3 = grad_A.dot(self.kf.P).dot(A.T)
            diff_l2 = -tmp - tmp.T -tmp2 - tmp2.T + tmp3 + tmp3.T
                
            grad[i] = -0.5*self.calc_logprod_derivative(Q, grad_Q, l2, diff_l2)
                
        return (val, grad)
    
    def eval_logp_y(self, y):
        """ Calculate a term of the I3 integral approximation
        and its gradient as specified in [1]"""

        # For later use
        R = self.kf.R
        grad_R = self.grad_R
        # Calculate l3 according to (19b)
        meas_diff = self.kf.measurement_diff(y,C=self.kf.C, h_k=self.kf.h_k) 
        l3 = meas_diff.dot(meas_diff.T)
        l3 += self.kf.C.dot(self.kf.P).dot(self.kf.C.T)
        
        (_tmp, ld) = numpy.linalg.slogdet(R)
        tmp = numpy.linalg.solve(R, l3)
        val = -0.5*(ld + numpy.trace(tmp))
        
        # Calculate gradient
        grad = numpy.zeros(self.params.shape)
        for i in range(len(self.params)):
            
            dl3 = numpy.zeros(l3.shape)
            if (self.grad_C != None):
                tmp2 = self.grad_C[i].dot(self.kf.P).dot(self.kf.C.T)
                tmp = self.grad_C[i].dot(self.kf.z)
                if (self.grad_h != None):
                    tmp += self.grad_h[i]
                dl3 += -tmp -tmp.T + tmp2 + tmp2.T
        
            if (grad_R != None): 
                dR = grad_R[i]
            else:
                dR = numpy.zeros(R.shape)

            grad[i] = -0.5*self.calc_logprod_derivative(R, dR, l3, dl3)
                
        return (val, grad)