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
                 Ae=None, Qe=None, Qez=None, fe=None, h=None, params=None, t0=0):
        super(MixedNLGaussian, self).__init__(z0=z0, P0=P0,
                                              Az=Az, C=C, 
                                              Qz=Qz, R=R,
                                              h_k=h, f_k=fz,
                                              t0=t0)
        self.Ae = numpy.copy(Ae)
        self.Az = numpy.copy(Az)
        self.A = numpy.vstack((self.Ae, self.Az))
        self.Qe = Qe
        # Store a copy of these variables, needed in clin_dynamics
        self.Qz = numpy.copy(self.kf.Q)
        self.fz = numpy.copy(self.kf.f_k)
        
        self.eta = numpy.copy(e0.reshape((-1,1)))

        self.x_zeros = numpy.zeros((len(self.eta)+len(self.kf.z), 1))

        if (Qez != None):
            self.Qez = numpy.copy(Qez)
        else:
            self.Qez = numpy.zeros((len(e0),len(z0)))

        self.Q = numpy.vstack((numpy.hstack((self.Qe, self.Qez)),
                               numpy.hstack((self.Qez.T, self.Qz))))
        
        if (fe != None):
            self.fe = fe
        else:
            self.fe = numpy.zeros((len(self.eta), 1))

        self.Sigma = None
        
        self.sampled_z = None
        self.z_tN = None
        self.P_tN = None
        self.M_tN = None
        if (params != None):
            self.params = numpy.copy(params)
        else:
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
        if (Az != None):
            self.Az = numpy.copy(Az)
        if (Ae != None or Az != None):
            if (Ae == None):
                Ae = self.Ae
            if (Az == None):
                Az = self.Az
            self.A = numpy.vstack((Ae, Az))
        if (Qe != None):
            self.Qe = numpy.copy(Qe)
        if (Qez != None):
            self.Qez = numpy.copy(Qez)
        if (Qz != None):
            self.Qz = numpy.copy(self.kf.Q)
        if (Qe != None or Qez != None or Qz != None):
            if (Qe == None):
                Qe = self.Qe
            if (Qez == None):
                Qez = self.Qez
            if (Qz == None):
                Qz = self.Qz
            self.Q = numpy.vstack((numpy.hstack((Qe, Qez)),
                               numpy.hstack((Qez.T, Qz))))
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

    def fwd_peak_density(self, u):
        """ Implements the fwd_peak_density function for MixedNLGaussian models """
        A = self.A
        self.Sigma = self.Q + A.dot(self.kf.P).dot(A.T)
        return kalman.lognormpdf(self.x_zeros, self.x_zeros, self.Sigma)
        
    def next_pdf(self, next_part, u):
        """ Implements the next_pdf function for MixedNLGaussian models """
        
        #nonlin_est = numpy.reshape(self.get_nonlin_state(),(-1,1))
        eta_est = self.pred_eta()
        
        eta_diff = next_part.eta - eta_est
        #z_diff= next_part.sampled_z - self.kf.predict()[0]
        z_diff= next_part.sampled_z - self.cond_predict(eta_est)[0]
        
        if (self.Sigma == None):
            self.fwd_peak_density(u) 
        
        diff = numpy.vstack((eta_diff, z_diff)).reshape((-1,1))
        # We can used cached self.Sigma since 'fwd_peak_density' will always be called first
        #Sigma = self.Q + A.dot(self.kf.P).dot(A.T)
        return kalman.lognormpdf(diff,mu=self.x_zeros,S=self.Sigma)
    
    def sample_smooth(self, next_part):
        """ Implements the sample_smooth function for MixedNLGaussian models """
        if (next_part != None):
            self.meas_eta_next(next_part.eta)
            self.cond_dynamics(next_part.eta)
            self.kf.smooth(next_part.kf.z, next_part.kf.P)

        self.sampled_z = numpy.random.multivariate_normal(self.kf.z.ravel(),self.kf.P).ravel().reshape((-1,1))
    
    def measure(self, y):
        y=numpy.reshape(y, (-1,1))
        return super(MixedNLGaussian, self).measure(y)
    
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
    
    def calc_logprod_derivative(self, A, dA, B, dB):
        """ I = logdet(A)+Tr(inv(A)*B)
            dI/dx = Tr(inv(A)*(dA - dA*inv(A)*B + dB) """
            
        tmp = numpy.linalg.solve(A, B)
        tmp2 = dA + dB - dA.dot(tmp)
        return numpy.trace(numpy.linalg.solve(A,tmp2))

    def calc_l1(self, z0, P0):
        z0_diff = self.kf.z - z0
        l1 = z0_diff.dot(z0_diff.T) + self.kf.P
        return l1

    def eval_eta0_logpdf(self, eta):
        """ Evaluate logprob of the initial non-linear state eta,
            default implementation assumes all are equal, override this
            if another behavior is desired """
        return 0.0
    
    def eval_grad_eta0_logpdf(self, eta):
        """ Evaluate logprob of the initial non-linear state eta,
            default implementation assumes all are equal, override this
            if another behavior is desired """
        return numpy.zeros(self.params.shape)
    
    # Default implementation, for when initial state is independent 
    # of parameters
    def get_z0_initial(self):
        return (self.z0, self.P0)

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
        
        (_tmp, ld) = numpy.linalg.slogdet(self.Q)
        tmp = numpy.linalg.solve(P0, l1)
        val = -0.5*(ld + numpy.trace(tmp)) 
       
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

            grad[i] = -0.5*self.calc_logprod_derivative(P0, dP0, l1, dl1)
        return (val + self.eval_eta0_logpdf(self.eta),
                grad + self.eval_grad_eta0_logpdf(self.eta))
    
    
    def calc_l2(self, x_next):
        x_kplus = numpy.vstack((x_next.eta, x_next.kf.z))
        f = numpy.vstack((self.fe, self.fz))

        A = self.A
        predict_err = x_kplus - f - A.dot(self.kf.z)
        
        l2 = predict_err.dot(predict_err.T) +A.dot(self.kf.P).dot(A.T)
        
        tmp = -self.Ae.dot(self.kf.M)
        l2[len(self.eta):,:len(self.eta)] += tmp.T
        l2[:len(self.eta),len(self.eta):] += tmp
        
        tmp2 = x_next.kf.P - self.kf.M.T.dot(self.kf.P) - self.kf.A.dot(self.kf.M)        

        l2[len(self.eta):,len(self.eta):] += tmp2
        return (l2, predict_err)
    
    def calc_diff_l2(self, x_next):
        
        (l2, predict_err) = self.calc_l2(x_next)
        A = self.A
        
        diff_l2 = list()
        
        for i in range(len(self.params)):
            diff_l2_i = numpy.zeros(l2.shape)
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
            diff_l2_i = -tmp - tmp.T
            tmp = grad_A.dot(self.kf.P).dot(A.T)
            diff_l2_i += tmp + tmp.T
            tmp = -grad_A.dot(self.kf.M)
            diff_l2_i[:,len(self.eta):] +=  tmp
            diff_l2_i[len(self.eta):, :] += tmp.T
        
            diff_l2.append(diff_l2_i)
            
        return (l2,diff_l2)
        
    def eval_logp_xnext(self, x_next):
        """ Calculate gradient of a term of the I2 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""
        # Calculate l2 according to (16)
        (l2, diff_l2) = self.calc_diff_l2(x_next)
      
        (_tmp, ld) = numpy.linalg.slogdet(self.Q)
        tmp = numpy.linalg.solve(self.Q, l2)
        val = -0.5*(ld + numpy.trace(tmp))
      
        # Calculate gradient
        grad = numpy.zeros(self.params.shape)
        for i in range(len(self.params)):
            
            grad_Q = numpy.zeros(self.Q.shape)
            
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

            
                            
            grad[i] = -0.5*self.calc_logprod_derivative(self.Q, grad_Q, l2, diff_l2[i])
                
        return (val, grad)
    
    def calc_l3(self, y):
        meas_diff = self.kf.measurement_diff(y,C=self.kf.C, h_k=self.kf.h_k) 
        l3 = meas_diff.dot(meas_diff.T)
        l3 += self.kf.C.dot(self.kf.P).dot(self.kf.C.T)
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
                tmp = self.grad_C[i].dot(self.kf.z).dot(meas_diff.T)
                if (self.grad_h != None):
                    tmp += self.grad_h[i].dot(meas_diff.T)
                dl3 += -tmp -tmp.T + tmp2 + tmp2.T

            if (grad_R != None): 
                dR = grad_R[i]
            else:
                dR = numpy.zeros(R.shape)

            grad[i] = -0.5*self.calc_logprod_derivative(R, dR, l3, dl3)

        return (val, grad)