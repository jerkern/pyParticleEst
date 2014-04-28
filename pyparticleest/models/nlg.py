""" Collection of functions and classes used for Particle Filtering/Smoothing """
import pyparticleest.part_utils as part_utils
import scipy.linalg
import numpy.random
import math
import abc
import pyximport
pyximport.install(inplace=True)
import pyparticleest.models.mlnlg_compute as mlnlg_compute
import pyparticleest.kalman as kalman



class NonlinearGaussian(part_utils.FFBSiRSInterface):
    """ Base class for particles of the type mixed linear/non-linear with additive gaussian noise.
    
        Implement this type of system by extending this class and provide the methods for returning 
        the system matrices at each time instant  """
        
    __metaclass__ = abc.ABCMeta
    
    def get_f(self, particles, u, t):
        None

    def get_Q(self, particles, u, t):
        return None
    
    def get_g(self, particles, t):
        return None
    
    def get_R(self, particles, t):
        return None
    
    def __init__(self, lxi, f=None, g=None, Q=None, R=None):
        if (f != None):
            self.f = numpy.copy(f)
        else:
            self.f = None
        if (g != None):
            self.g = numpy.copy(g)
        else:
            self.g = None
        if (Q != None):
            self.Q = numpy.copy(Q)
        else:
            self.Q = None
        if (R != None):
            self.R = numpy.copy(R)
        else:
            self.R = None
            
        self.lxi = lxi
        
    def sample_process_noise(self, particles, u, t): 
        """ Return sampled process noise for the non-linear states """
        N = len(particles)
        Q = self.get_Q(particles=particles, u=u, t=t)
        noise = numpy.random.normal(size=(self.lxi,N))
        if (Q == None):
            Q = self.Q
            Qchol = numpy.triu(scipy.linalg.cho_factor(Q)[0])
            noise = Qchol.dot(noise)
        else:
            for i in xrange(N):
                Qchol = numpy.triu(scipy.linalg.cho_factor(Q[i])[0])
                noise[:,i] =  Qchol.dot(noise[:,i])
           
        return noise.T

    def update(self, particles, u, t, noise):
        """ Update estimate using 'data' as input """
        N = len(particles)
        f = self.get_f(particles=particles, u=u, t=t)
        if (f == None):
            f = self.f
            particles += f + noise
        else:
            for i in xrange(N):
                particles[i] += f[i] + noise[i]
        return particles

    def measure(self, particles, y, t):
        """ Return the log-pdf value of the measurement """
        g = self.get_g(particles=particles, t=t)
        R = self.get_R(particles=particles, t=t)
        N = len(particles)
        lpy = numpy.empty(N)
        if (g == None and R == None):
            g = self.g
            R = self.R
            Rchol = scipy.linalg.cho_factor(self.R)
            lp = kalman.lognormpdf_cho(y-self.g,Rchol)
            for i in xrange(N):
                lpy[i] = lp
        elif (R == None):
            Rchol = scipy.linalg.cho_factor(self.R)
            for i in xrange(N):
                lpy[i] = kalman.lognormpdf_cho(y-g[i],Rchol)
        elif (g == None):
            diff = y-self.g
            for i in xrange(N):
                Rchol = scipy.linalg.cho_factor(R[i])
                lpy[i] = kalman.lognormpdf_cho(diff,Rchol)
        else:
            for i in xrange(N):
                Rchol = scipy.linalg.cho_factor(R[i])
                lpy[i] = kalman.lognormpdf_cho(y-g[i],Rchol)
            
        return lpy
    
    def eval_1st_stage_weights(self, particles, u, y, t):
        part = numpy.copy(particles)
        xin = self.pred_xi(part, u, t)
        self.cond_predict(part, xin, u, t)
        return self.measure(part, y, t)
        
    def next_pdf_max(self, particles, u, t):
        pass
        
    def next_pdf(self, particles, next_part, u, t):
        """ Implements the next_pdf function for MixedNLGaussian models """
        pass
    
    def sample_smooth(self, particles, next_part, u, t):
        """ Implements the sample_smooth function for MixedNLGaussian models """
        return particles
    
    def copy_ind(self, particles, new_ind):
        new_part = particles[new_ind]
        return new_part
    
    def set_params(self, params):
        self.params = numpy.copy(params).reshape((-1,1))

#    def get_pred_dynamics_grad(self, particles, u, t):
#        """ Override this method if (A, f, Q) depends on the parameters """
#        return (None, None, None)
#    
#    def get_meas_dynamics_grad(self, particles, y, t):
#        """ Override this method if (C, h, R) depends on the parameters """
#        return (None, None, None)
    
#    def eval_logp_x0(self, particles, t):
#        """ Calculate gradient of a term of the I1 integral approximation
#            as specified in [1].
#            The gradient is an array where each element is the derivative with 
#            respect to the corresponding parameter"""    
#            
#        # Calculate l1 according to (19a)
#        N = len(particles)
#        (xil, zl, Pl) = self.get_states(particles)
#        (z0, P0) = self.get_rb_initial(xil)
#        lpxi0 = self.eval_logp_xi0(xil)
#        lpz0 = 0.0
#        for i in xrange(N):
#            l1 = self.calc_l1(zl[i], Pl[i], z0[i], P0[i])
#            P0cho = scipy.linalg.cho_factor(P0[i], check_finite=False)
#            #(_tmp, ld) = numpy.linalg.slogdet(P0[i])
#            ld = numpy.sum(numpy.log(numpy.diagonal(P0cho[0])))*2
#            tmp = scipy.linalg.cho_solve(P0cho, l1, check_finite=False)
#            lpz0 -= 0.5*(ld + numpy.trace(tmp))
#        return lpxi0 + lpz0
#    
#    def eval_logp_x0_val_grad(self, particles, t):
#        lpz0_grad = numpy.zeros(self.params.shape)
#        
#        # Calculate l1 according to (19a)
#        N = len(particles)
#        (xil, zl, Pl) = self.get_states(particles)
#        (z0, P0) = self.get_rb_initial(xil)
#        (z0_grad, P0_grad) = self.get_rb_initial_grad(xil)
#        lpxi0 = self.eval_logp_xi0(xil)
#        lpxi0_grad = self.eval_logp_xi0_grad(xil)
#        lpz0 = 0.0
#        for i in xrange(N):
#            l1 = self.calc_l1(zl[i], Pl[i], z0[i], P0[i])
#            P0cho = scipy.linalg.cho_factor(P0[i], check_finite=False)
#            #(_tmp, ld) = numpy.linalg.slogdet(P0[i])
#            ld = numpy.sum(numpy.log(numpy.diagonal(P0cho[0])))*2
#            tmp = scipy.linalg.cho_solve(P0cho, l1, check_finite=False)
#            lpz0 -= 0.5*(ld + numpy.trace(tmp))
#        
#            # Calculate gradient
#            for j in range(len(self.params)):
#                tmp = z0_grad[i][j].dot((zl[i]-z0[i]).T)
#                dl1 = -tmp -tmp.T
#                lpz0_grad[j] -= 0.5*mlnlg_compute.compute_logprod_derivative(P0cho, P0_grad[i][j], l1, dl1)
#                
#        return (lpxi0 + lpz0,
#                lpxi0_grad + lpz0_grad)
#    
#
#
#       
#    def eval_logp_xnext(self, particles, x_next, u, t):
#        """ Calculate gradient of a term of the I2 integral approximation
#            as specified in [1].
#            The gradient is an array where each element is the derivative with 
#            respect to the corresponding parameter"""
#        # Calculate l2 according to (16)
#        N = len(particles)
#        lpxn = 0.0
#        
#        (_xi, z, P) = self.get_states(particles)
#        Mzl = self.get_Mz(particles)
#        (xin, zn, Pn) = self.get_states(x_next)
#        
#        (A, f, Q, _, _, Q_identical) = self.calc_A_f_Q(particles, u, t)
#        l2 = self.calc_l2(xin, zn, Pn, z, P, A, f, Mzl)
#        if (Q_identical):
#            Qcho = scipy.linalg.cho_factor(Q[0], check_finite=False)
#            #(_tmp, ld) = numpy.linalg.slogdet(Q[0])
#            ld = numpy.sum(numpy.log(numpy.diagonal(Qcho[0])))*2
#            for i in xrange(N):
#                tmp = scipy.linalg.cho_solve(Qcho, l2[i], check_finite=False)
#                lpxn -= 0.5*(ld + numpy.trace(tmp))                                             
#        else:
#            for i in xrange(N):
#                Qcho = scipy.linalg.cho_factor(Q[i], check_finite=False)
#                #(_tmp, ld) = numpy.linalg.slogdet(Q[i])
#                ld = numpy.sum(numpy.log(numpy.diagonal(Qcho[0])))*2
#                tmp = scipy.linalg.cho_solve(Qcho, l2[i], check_finite=False)
#                lpxn -= 0.5*(ld + numpy.trace(tmp))
#      
#        return lpxn
#
#    def eval_logp_xnext_val_grad(self, particles, x_next, u, t):
#        """ Calculate gradient of a term of the I2 integral approximation
#            as specified in [1].
#            The gradient is an array where each element is the derivative with 
#            respect to the corresponding parameter"""
#        N = len(particles)
#        lpxn = 0.0
#        lpxn_grad = numpy.zeros(self.params.shape)
#        
#        (A_grad, f_grad, Q_grad) = self.get_pred_dynamics_grad(particles=particles, u=u, t=t)
#        if (A_grad == None and f_grad == None and Q_grad == None):
#            lpxn = self.eval_logp_xnext(particles, x_next, u, t)
#        else:
#        
#            (_xi, zl, Pl) = self.get_states(particles)
#            Mzl = self.get_Mz(particles)
#            (xin, zn, Pn) = self.get_states(x_next)
#            
#            (A, f, Q, _, _, Q_identical) = self.calc_A_f_Q(particles, u, t)
#            
#    
#            dim = self.lxi + self.kf.lz
#    
#            if (Q_grad == None):
#                Q_grad = N*(numpy.zeros((len(self.params), dim, dim)),)
#            
#            (l2, l2_grad) = self.calc_l2_grad(xin, zn, Pn, zl, Pl, A, f, Mzl, f_grad, A_grad)
#            if (Q_identical):
#                Qcho = scipy.linalg.cho_factor(Q[0], check_finite=False)
#                #(_tmp, ld) = numpy.linalg.slogdet(Q[0])
#                ld = numpy.sum(numpy.log(numpy.diagonal(Qcho[0])))*2
#                for i in xrange(N):
#                    tmp = scipy.linalg.cho_solve(Qcho, l2[i], check_finite=False)
#                    lpxn -= 0.5*(ld + numpy.trace(tmp))   
#                    for j in xrange(len(self.params)):
#                        lpxn_grad[j] -= 0.5*mlnlg_compute.compute_logprod_derivative(Qcho, Q_grad[i][j],
#                                                                       l2[i], l2_grad[i][j])                                          
#            else:
#                for i in xrange(N):
#                    Qcho = scipy.linalg.cho_factor(Q[i], check_finite=False)
#                    #(_tmp, ld) = numpy.linalg.slogdet(Q[i])
#                    ld = numpy.sum(numpy.log(numpy.diagonal(Qcho[0])))*2
#                    tmp = scipy.linalg.cho_solve(Qcho, l2[i], check_finite=False)
#                    lpxn -= 0.5*(ld + numpy.trace(tmp))
#                    for j in xrange(len(self.params)):
#                        lpxn_grad[j] -= 0.5*mlnlg_compute.compute_logprod_derivative(Qcho, Q_grad[i][j],
#                                                                       l2[i], l2_grad[i][j])
#          
#        return (lpxn, lpxn_grad)
#
#    def eval_logp_y(self, particles, y, t):
#        """ Calculate a term of the I3 integral approximation
#        and its gradient as specified in [1]"""
#        N = len(particles)
#        (y, Cz, hz, Rz, _, _, Rz_identical) = self.get_meas_dynamics_int(particles, y, t)
#        (_xil, zl, Pl) = self.get_states(particles)
#        logpy = 0.0
#        l3 = self.calc_l3(y, zl, Pl, Cz, hz)
#        if (Rz_identical):
#            Rzcho = scipy.linalg.cho_factor(Rz[0], check_finite=False)
#            #(_tmp, ld) = numpy.linalg.slogdet(Rz[0])
#            ld = numpy.sum(numpy.log(numpy.diagonal(Rzcho[0])))*2
#            for i in xrange(N):
#                tmp = scipy.linalg.cho_solve(Rzcho, l3[i], check_finite=False)
#                logpy -= 0.5*(ld + numpy.trace(tmp))
#        else:
#            for i in xrange(N):
#            # Calculate l3 according to (19b)
#                Rzcho = scipy.linalg.cho_factor(Rz[i], check_finite=False)
#                #(_tmp, ld) = numpy.linalg.slogdet(Rz[i])
#                ld = numpy.sum(numpy.log(numpy.diagonal(Rzcho[0])))*2
#                tmp = scipy.linalg.cho_solve(Rzcho, l3[i], check_finite=False)
#                logpy -= 0.5*(ld + numpy.trace(tmp))
#
#        return logpy
#
#    def eval_logp_y_val_grad(self, particles, y, t):
#        """ Calculate a term of the I3 integral approximation
#        and its gradient as specified in [1]"""
#        
#        N = len(particles)
#        logpy = 0.0
#        lpy_grad = numpy.zeros(self.params.shape)
#        (y, Cz, hz, Rz, _, _, Rz_identical) = self.get_meas_dynamics_int(particles, y, t)
#        (C_grad, h_grad, R_grad) = self.get_meas_dynamics_grad(particles=particles, y=y, t=t)
#        if (C_grad == None and h_grad == None and R_grad == None):
#            logpy = self.eval_logp_y(particles, y, t)
#        else:
#            if (R_grad == None):
#                R_grad = N*(numpy.zeros((len(self.params), len(y), len(y))),)
#                
#            (_xil, zl, Pl) = self.get_states(particles)
#            
#            (l3, l3_grad) = self.calc_l3_grad(y, zl, Pl, Cz, hz, C_grad, h_grad)
#            
#            if (Rz_identical):
#                Rzcho = scipy.linalg.cho_factor(Rz[0], check_finite=False)
#                #(_tmp, ld) = numpy.linalg.slogdet(Rz[0])
#                ld = numpy.sum(numpy.log(numpy.diagonal(Rzcho[0])))*2
#                for i in xrange(N):
#                    tmp = scipy.linalg.cho_solve(Rzcho, l3[i], check_finite=False)
#                    logpy -= 0.5*(ld + numpy.trace(tmp))
#                    for j in range(len(self.params)):
#                        lpy_grad[j] -= 0.5*mlnlg_compute.compute_logprod_derivative(Rzcho, R_grad[i][j],
#                                                                      l3[i], l3_grad[i][j])
#            else:
#                for i in xrange(N):
#                    Rzcho = scipy.linalg.cho_factor(Rzcho, check_finite=False)
#                    #(_tmp, ld) = numpy.linalg.slogdet(Rz[i])
#                    ld = numpy.sum(numpy.log(numpy.diagonal(Rzcho[0])))*2
#                    tmp = scipy.linalg.cho_solve(Rzcho, l3[i])
#                    logpy -= 0.5*(ld + numpy.trace(tmp))
#                    for j in range(len(self.params)):
#                        lpy_grad[j] -= 0.5*mlnlg_compute.compute_logprod_derivative(Rzcho, R_grad[i][j],
#                                                                      l3[i], l3_grad[i][j])
#
#        return (logpy, lpy_grad)


class NonlinearGaussianInitialGaussian(NonlinearGaussian):
    def __init__(self, x0, Px0=None, **kwargs):
        
        self.x0 = numpy.copy(x0).reshape((-1,1))
        if (Px0 == None):
            self.Px0 = numpy.zeros((len(self.x0),len(self.x0)))
        else:
            self.Px0 = numpy.copy((Px0))

        super(NonlinearGaussianInitialGaussian, self).__init__(lxi=len(self.x0), 
                                                               **kwargs)

    def create_initial_estimate(self, N):
        Pchol = scipy.linalg.cho_factor(self.Px0)[0]
        noise = numpy.random.normal(size=(self.lxi,N))
        particles = (Pchol.dot(noise)).T+self.x0  
        return particles     

    def eval_logp_x0(self, particles, t):
        """ Calculate gradient of a term of the I1 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""    
            
        N = len(particles)
        res = numpy.empty(N)
        Pchol = scipy.linalg.cho_factor(self.Px0, check_finite=False)
        for i in xrange(N):
            res[i] = kalman.lognormpdf_cho(particles[i] - self.xi0, Pchol)
        return res
    
#    def eval_logp_xi0_grad(self, xil):
#        """ Evaluate probabilty of xi0 """
#        N = len(xil)
#        (xi0_grad, Pxi0_grad) = self.get_xi_intitial_grad(N)
#        lpxi0_grad = numpy.zeros(self.params.shape)
#        Pxi0cho = scipy.linalg.cho_factor(self.Pxi0, check_finite=False)
#        for i in xrange(N):
#            tmp = xil[i]-self.xi0
#            l0 = tmp.dot(tmp.T)
#            for j in range(len(self.params)):
#                tmp2 = tmp.dot(xi0_grad[i][j].T)
#                l0_grad = tmp2 + tmp2.T
#                lpxi0_grad[j] -= 0.5*mlnlg_compute.compute_logprod_derivative(Pxi0cho, Pxi0_grad[i][j], l0, l0_grad)
#                
#        return lpxi0_grad
