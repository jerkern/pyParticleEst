""" Collection of functions and classes used for Particle Filtering/Smoothing """
from pyparticleest.interfaces import FFBSi
import pyximport
pyximport.install(inplace=True)
import pyparticleest.models.mlnlg_compute as mlnlg_compute
import pyparticleest.utils.kalman as kalman
import numpy
import scipy.linalg

class LTV(FFBSi):
    """ Base class for particles of the type linear time varying with additive gaussian noise.

        Implement this type of system by extending this class and provide the methods for returning 
        the system matrices at each time instant  """
    def __init__(self, z0, P0, A=None, C=None, Q=None, 
             R=None, f=None, h=None, params=None, t0=0):
        self.z0 = numpy.copy(z0).reshape((-1,1))
        self.P0 = numpy.copy(P0)
        if (f == None):
            f = numpy.zeros_like(self.z0)
        self.kf = kalman.KalmanSmoother(lz=len(self.z0),
                                        A=A, C=C, 
                                        Q=Q, R=R,
                                        f_k=f, h_k=h)
        
        self.t = t0

    def create_initial_estimate(self, N):
        if (N > 1):
            print("N > 1 redundamt for LTV system (N={0})".format(N), )
        particles = numpy.empty((N,), dtype=numpy.ndarray)
        lz = len(self.z0)
        dim = lz + lz*lz
        
        for i in xrange(N):
            particles[i] = numpy.empty(dim)
            particles[i][:lz] = numpy.copy(self.z0).ravel()
            particles[i][lz:] = numpy.copy(self.P0).ravel()  
        return particles
    
    def set_states(self, particles, z_list, P_list):
        """ Set the estimate of the Rao-Blackwellized states """
        lz = len(self.z0)
        N = len(particles)
        for i in xrange(N):
            particles[i][:lz] = z_list[i].ravel()
            lzP = lz + lz*lz
            particles[i][lz:lzP] = P_list[i].ravel()
 
    def get_states(self, particles):
        """ Returns two variables, the first a list containing all the
            expected values, the second a list of the corresponding covariance
            matrices"""
        N = len(particles)
        zl = list()
        Pl = list()
        lz = len(self.z0)
        for i in xrange(N):
            zl.append(particles[i][:lz].reshape(-1,1))
            lzP = lz + lz*lz
            Pl.append(particles[i][lz:lzP].reshape(self.P0.shape))
        
        return (zl, Pl)

    def get_pred_dynamics(self, u, t):
        # Return (A, f, Q) 
        return (None, None, None)

    def update(self, particles, u, t, noise):
        """ Update estimate using noise as input """
        # Update linear estimate with data from measurement of next non-linear
        # state 
        (zl, Pl) = self.get_states(particles)
        (A, f, Q) = self.get_pred_dynamics(u=u, t=t)
        self.kf.set_dynamics(A=A, Q=Q, f_k=f)
        for i in xrange(len(zl)):
            # Predict z_{t+1}
            (zl[i], Pl[i]) = self.kf.predict(zl[i], Pl[i])
        
        # Predict next states conditioned on eta_next
        self.set_states(particles, zl, Pl)
        self.t = self.t + 1.0
    
    def get_meas_dynamics(self, y, t):
        return (y, None, None, None)
    
    def measure(self, particles, y, t):
        """ Return the log-pdf value of the measurement """

       
        (zl, Pl) = self.get_states(particles)
        (y, C, h, R) = self.get_meas_dynamics(y=y, t=t)
        self.kf.set_dynamics(C=C, R=R, h_k=h)  
        lyz = numpy.empty((len(particles)))
        for i in xrange(len(zl)):
            # Predict z_{t+1}
            lyz[i] = self.kf.measure(y, zl[i], Pl[i])
        
        self.set_states(particles, zl, Pl)
        return lyz
    
    def next_pdf(self, particles, next_cpart, u, t):
        """ Return the log-pdf value for the possible future state 'next' given input u """
        N = len(particles)
        return numpy.zeros((N,))
    
    def sample_process_noise(self, particles, u, t): 
        return None
    
    def sample_smooth(self, particle, next_part, u, y, t):
        """ Update ev. Rao-Blackwellized states conditioned on "next_part" """
        
        (zl, Pl) = self.get_states(particle)
        M = len(particle)
        lz = len(self.z0)
        lzP = lz + lz*lz
        res = numpy.empty((M,lz+2*lz**2))
        for j in xrange(M):
            if (next_part != None):
                zn = next_part[j, :lz].reshape((lz,1))
                Pn = next_part[j, lz:lzP].reshape((lz,lz))
                (A, f, Q) = self.get_pred_dynamics(u=u, t=t)
                self.kf.set_dynamics(A=A, Q=Q, f_k=f)
                (zs, Ps, Ms) = self.kf.smooth(zl[0], Pl[0], zn, Pn, self.kf.A, self.kf.f_k, self.kf.Q)
            else:
                zs = zl[j]
                Ps = Pl[j]
                Ms = numpy.zeros_like(Ps)
            res[j] = numpy.hstack((zs.ravel(), Ps.ravel(), Ms.ravel()))
        
        return res


    def fwd_peak_density(self, u, t):
        return 0.0

    def eval_logp_x0(self, particles, t):
        """ Calculate gradient of a term of the I1 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""    
        # Calculate l1 according to (19a)
        N = len(particles)
        (zl, Pl) = self.get_states(particles)
        lpz0 = numpy.empty(N)
        for i in xrange(N):
            l1 = self.calc_l1(zl[i], Pl[i], self.z0, self.P0)
            (_tmp, ld) = numpy.linalg.slogdet(self.P0)
            tmp = numpy.linalg.solve(self.P0, l1)
            lpz0[i] = -0.5*(ld + numpy.trace(tmp))
        return lpz0
    
    def eval_logp_x0_val_grad(self, particles, t):
        """ Calculate gradient of a term of the I1 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""    
        # Calculate l1 according to (19a)
        N = len(particles)
        lparam = len(self.params)
        lpz0_grad = numpy.zeros(lparam)
        (zl, Pl) = self.get_states(particles)
        (z0_grad, P0_grad) = self.get_initial_grad()
        if (z0_grad == None and P0_grad == None):
            lpz0 = self.eval_logp_x0(particles, t)
        else:
            lpz0 = 0.0
            P0cho = scipy.linalg.cho_factor(self.P0)
            ld = numpy.sum(numpy.log(numpy.diagonal(P0cho[0])))*2
            for i in xrange(N):
                (l1, l1_grad) = self.calc_l1_grad(zl[i], Pl[i], self.z0, self.P0, z0_grad)
                tmp = scipy.linalg.cho_solve(P0cho, l1)
                lpz0 += -0.5*(ld + numpy.trace(tmp))
                for j in range(len(self.params)):
                    lpz0_grad[j] -= 0.5*mlnlg_compute.compute_logprod_derivative(P0cho, P0_grad[j], l1, l1_grad[j])
        return (lpz0, lpz0_grad)
    

        
    def eval_logp_xnext(self, particles, x_next, u, t):
        """ Calculate gradient of a term of the I2 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""
        # Calculate l2 according to (16)
        N = len(particles)    
        (zl, Pl) = self.get_states(particles)
        (zn, Pn) = self.get_states(x_next)
        (A, f, Q) = self.get_pred_dynamics(u=u, t=t)
        (A_grad, f_grad, Q_grad) = self.get_pred_dynamics_grad(u=u, t=t)
        self.kf.set_dynamics(A=A, Q=Q, f_k=f)
        self.t = t
        lpxn = numpy.empty(N)
        
        for k in xrange(N):
            lz = len(self.z0)
            lzP = lz + lz*lz
            Mz = particles[k][lzP:].reshape((lz,lz))
            (l2, _A, _M_ext, _predict_err) = self.calc_l2(zn[k], Pn[k], zl[k], Pl[k], self.kf.A, self.kf.f_k, Mz)
            (_tmp, ld) = numpy.linalg.slogdet(self.kf.Q)
            tmp = numpy.linalg.solve(self.kf.Q, l2)
            lpxn[k] = -0.5*(ld + numpy.trace(tmp))
        
        return lpxn
    
    def eval_logp_xnext_val_grad(self, particles, x_next, u, t):
        """ Calculate gradient of a term of the I2 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""
        # Calculate l2 according to (16)
        N = len(particles)
        lparam = len(self.params)
        (zl, Pl) = self.get_states(particles)
        (zn, Pn) = self.get_states(x_next)
        (A, f, Q) = self.get_pred_dynamics(u=u, t=t)
        (A_grad, f_grad, Q_grad) = self.get_pred_dynamics_grad(u=u, t=t)
        lpxn_grad = numpy.zeros(lparam)
        if (A_grad == None and f_grad == None and Q_grad == None):
            lpxn = self.eval_logp_xnext(particles, x_next, u, t)
        else:
            self.kf.set_dynamics(A=A, Q=Q, f_k=f)
            lpxn = 0.0
            Qcho = scipy.linalg.cho_factor(self.kf.Q, check_finite=False)
            ld = numpy.sum(numpy.log(numpy.diagonal(Qcho[0])))*2
            
            if (Q_grad == None):
                Q_grad = numpy.zeros((len(self.params), self.kf.lz, self.kf.lz))
            
            for k in xrange(N):
                lz = len(self.z0)
                lzP = lz + lz*lz
                Mz = particles[k][lzP:].reshape((lz,lz))
                (l2, l2_grad) = self.calc_l2_grad(zn[k], Pn[k], zl[k], Pl[k], self.kf.A, self.kf.f_k, Mz, A_grad, f_grad)
                tmp = scipy.linalg.cho_solve(Qcho, l2)
                lpxn += -0.5*(ld + numpy.trace(tmp))

                for j in range(len(self.params)):
                    lpxn_grad[j] -= 0.5*mlnlg_compute.compute_logprod_derivative(Qcho, Q_grad[j], l2, l2_grad[j])
      
        return (lpxn, lpxn_grad)
    


    def eval_logp_y(self, particles, y, t):
        """ Calculate a term of the I3 integral approximation
        and its gradient as specified in [1]"""
        N = len(particles)
        self.t = t
        (y, C, h, R) = self.get_meas_dynamics(y=y, t=t)
        self.kf.set_dynamics(C=C, R=R, h_k=h) 
        (zl, Pl) = self.get_states(particles)
        logpy = numpy.empty(N)
        for i in xrange(N):
        # Calculate l3 according to (19b)
            # Calculate l3 according to (19b)
            l3 = self.calc_l3(y, zl[i], Pl[i])
            (_tmp, ld) = numpy.linalg.slogdet(self.kf.R)
            tmp = numpy.linalg.solve(self.kf.R, l3)
            logpy[i] = -0.5*(ld + numpy.trace(tmp))

        return logpy
    
    def eval_logp_y_val_grad(self, particles, y, t):
        """ Calculate a term of the I3 integral approximation
        and its gradient as specified in [1]"""
        N = len(particles)
        lparam = len(self.params)
        (y, C, h, R) = self.get_meas_dynamics(y=y, t=t)
        (C_grad, h_grad, R_grad) = self.get_meas_dynamics_grad(y=y, t=t)
        logpy_grad = numpy.zeros(lparam)
        if (C_grad == None and h_grad == None and R_grad == None):
            logpy = self.eval_logp_y(particles, y, t)
        else:
            
            self.kf.set_dynamics(C=C, R=R, h_k=h) 
            Rcho = scipy.linalg.cho_factor(self.kf.R, check_finite=False)
            ld = numpy.sum(numpy.log(numpy.diagonal(Rcho[0])))*2
            (zl, Pl) = self.get_states(particles)
            logpy = 0.0
            
            if (R_grad == None):
                R_grad = numpy.zeros((len(self.params), len(y), len(y)))
            
            for i in xrange(N):
            # Calculate l3 according to (19b)
                # Calculate l3 according to (19b)
                (l3, l3_grad) = self.calc_l3_grad(y, zl[i], Pl[i])
                tmp = scipy.linalg.cho_solve(Rcho, l3)
                logpy += -0.5*(ld + numpy.trace(tmp))
                
                for j in range(len(self.params)):
                    logpy_grad[j] -= 0.5*mlnlg_compute.compute_logprod_derivative(Rcho, R_grad[j], l3, l3_grad[j])

        return (logpy, logpy_grad)
    
    def get_pred_dynamics_grad(self, u, t):
        """ Override this method if (A, f, Q) depends on the parameters """
        return (None, None, None)
    
    def get_meas_dynamics_grad(self, y, t):
        """ Override this method if (C, h, R) depends on the parameters """
        return (None, None, None)
    
    def get_initial_grad(self):
        """ Default implementation has no dependence on xi, override if needed """
        lparam = len(self.params)
        return (numpy.zeros((lparam, self.kf.lz, 1)),
                numpy.zeros((lparam, self.kf.lz, self.kf.lz)))

    def calc_l1(self, z, P, z0, P0):
        z0_diff = z - z0
        l1 = z0_diff.dot(z0_diff.T) + P
        return l1

    def calc_l1_grad(self, z, P, z0, P0, z0_grad):
        lparams = len(self.params)
        z0_diff = z - z0
        l1 = z0_diff.dot(z0_diff.T) + P
        l1_diff = numpy.zeros((lparams, self.kf.lz, self.kf.lz))
        if (z0_grad != None):
            for j in xrange(lparams):
                tmp = -z0_grad[j].dot(z0_diff.T)
                l1_diff[j] += tmp + tmp.T
        return (l1, l1_diff)

    def calc_l2(self, zn, Pn, z, P, A, f, M):
        predict_err = zn - f - A.dot(z)
        AM = A.dot(M)
        l2 = predict_err.dot(predict_err.T)
        l2 += Pn + A.dot(P).dot(A.T) - AM.T - AM
        return (l2, A, M, predict_err)
    
    def calc_l2_grad(self, zn, Pn, z, P, A, f, M, A_grad, f_grad):
        lparam = len(self.params)
        predict_err = zn - f - A.dot(z)
        AM = A.dot(M)
        l2 = predict_err.dot(predict_err.T)
        l2 += Pn + A.dot(P).dot(A.T) - AM.T - AM
        l2_grad = numpy.zeros((lparam, self.kf.lz, self.kf.lz))
        if (f_grad != None):
            for j in xrange(lparam):
                tmp = -f_grad[j].dot(predict_err.T)
                l2_grad[j] += tmp + tmp.T
        if (A_grad != None):
            for j in xrange(lparam):
                tmp = -A_grad[j].dot(z).dot(predict_err.T)
                l2_grad[j] += tmp + tmp.T
                tmp = A_grad[j].dot(P).dot(A.T)
                l2_grad[j] += tmp + tmp.T
                tmp = -A_grad[j].dot(M)
                l2_grad[j] += tmp + tmp.T
        return (l2, l2_grad)
        
    def calc_l3(self, y, z, P):
        meas_diff = self.kf.measurement_diff(y.reshape((-1,1)),
                                             z,
                                             C=self.kf.C,
                                             h_k=self.kf.h_k) 
        l3 = meas_diff.dot(meas_diff.T)
        l3 += self.kf.C.dot(P).dot(self.kf.C.T)
        return l3
    
    def calc_l3_grad(self, y, z, P, C_grad, h_grad):
        lparam = len(self.params)
        meas_diff = self.kf.measurement_diff(y.reshape((-1,1)),
                                             z,
                                             C=self.kf.C,
                                             h_k=self.kf.h_k) 
        l3 = meas_diff.dot(meas_diff.T)
        l3 += self.kf.C.dot(P).dot(self.kf.C.T)
        l3_grad = numpy.zeros((lparam, len(y), len(y)))
        if (h_grad != None):
            for j in xrange(lparam):
                tmp = -h_grad[j].dot(meas_diff)
                l3_grad[j] += tmp + tmp.T
        if (C_grad != None):
            for j in xrange(lparam):
                tmp = -C_grad[j].dot(z).dot(meas_diff)
                l3_grad[j] += tmp + tmp.T
                tmp = C_grad[j].dot(P).dot(self.kf.C)
                l3_grad[j] += tmp + tmp.T
        return (l3, l3_grad)