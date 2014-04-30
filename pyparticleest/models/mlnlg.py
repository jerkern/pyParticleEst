""" Collection of functions and classes used for Particle Filtering/Smoothing """
from pyparticleest.part_utils import RBPSBase
import scipy.linalg
import numpy.random
import math
import pyximport
pyximport.install(inplace=True)
import pyparticleest.models.mlnlg_compute as mlnlg_compute
import pyparticleest.kalman as kalman



class MixedNLGaussian(RBPSBase):
    """ Base class for particles of the type mixed linear/non-linear with additive gaussian noise.
    
        Implement this type of system by extending this class and provide the methods for returning 
        the system matrices at each time instant  """
    def __init__(self, lxi, lz, Az=None, C=None, Qz=None, R=None, fz=None,
                 Axi=None, Qxi=None, Qxiz=None, fxi=None, h=None, params=None):
        if (Axi != None):
            self.Axi = numpy.copy(Axi)
        else:
            self.Axi = None
        if (fxi != None):
            self.fxi = numpy.copy(fxi)
        else:
            self.fxi = numpy.zeros((lxi,1))
        if (Qxi != None):
            self.Qxi = numpy.copy(Qxi)
        else:
            self.Qxi = None
        if (Qxiz != None):
            self.Qxiz = numpy.copy(Qxiz)
        else:
            self.Qxiz = None
            
        self.lxi = lxi
        
        return super(MixedNLGaussian, self).__init__(lz=lz,
                                                     Az=Az, C=C, 
                                                     Qz=Qz, R=R,
                                                     hz=h, fz=fz)

    def set_dynamics(self, Az=None, fz=None, Qz=None, R=None,
                     Axi=None, fxi=None, Qxi=None, Qxiz=None, 
                     C=None, h=None):
        super(MixedNLGaussian, self).set_dynamics(Az=Az, C=C, Qz=Qz, R=R, fz=fz,hz=h)

        if (Axi != None):
            self.Axi = numpy.copy(Axi)
        if (Az != None):
            self.Az = numpy.copy(Az)
        if (Qxi != None):
            self.Qxi = numpy.copy(Qxi)
        if (Qxiz != None):
            self.Qxiz = numpy.copy(Qxiz)
        if (Qz != None):
            self.Qz = numpy.copy(self.kf.Q)
        if (fz != None):
            self.fz = numpy.copy(self.kf.f_k)
        if (fxi != None):
            self.fxi = numpy.copy(fxi)

    def sample_process_noise(self, particles, u, t): 
        """ Return sampled process noise for the non-linear states """
        (Axi, fxi, Qxi, _, _, _) = self.get_nonlin_pred_dynamics_int(particles=particles, u=u, t=t)
        (_xil, zl, Pl) = self.get_states(particles)
        N = len(particles)
        # This is probably not so nice performance-wise, but will
        # work initially to profile where the bottlenecks are.
                    
        dim=len(_xil[0])
        noise = numpy.empty((N,dim))
        zeros = numpy.zeros(dim)
        for i in xrange(N):
            Sigma = Qxi[i] + Axi[i].dot(Pl[i]).dot(Axi[i].T)
            noise[i] =  numpy.random.multivariate_normal(zeros, Sigma).ravel()    
        return noise

    def calc_xi_next(self, particles, noise, u, t):
        """ Update non-linear state using sampled noise,
        # the noise term here includes the uncertainty from z """
        xi_pred = self.pred_xi(particles=particles, u=u, t=t)
        xi_next = xi_pred + noise
   
        return xi_next

    def pred_xi(self, particles, u, t):
        N = len(particles)
        (Axi, fxi, _, _, _, _) = self.get_nonlin_pred_dynamics_int(particles=particles, u=u, t=t)
        (xil, zl, _Pl) = self.get_states(particles)
        dim=len(xil[0])
        xi_next = numpy.empty((N,dim))
        # This is probably not so nice performance-wise, but will
        # work initially to profile where the bottlenecks are.
        for i in xrange(N):
            xi_next[i] =  Axi[i].dot(zl[i]) + fxi[i]
        return xi_next
    
    def meas_xi_next(self, particles, xi_next, u, t):
        """ Update estimate using observation of next state """
        # This is what is sometimes called "the second measurement update"
        # for Rao-Blackwellized particle filters
        
        N = len(particles)
        (xil, zl, Pl) = self.get_states(particles)
        (Axi, fxi, Qxi, _, _, _) = self.get_nonlin_pred_dynamics_int(particles=particles, u=u, t=t)
        for i in xrange(N):
            self.kf.measure_full(y=xi_next[i].reshape((self.lxi,1)),
                                 z=zl[i].reshape((self.kf.lz,1)),
                                 P=Pl[i], C=Axi[i], h_k=fxi[i], R=Qxi[i])
        
        # Predict next states conditioned on eta_next
        self.set_states(particles, xil, zl, Pl)
    
    def get_cross_covariance(self, particles, u, t):
        return None
    
    def calc_cond_dynamics(self, particles, xi_next, u, t):
        #Compensate for noise correlation
        N = len(particles)
        #(xil, zl, Pl) = self.get_states(particles)
        
        (Az, fz, Qz, _, _, _) = self.get_lin_pred_dynamics_int(particles=particles, u=u, t=t)

        Qxiz = self.get_cross_covariance(particles=particles, u=u, t=t)
        if (Qxiz == None and self.Qxiz == None):
            return (Az, fz, Qz)
        if (Qxiz == None):
            Qxiz = N*(self.Qxiz,)
        
        (Axi, fxi, Qxi, _, _, _) = self.get_nonlin_pred_dynamics_int(particles=particles, u=u, t=t)
        
        Acond = list()
        fcond = list()
        Qcond = list()
        
        for i in xrange(N):
            #TODO linalg.solve instead?
            tmp = Qxiz[i].T.dot(scipy.linalg.inv(Qxi[i]))
            Acond.append(Az[i] - tmp.dot(Axi[i]))
            #Acond.append(Az[i])
            fcond.append(fz[i] +  tmp.dot(xi_next[i] - fxi[i]))
            # TODO, shouldn't Qz be affected? Why wasn't it before?
            Qcond.append(Qz[i])
        
        return (Acond, fcond, Qcond)
        
        
    
    def cond_predict(self, particles, xi_next, u, t):
        # Calc (z_t | xi_{t+1}, y_t)
        self.meas_xi_next(particles=particles, xi_next=xi_next, u=u, t=t)
        #Compensate for noise correlation
        (Az, fz, Qz) = self.calc_cond_dynamics(particles=particles, xi_next=xi_next, u=u, t=t)
        (_, zl, Pl) = self.get_states(particles)
        # Predict next states conditioned on xi_next
        for i in xrange(len(zl)):
            # Predict z_{t+1}
            (zl[i], Pl[i]) = self.kf.predict_full(z=zl[i], P=Pl[i], A=Az[i], f_k=fz[i], Q=Qz[i])
        
        self.set_states(particles, xi_next, zl, Pl)

    def eval_1st_stage_weights(self, particles, u, y, t):
        part = numpy.copy(particles)
        xin = self.pred_xi(part, u, t)
        self.cond_predict(part, xin, u, t)
        return self.measure(part, y, t)
        
        
    def measure(self, particles, y, t):
        """ Return the log-pdf value of the measurement """
        
        (xil, zl, Pl) = self.get_states(particles)
        N = len(particles)
        (y, Cz, hz, Rz, Cz_identical, _, Rz_identical) = self.get_meas_dynamics_int(particles=particles, y=y, t=t)
            
        lyz = numpy.empty(N)
        if (Rz_identical):
            if (Cz_identical and Cz[0] == None):
                diff = y - hz
                dim = Rz[0].shape[0]
                if (dim == 1):
                    lyz = kalman.lognormpdf_scalar(diff.ravel(), Rz[0])
                else:
                    Rchol = scipy.linalg.cho_factor(Rz[0])
                    lyz = kalman.lognormpdf_cho_vec(diff.reshape((-1,dim,1)), Rchol)
            if (Rz[0].shape[0] == 1):
                for i in xrange(len(zl)):
                    lyz[i] = self.kf.measure_full_scalar(y=y, z=zl[i], P=Pl[i], C=Cz[i], h_k=hz[i], R=Rz[i])
            else:
                for i in xrange(len(zl)):
                    lyz[i] = self.kf.measure_full(y=y, z=zl[i], P=Pl[i], C=Cz[i], h_k=hz[i], R=Rz[i])
        
        self.set_states(particles, xil, zl, Pl)
        return lyz

    def calc_A_f_Q(self, particles, u, t):
        N = len(particles)
        (Az, fz, Qz, Az_identical, fz_identical, Qz_identical) = self.get_lin_pred_dynamics_int(particles=particles, u=u, t=t)
        (Axi, fxi, Qxi, Axi_identical, fxi_identical, Qxi_identical) = self.get_nonlin_pred_dynamics_int(particles=particles, u=u, t=t)
        Qxiz = self.get_cross_covariance(particles=particles, u=u, t=t)
        Qxiz_identical = False

        A_identical = False
        f_identical = False
        Q_identical = False

        if (Qxiz == None):
            Qxiz_identical = True
            if (self.Qxiz == None):
                Qxiz = N*(numpy.zeros((Qxi[0].shape[0],Qz[0].shape[0])),)
            else:
                Qxiz = N*(self.Qxiz,)
        
        if (Az_identical and Axi_identical):
            A = numpy.repeat(numpy.vstack((Axi[0], Az[0]))[numpy.newaxis],N,0)
            A_identical = True
        else:
            A = list()
            for i in xrange(N):
                A.append(numpy.vstack((Axi[i], Az[i])))
                
        if (fxi_identical and fz_identical):
            f = N*(numpy.vstack((fxi[0], fz[0])),)
            f_identical = True
        else:
            f = list()
            for i in xrange(N):
                f.append(numpy.vstack((fxi[i], fz[i])))
                
        if (Qxi_identical and Qz_identical and Qxiz_identical):
            Q = N*(numpy.vstack((numpy.hstack((Qxi[0], Qxiz[0])),
                              numpy.hstack((Qxiz[0].T, Qz[0])))),)
            Q_identical = True
        else:
            Q = list()
            for i in xrange(N):
                Q.append(numpy.vstack((numpy.hstack((Qxi[i], Qxiz[i])),
                              numpy.hstack((Qxiz[i].T, Qz[i])))))
                
        return (A, f, Q, A_identical, f_identical, Q_identical)
    
    def next_pdf_max(self, particles, u, t):
        """ Implements the fwd_peak_density function for MixedNLGaussian models """
        N = len(particles)
        pmax = numpy.empty(N)
        (_, _, Pl) = self.get_states(particles)
        (A, _, Q, _, _, _) = self.calc_A_f_Q(particles, u=u, t=t)
        dim=self.lxi + self.kf.lz
        zeros = numpy.zeros((dim,1))
        l2pi = math.log(2*math.pi)
        for i in xrange(N):
            Sigma = Q[i] + A[i].dot(Pl[i]).dot(A[i].T)
            Schol = scipy.linalg.cho_factor(Sigma, check_finite=False)
#            pmax[i] = kalman.lognormpdf_cho(zeros, zeros, Schol)
            ld = numpy.sum(numpy.log(numpy.diag(Schol[0])))*2
            pmax[i] = -0.5*(dim*l2pi+ld)
        
        return numpy.max(pmax)
        
    def next_pdf(self, particles, next_part, u, t):
        """ Implements the next_pdf function for MixedNLGaussian models """
        
        N = len(particles)
        Nn = len(next_part)
        if (N > 1 and Nn == 1):
            next_part = numpy.repeat(next_part, N, 0)
        lpx = numpy.empty(N)
        (_, zl, Pl) = self.get_states(particles)
        (A, f, Q, _, _, _) = self.calc_A_f_Q(particles, u=u, t=t)
        
        for i in xrange(N):
            x_next = next_part[i,:self.lxi+self.kf.lz].reshape((self.lxi+self.kf.lz,1))
            xp = f[i] + A[i].dot(zl[i])
            Sigma = A[i].dot(Pl[i]).dot(A[i].T) + Q[i]
            Schol = scipy.linalg.cho_factor(Sigma, check_finite=False)
            lpx[i] = kalman.lognormpdf_cho(x_next-xp,Schol)
        
        return lpx
    
    def sample_smooth(self, particles, next_part, u, t):
        """ Implements the sample_smooth function for MixedNLGaussian models """
        M = len(particles)
        res = numpy.zeros((M,self.lxi+2*(self.kf.lz + 2*self.kf.lz**2)))
        part = numpy.copy(particles)
        (xil, zl, Pl) = self.get_states(part)
        
        if (next_part != None):
            (xinl, znl, _unused) = self.get_states(next_part)
            (Acond, fcond, Qcond) = self.calc_cond_dynamics(part, xinl, u=u, t=t)
        
            self.meas_xi_next(part, xinl, u=u, t=t)
            
            (xil, zl, Pl) = self.get_states(part)
                                            
            for j in range(M):
                self.kf.measure_full(znl[j], zl[j], Pl[j],
                                     C=Acond[j], h_k=fcond[j], R=Qcond[j])
            self.set_states(particles, xil, zl, Pl)

        for j in range(M):
            xi = numpy.copy(xil[j]).ravel()
            z = numpy.random.multivariate_normal(zl[j].ravel(), 
                                                 Pl[j]).ravel()
            res[j,:self.lxi+self.kf.lz] = numpy.hstack((xi, z))
        return res
#    def post_smoothing(self, st):
#        return
#    
#    def next_pdf(self, particles, next_part, u, t):
#        """ Implements the next_pdf function for MixedNLGaussian models """
#        
#        N = len(particles)
#        Nn = len(next_part)
#        if (N > 1 and Nn == 1):
#            next_part = numpy.repeat(next_part, N, 0)
#        lpx = numpy.empty(N)
#        (_, zl, Pl) = self.get_states(particles)
#        (A, f, Q, _, _, _) = self.calc_A_f_Q(particles, u=u, t=t)
#        
#        for i in xrange(N):
#            x_next = numpy.vstack((next_part[i,:self.lxi].reshape((self.lxi,1)),
#                                   next_part[i,-self.kf.lz:].reshape((self.kf.lz,1))))
#            xp = f[i] + A[i].dot(zl[i])
#            Sigma = A[i].dot(Pl[i]).dot(A[i].T) + Q[i]
#            Schol = scipy.linalg.cho_factor(Sigma, check_finite=False)
#            lpx[i] = kalman.lognormpdf_cho(x_next-xp,Schol)
#        
#        return lpx
#
#    def sample_smooth(self, particles, next_part, u, t):
#        """ Implements the sample_smooth function for MixedNLGaussian models """
#        M = len(particles)
#        res = numpy.zeros((M,self.lxi+self.kf.lz + 2*self.kf.lz**2+self.kf.lz))
#        part = numpy.copy(particles)
#        (xil, zl, Pl) = self.get_states(part)
#        
#        Mz = numpy.zeros((M, self.kf.lz, self.kf.lz))
#        
#        if (next_part != None):
#            (xinl, znl, Pnl) = self.get_states(next_part)
#            (A, f, Q, _, _, _) = self.calc_A_f_Q(part, u=u, t=t)
#            (xil, zl, Pl) = self.get_states(part)
#            for j in range(M):
#                W = A[j].T.dot(numpy.linalg.inv(Q[j]))
#                Wa = W[:,:self.lxi]
#                Wz = W[:,self.lxi:]
#                fa = f[j][:self.lxi]
#                fz = f[j][self.lxi:]
#                Sigma = Pl[j] - Pl[j].dot(A[j].T).dot(
#                                   numpy.linalg.inv(Q[j]+A[j].dot(Pl[j].dot(A[j].T)))
#                                   ).dot(A[j]).dot(Pl[j])
#                c = Sigma.dot(Wa.dot(xinl[j]-fa)-Wz.dot(fz)+numpy.linalg.solve(Pl[j],zl[j]))
#                zl[j] = Sigma.dot(Wz).dot(znl[j])+c
#                Mz[j] = Sigma.dot(Wz).dot(Pnl[j])
#                Pl[j] = Sigma + Mz[j].dot(Wz.T).dot(Sigma)
#               
#               
#        self.set_states(res, xil, zl, Pl)
#        self.set_Mz(res, Mz)
#
#        for j in range(M):
#            z = numpy.random.multivariate_normal(zl[j].ravel(), 
#                                                 Pl[j]).ravel()
#            res[j,-self.kf.lz:] = z
#        return res

    def copy_ind(self, particles, new_ind=None):
        if (new_ind != None):
            return numpy.copy(particles[new_ind])
        else:
            return numpy.copy(particles)
    
    def set_states(self, particles, xi_list, z_list, P_list):
        """ Set the estimate of the Rao-Blackwellized states """
        N = len(particles)
        zend = self.lxi+self.kf.lz
        Pend = zend+self.kf.lz**2

        particles[:,:self.lxi] = xi_list.reshape((N, self.lxi))
        particles[:,self.lxi:zend] = z_list.reshape((N, self.kf.lz))
        particles[:,zend:Pend] = P_list.reshape((N, self.kf.lz**2))
 
    def get_states(self, particles):
        """ Return the estimate of the Rao-Blackwellized states.
            Must return two variables, the first a list containing all the
            expected values, the second a list of the corresponding covariance
            matrices"""
        N = len(particles)
        zend = self.lxi+self.kf.lz
        Pend = zend+self.kf.lz**2
        
        xil = particles[:,:self.lxi, numpy.newaxis]
        zl = particles[:,self.lxi:zend, numpy.newaxis]
        Pl = particles[:,zend:Pend].reshape((N, self.kf.lz, self.kf.lz))
        
        return (xil, zl, Pl)
    
    def get_Mz(self, smooth_particles):
        """ Return the estimate of the Rao-Blackwellized states.
            Must return two variables, the first a list containing all the
            expected values, the second a list of the corresponding covariance
            matrices"""
        N = len(smooth_particles)
        zend = self.lxi+self.kf.lz
        Pend = zend+self.kf.lz**2
        Mend = Pend + self.kf.lz**2

        Mz = smooth_particles[:,Pend:Mend].reshape((N, self.kf.lz, self.kf.lz))
        return Mz
    
    def set_Mz(self, smooth_particles, Mz):
        N = len(smooth_particles)
        zend = self.lxi+self.kf.lz
        Pend = zend+self.kf.lz**2
        Mend = Pend + self.kf.lz**2
            
        smooth_particles[:,Pend:Mend] = Mz.reshape((N, self.kf.lz**2))
    
    def set_params(self, params):
        self.params = numpy.copy(params).reshape((-1,1))

    def get_pred_dynamics_grad(self, particles, u, t):
        """ Override this method if (A, f, Q) depends on the parameters """
        return (None, None, None)
    
    def get_meas_dynamics_grad(self, particles, y, t):
        """ Override this method if (C, h, R) depends on the parameters """
        return (None, None, None)
    
    def eval_logp_xi0(self, xil):
        """ Evaluate logprob of the initial non-linear state eta,
            default implementation assumes all are equal, override this
            if another behavior is desired """
        return 0.0
    
    def eval_logp_xi0_grad(self, xil):
        """ Evaluate logprob of the initial non-linear state eta,
            default implementation assumes all are equal, override this
            if another behavior is desired """
        return numpy.zeros(self.params.shape)


    def calc_l1(self, z, P, z0, P0):
        z0_diff = z - z0
        l1 = z0_diff.dot(z0_diff.T) + P
        return l1
        
    def eval_logp_x0(self, particles, t):
        """ Calculate gradient of a term of the I1 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""    
            
        # Calculate l1 according to (19a)
        N = len(particles)
        (xil, zl, Pl) = self.get_states(particles)
        (z0, P0) = self.get_rb_initial(xil)
        lpxi0 = self.eval_logp_xi0(xil)
        lpz0 = 0.0
        for i in xrange(N):
            l1 = self.calc_l1(zl[i], Pl[i], z0[i], P0[i])
            P0cho = scipy.linalg.cho_factor(P0[i], check_finite=False)
            #(_tmp, ld) = numpy.linalg.slogdet(P0[i])
            ld = numpy.sum(numpy.log(numpy.diagonal(P0cho[0])))*2
            tmp = scipy.linalg.cho_solve(P0cho, l1, check_finite=False)
            lpz0 -= 0.5*(ld + numpy.trace(tmp))
        return lpxi0 + lpz0
    
    def eval_logp_x0_val_grad(self, particles, t):
        lpz0_grad = numpy.zeros(self.params.shape)
        
        # Calculate l1 according to (19a)
        N = len(particles)
        (xil, zl, Pl) = self.get_states(particles)
        (z0, P0) = self.get_rb_initial(xil)
        (z0_grad, P0_grad) = self.get_rb_initial_grad(xil)
        lpxi0 = self.eval_logp_xi0(xil)
        lpxi0_grad = self.eval_logp_xi0_grad(xil)
        lpz0 = 0.0
        for i in xrange(N):
            l1 = self.calc_l1(zl[i], Pl[i], z0[i], P0[i])
            P0cho = scipy.linalg.cho_factor(P0[i], check_finite=False)
            #(_tmp, ld) = numpy.linalg.slogdet(P0[i])
            ld = numpy.sum(numpy.log(numpy.diagonal(P0cho[0])))*2
            tmp = scipy.linalg.cho_solve(P0cho, l1, check_finite=False)
            lpz0 -= 0.5*(ld + numpy.trace(tmp))
        
            # Calculate gradient
            for j in range(len(self.params)):
                tmp = z0_grad[i][j].dot((zl[i]-z0[i]).T)
                dl1 = -tmp -tmp.T
                lpz0_grad[j] -= 0.5*mlnlg_compute.compute_logprod_derivative(P0cho, P0_grad[i][j], l1, dl1)
                
        return (lpxi0 + lpz0,
                lpxi0_grad + lpz0_grad)
    

    def calc_l2(self, xin, zn, Pn, zl, Pl, A, f, M):
        N = len(xin)
        dim = self.lxi + self.kf.lz
        xn = numpy.hstack((xin, zn))
        perr = numpy.zeros((N, dim, 1))
        mlnlg_compute.compute_pred_err(N, dim, xn, f, A, zl, perr)
        l2 = numpy.zeros((N, dim, dim))
        mlnlg_compute.compute_l2(N, self.lxi, dim, perr, Pn, A, Pl, M, l2)
        return l2
 
    def calc_l2_grad(self, xin, zn, Pn, zl, Pl, A, f, M, f_grad, A_grad):
        N = len(xin)
        dim = self.lxi + self.kf.lz
        xn = numpy.hstack((xin, zn))
        perr = numpy.zeros((N, dim, 1))
        mlnlg_compute.compute_pred_err(N, dim, xn, f, A, zl, perr)
        l2 = numpy.zeros((N, dim, dim))
        mlnlg_compute.compute_l2(N, self.lxi, dim, perr, Pn, A, Pl, M, l2)
#        diff_l2 = compute_l2_grad(perr, len(self.params), self.lxi, zl, Pl, M, A, f_grad, A_grad)
        diff_l2 = numpy.zeros((N, len(self.params), perr.shape[1], perr.shape[1]), dtype=numpy.double, order='C')
        tmp1 = numpy.zeros((self.lxi+self.kf.lz,self.lxi+self.kf.lz), dtype=numpy.double, order='C')
        tmp2 = numpy.zeros((self.lxi+self.kf.lz,self.kf.lz), dtype=numpy.double, order='C')
        if (f_grad != None):
            mlnlg_compute.compute_l2_grad_f(N, len(self.params), self.lxi+self.kf.lz, diff_l2, 
                              perr, f_grad, tmp1)
        if (A_grad != None):
            mlnlg_compute.compute_l2_grad_A(N, len(self.params), self.lxi+self.kf.lz, diff_l2,
                              perr, self.lxi, Pn, zl, Pl, M, A, A_grad, tmp1, tmp2)
        return (l2, diff_l2)   
       
    def eval_logp_xnext(self, particles, x_next, u, t):
        """ Calculate gradient of a term of the I2 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""
        # Calculate l2 according to (16)
        N = len(particles)
        lpxn = 0.0
        
        (_xi, z, P) = self.get_states(particles)
        Mzl = self.get_Mz(particles)
        (xin, zn, Pn) = self.get_states(x_next)
        
        (A, f, Q, _, _, Q_identical) = self.calc_A_f_Q(particles, u, t)
        l2 = self.calc_l2(xin, zn, Pn, z, P, A, f, Mzl)
        if (Q_identical):
            Qcho = scipy.linalg.cho_factor(Q[0], check_finite=False)
            #(_tmp, ld) = numpy.linalg.slogdet(Q[0])
            ld = numpy.sum(numpy.log(numpy.diagonal(Qcho[0])))*2
            for i in xrange(N):
                tmp = scipy.linalg.cho_solve(Qcho, l2[i], check_finite=False)
                lpxn -= 0.5*(ld + numpy.trace(tmp))                                             
        else:
            for i in xrange(N):
                Qcho = scipy.linalg.cho_factor(Q[i], check_finite=False)
                #(_tmp, ld) = numpy.linalg.slogdet(Q[i])
                ld = numpy.sum(numpy.log(numpy.diagonal(Qcho[0])))*2
                tmp = scipy.linalg.cho_solve(Qcho, l2[i], check_finite=False)
                lpxn -= 0.5*(ld + numpy.trace(tmp))
      
        return lpxn

    def eval_logp_xnext_val_grad(self, particles, x_next, u, t):
        """ Calculate gradient of a term of the I2 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""
        N = len(particles)
        lpxn = 0.0
        lpxn_grad = numpy.zeros(self.params.shape)
        
        (A_grad, f_grad, Q_grad) = self.get_pred_dynamics_grad(particles=particles, u=u, t=t)
        if (A_grad == None and f_grad == None and Q_grad == None):
            lpxn = self.eval_logp_xnext(particles, x_next, u, t)
        else:
        
            (_xi, zl, Pl) = self.get_states(particles)
            Mzl = self.get_Mz(particles)
            (xin, zn, Pn) = self.get_states(x_next)
            
            (A, f, Q, _, _, Q_identical) = self.calc_A_f_Q(particles, u, t)
            
    
            dim = self.lxi + self.kf.lz
    
            if (Q_grad == None):
                Q_grad = N*(numpy.zeros((len(self.params), dim, dim)),)
            
            (l2, l2_grad) = self.calc_l2_grad(xin, zn, Pn, zl, Pl, A, f, Mzl, f_grad, A_grad)
            if (Q_identical):
                Qcho = scipy.linalg.cho_factor(Q[0], check_finite=False)
                #(_tmp, ld) = numpy.linalg.slogdet(Q[0])
                ld = numpy.sum(numpy.log(numpy.diagonal(Qcho[0])))*2
                for i in xrange(N):
                    tmp = scipy.linalg.cho_solve(Qcho, l2[i], check_finite=False)
                    lpxn -= 0.5*(ld + numpy.trace(tmp))   
                    for j in xrange(len(self.params)):
                        lpxn_grad[j] -= 0.5*mlnlg_compute.compute_logprod_derivative(Qcho, Q_grad[i][j],
                                                                       l2[i], l2_grad[i][j])                                          
            else:
                for i in xrange(N):
                    Qcho = scipy.linalg.cho_factor(Q[i], check_finite=False)
                    #(_tmp, ld) = numpy.linalg.slogdet(Q[i])
                    ld = numpy.sum(numpy.log(numpy.diagonal(Qcho[0])))*2
                    tmp = scipy.linalg.cho_solve(Qcho, l2[i], check_finite=False)
                    lpxn -= 0.5*(ld + numpy.trace(tmp))
                    for j in xrange(len(self.params)):
                        lpxn_grad[j] -= 0.5*mlnlg_compute.compute_logprod_derivative(Qcho, Q_grad[i][j],
                                                                       l2[i], l2_grad[i][j])
          
        return (lpxn, lpxn_grad)

    def calc_l3(self, y, zl, Pl, Cl, hl):
        N = len(zl)
        l3 = numpy.zeros((N, len(y), len(y)))
        for i in xrange(N):
            meas_diff = self.kf.measurement_diff(y,zl[i],Cl[i], hl[i]) 
            l3[i] = meas_diff.dot(meas_diff.T) + Cl[i].dot(Pl[i]).dot(Cl[i].T)
        return l3
    
    def calc_l3_grad(self, y, zl, Pl, Cl, hl, C_grad, h_grad):
        N = len(zl)
        l3 = numpy.zeros((N, len(y), len(y)))
        diff_l3 = numpy.zeros((N, len(self.params), len(y), len(y)))
        
        for i in xrange(N):
            meas_diff = self.kf.measurement_diff(y,zl[i],Cl[i], hl[i]) 
            l3[i] = meas_diff.dot(meas_diff.T) + Cl[i].dot(Pl[i]).dot(Cl[i].T)
            
            if (C_grad != None):
                C_grad = N*(numpy.zeros((len(self.params), len(y), self.kf.lz)),)
                for j in xrange(len(self.params)):
                    tmp2 = C_grad[i][j].dot(Pl[i]).dot(Cl[i].T)
                    tmp = C_grad[i][j].dot(zl[i]).dot(meas_diff.T)
                    diff_l3[i][j] += -tmp -tmp.T + tmp2 + tmp2.T
            if (h_grad != None):
                for j in xrange(len(self.params)):
                    tmp = h_grad[i][j].dot(meas_diff.T)
                    diff_l3[i][j] += -tmp -tmp.T
            
        return (l3, diff_l3)
    
    def eval_logp_y(self, particles, y, t):
        """ Calculate a term of the I3 integral approximation
        and its gradient as specified in [1]"""
        N = len(particles)
        (y, Cz, hz, Rz, _, _, Rz_identical) = self.get_meas_dynamics_int(particles, y, t)
        (_xil, zl, Pl) = self.get_states(particles)
        logpy = 0.0
        l3 = self.calc_l3(y, zl, Pl, Cz, hz)
        if (Rz_identical):
            Rzcho = scipy.linalg.cho_factor(Rz[0], check_finite=False)
            #(_tmp, ld) = numpy.linalg.slogdet(Rz[0])
            ld = numpy.sum(numpy.log(numpy.diagonal(Rzcho[0])))*2
            for i in xrange(N):
                tmp = scipy.linalg.cho_solve(Rzcho, l3[i], check_finite=False)
                logpy -= 0.5*(ld + numpy.trace(tmp))
        else:
            for i in xrange(N):
            # Calculate l3 according to (19b)
                Rzcho = scipy.linalg.cho_factor(Rz[i], check_finite=False)
                #(_tmp, ld) = numpy.linalg.slogdet(Rz[i])
                ld = numpy.sum(numpy.log(numpy.diagonal(Rzcho[0])))*2
                tmp = scipy.linalg.cho_solve(Rzcho, l3[i], check_finite=False)
                logpy -= 0.5*(ld + numpy.trace(tmp))

        return logpy

    def eval_logp_y_val_grad(self, particles, y, t):
        """ Calculate a term of the I3 integral approximation
        and its gradient as specified in [1]"""
        
        N = len(particles)
        logpy = 0.0
        lpy_grad = numpy.zeros(self.params.shape)
        (y, Cz, hz, Rz, _, _, Rz_identical) = self.get_meas_dynamics_int(particles, y, t)
        (C_grad, h_grad, R_grad) = self.get_meas_dynamics_grad(particles=particles, y=y, t=t)
        if (C_grad == None and h_grad == None and R_grad == None):
            logpy = self.eval_logp_y(particles, y, t)
        else:
            if (R_grad == None):
                R_grad = N*(numpy.zeros((len(self.params), len(y), len(y))),)
                
            (_xil, zl, Pl) = self.get_states(particles)
            
            (l3, l3_grad) = self.calc_l3_grad(y, zl, Pl, Cz, hz, C_grad, h_grad)
            
            if (Rz_identical):
                Rzcho = scipy.linalg.cho_factor(Rz[0], check_finite=False)
                #(_tmp, ld) = numpy.linalg.slogdet(Rz[0])
                ld = numpy.sum(numpy.log(numpy.diagonal(Rzcho[0])))*2
                for i in xrange(N):
                    tmp = scipy.linalg.cho_solve(Rzcho, l3[i], check_finite=False)
                    logpy -= 0.5*(ld + numpy.trace(tmp))
                    for j in range(len(self.params)):
                        lpy_grad[j] -= 0.5*mlnlg_compute.compute_logprod_derivative(Rzcho, R_grad[i][j],
                                                                      l3[i], l3_grad[i][j])
            else:
                for i in xrange(N):
                    Rzcho = scipy.linalg.cho_factor(Rzcho, check_finite=False)
                    #(_tmp, ld) = numpy.linalg.slogdet(Rz[i])
                    ld = numpy.sum(numpy.log(numpy.diagonal(Rzcho[0])))*2
                    tmp = scipy.linalg.cho_solve(Rzcho, l3[i])
                    logpy -= 0.5*(ld + numpy.trace(tmp))
                    for j in range(len(self.params)):
                        lpy_grad[j] -= 0.5*mlnlg_compute.compute_logprod_derivative(Rzcho, R_grad[i][j],
                                                                      l3[i], l3_grad[i][j])

        return (logpy, lpy_grad)


class MixedNLGaussianInitialGaussian(MixedNLGaussian):
    def __init__(self, xi0, z0, Pxi0=None, Pz0=None, **kwargs):
        
                # No uncertainty in initial state
        self.xi0 = numpy.copy(xi0).reshape((-1,1))
        if (Pxi0 == None):
            self.Pxi0 = numpy.zeros((len(self.xi0),len(self.xi0)))
        else:
            self.Pxi0 = numpy.copy((Pxi0))
        if (Pz0 == None):
            self.Pz0 = numpy.zeros((len(self.z0),len(self.z0)))
        else:
            self.Pz0 = numpy.copy((Pz0))
        self.z0 =  numpy.copy(z0).reshape((-1,1))
        self.Pz0 = numpy.copy(Pz0)
        super(MixedNLGaussianInitialGaussian, self).__init__(lxi=len(self.xi0),
                                                             lz=len(self.z0),
                                                             **kwargs)

    def create_initial_estimate(self, N):
        dim = self.lxi + self.kf.lz + self.kf.lz**2
        particles = numpy.empty((N, dim))
        
        for i in xrange(N):
            particles[i,0:self.lxi] = numpy.random.multivariate_normal(self.xi0.ravel(), self.Pxi0)
            particles[i,self.lxi:(self.lxi+self.kf.lz)] = numpy.copy(self.z0).ravel()
            particles[i,(self.lxi+self.kf.lz):] = numpy.copy(self.Pz0).ravel()  
        return particles     

    def get_rb_initial(self, xi0):
        """ Default implementation has no dependence on xi, override if needed """
        N = len(xi0)
        z_list = numpy.repeat(self.z0.reshape((1,self.kf.lz,1)), N, 0)
        P_list =  numpy.repeat(self.Pz0.reshape((1,self.kf.lz,self.kf.lz)), N, 0)
        return (z_list, P_list)

    def get_rb_initial_grad(self, xi0):
        """ Default implementation has no dependence on xi, override if needed """
        N = len(xi0)
        return (N*(numpy.zeros((N, len(self.params), self.kf.lz, 1)),),
                N*(numpy.zeros((N, len(self.params), self.kf.lz, self.kf.lz)),))
    
    def eval_logp_xi0(self, xil):
        """ Calculate gradient of a term of the I1 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""    
            
        N = len(xil)
        res = numpy.empty(N)
        Pchol = scipy.linalg.cho_factor(self.Pxi0, check_finite=False)
        for i in xrange(N):
            res[i] = kalman.lognormpdf_cho(xil[i] - self.xi0, Pchol)
        return res
    
    def get_xi_intitial_grad(self, N):
        return (N*(numpy.zeros((len(self.params), self.lxi, 1)),),
                N*(numpy.zeros((len(self.params), self.lxi, self.lxi)),))
        
    def eval_logp_xi0_grad(self, xil):
        """ Evaluate probabilty of xi0 """
        N = len(xil)
        (xi0_grad, Pxi0_grad) = self.get_xi_intitial_grad(N)
        lpxi0_grad = numpy.zeros(self.params.shape)
        Pxi0cho = scipy.linalg.cho_factor(self.Pxi0, check_finite=False)
        for i in xrange(N):
            tmp = xil[i]-self.xi0
            l0 = tmp.dot(tmp.T)
            for j in range(len(self.params)):
                tmp2 = tmp.dot(xi0_grad[i][j].T)
                l0_grad = tmp2 + tmp2.T
                lpxi0_grad[j] -= 0.5*mlnlg_compute.compute_logprod_derivative(Pxi0cho, Pxi0_grad[i][j], l0, l0_grad)
                
        return lpxi0_grad
