""" Collection of functions and classes used for Particle Filtering/Smoothing """
import pyparticleest.kalman as kalman
from pyparticleest.part_utils import RBPSBase
import numpy
import numpy.random
import copy

class MixedNLGaussian(RBPSBase):
    """ Base class for particles of the type mixed linear/non-linear with additive gaussian noise.
    
        Implement this type of system by extending this class and provide the methods for returning 
        the system matrices at each time instant  """
    def __init__(self, lxi, lz, Az=None, C=None, Qz=None, R=None, fz=None,
                 Axi=None, Qxi=None, Qxiz=None, fxi=None, h=None, params=None, t0=0):
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
                                                     hz=h, fz=fz,
                                                     t0=t0)

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
            self.kf.measure_full(y=xi_next[i].reshape((self.lxi,1)), z=zl[i], P=Pl[i], C=Axi[i], h_k=fxi[i], R=Qxi[i])
        
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
            tmp = Qxiz[i].T.dot(numpy.linalg.inv(Qxi[i]))
            Acond.append(Az[i] - tmp.dot(Axi[i]))
            #Acond.append(Az[i])
            fcond.append(fz[i] +  tmp.dot(xi_next[i] - fxi[i]))
            # TODO, shouldn't Qz be affected? Why wasn't it before?
            Qcond.append(Qz[i])
        
        return (Acond, fcond, Qcond)
        
        
    
    def cond_predict(self, particles, xi_next, u, t):
        #Compensate for noise correlation
        (Az, fz, Qz) = self.calc_cond_dynamics(particles=particles, xi_next=xi_next, u=u, t=t)
        (xil, zl, Pl) = self.get_states(particles)
        for i in xrange(len(zl)):
            (zl[i], Pl[i]) = self.kf.predict_full(z=zl[i], P=Pl[i], A=Az[i], f_k=fz[i], Q=Qz[i])
        
        # Predict next states conditioned on eta_next
        self.set_states(particles, xil, zl, Pl)
        
    def measure(self, particles, y, t):
        """ Return the log-pdf value of the measurement """
        
        (xil, zl, Pl) = self.get_states(particles)
        N = len(particles)
        (y, Cz, hz, Rz, _, _, _) = self.get_meas_dynamics_int(particles=particles, y=y, t=t)
            
        lyz = numpy.empty(N)
        for i in xrange(len(zl)):
            # Predict z_{t+1}
            lyz[i] = self.kf.measure_full(y=y, z=zl[i], P=Pl[i], C=Cz[i], h_k=hz[i], R=Rz[i])
            
        self.set_states(particles, xil, zl, Pl)
        return lyz

    def calc_A_f_Q(self, particles, u, t):
        N = len(particles)
        (Az, fz, Qz, Az_identical, fz_identical, Qz_identical) = self.get_lin_pred_dynamics_int(particles=particles, u=u, t=t)
        (Axi, fxi, Qxi, Axi_identical, fxi_identical, Qxi_identical) = self.get_nonlin_pred_dynamics_int(particles=particles, u=u, t=t)
        Qxiz = self.get_cross_covariance(particles=particles, u=u, t=t)
        Qxiz_identical = False

        if (Qxiz == None):
            Qxiz_identical = True
            if (self.Qxiz == None):
                Qxiz = N*(numpy.zeros((Qxi[0].shape[0],Qz[0].shape[0])),)
            else:
                Qxiz = N*(self.Qxiz,)
        
        if (Az_identical and Axi_identical):
            A = N*(numpy.vstack((Axi[0], Az[0])),)
        else:
            A = list()
            for i in xrange(N):
                A.append(numpy.vstack((Axi[i], Az[i])))
                
        if (fxi_identical and fz_identical):
            f = N*(numpy.vstack((fxi[0], fz[0])),)
        else:
            f = list()
            for i in xrange(N):
                f.append(numpy.vstack((fxi[i], fz[i])))
                
        if (Qxi_identical and Qz_identical and Qxiz_identical):
            Q = N*(numpy.vstack((numpy.hstack((Qxi[0], Qxiz[0])),
                              numpy.hstack((Qxiz[0].T, Qz[0])))),)
        else:
            Q = list()
            for i in xrange(N):
                Q.append(numpy.vstack((numpy.hstack((Qxi[i], Qxiz[i])),
                              numpy.hstack((Qxiz[i].T, Qz[i])))))
                
        return (A, f, Q)
    
    def calc_A_f_Q_grad(self, particles, u, t):
        N = len(particles)
        (A_grad, f_grad, Q_grad) = self.get_pred_dynamics_grad(particles=particles, u=u, t=t)

        dim = self.lxi + self.kf.lz

        if (A_grad == None):
            A_grad = N*(numpy.zeros((dim, dim)),)
        if (f_grad == None):
            f_grad = N*(numpy.zeros((dim, 1)),)
        if (Q_grad == None):
            Q_grad = N*(numpy.zeros((dim, dim)),)
                
        return (A_grad, f_grad, Q_grad)

    def calc_C_h_R_grad(self, particles, y, t):
        N = len(particles)
        (C_grad, h_grad, R_grad) = self.get_meas_dynamics_grad(particles=particles, y=y, t=t)

        if (C_grad == None):
            C_grad = N*(numpy.zeros((len(y), self.kf.lz)),)
        if (h_grad == None):
            h_grad = N*(numpy.zeros((len(y), 1)),)
        if (R_grad == None):
            R_grad = N*(numpy.zeros((len(y), len(y))),)
                
        return (C_grad, h_grad, R_grad)    
    
    def next_pdf_max(self, particles, u, t):
        """ Implements the fwd_peak_density function for MixedNLGaussian models """
        N = len(particles)
        pmax = numpy.empty(N)
        (_, _, Pl) = self.get_states(particles)
        (A, _, Q) = self.calc_A_f_Q(particles, u=u, t=t)
        dim=self.lxi + self.kf.lz
        zeros = numpy.zeros((dim,1))
        for i in xrange(N):
            self.Sigma = Q[i] + A[i].dot(Pl[i]).dot(A[i].T)
            pmax[i] = kalman.lognormpdf(zeros, zeros, self.Sigma)
        
        return numpy.max(pmax)
        
    def next_pdf(self, particles, next_part, u, t):
        """ Implements the next_pdf function for MixedNLGaussian models """
        
        N = len(particles)
        Nn = len(next_part)
        if (N > 1 and Nn == 1):
            next_part = numpy.repeat(next_part, N, 0)
        lpx = numpy.empty(N)
        (_, zl, Pl) = self.get_states(particles)
        (A, f, Q) = self.calc_A_f_Q(particles, u=u, t=t)
        
        for i in xrange(N):
            x_next = next_part[i,:self.lxi+self.kf.lz].reshape((self.lxi+self.kf.lz,1))
            xp = f[0] + A[i].dot(zl[i])
            Sigma = A[i].dot(Pl[i]).dot(A[i].T) + Q[i]
            lpx[i] = kalman.lognormpdf(x_next,mu=xp,S=Sigma)
        
        return lpx
    
    def sample_smooth(self, particles, next_part, u, t):
        """ Implements the sample_smooth function for MixedNLGaussian models """
        M = len(particles)
        res = numpy.zeros((M,self.lxi+self.kf.lz + 2*self.kf.lz**2))
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
            xi = copy.copy(xil[j]).ravel()
            z = numpy.random.multivariate_normal(zl[j].ravel(), 
                                                 Pl[j]).ravel()
            res[j,:self.lxi+self.kf.lz] = numpy.hstack((xi, z))
        return res

    def copy_ind(self, particles, new_ind):
        new_part = particles[new_ind]
        return new_part
    
    def set_states(self, particles, xi_list, z_list, P_list):
        """ Set the estimate of the Rao-Blackwellized states """
        N = len(particles)
        zend = self.lxi+self.kf.lz
        Pend = zend+self.kf.lz**2
#        for i in xrange(N):
#            particles[i,:self.lxi] = xi_list[i].ravel()
#            particles[i,self.lxi:zend] = z_list[i].ravel()
#            particles[i,zend:Pend] = P_list[i].ravel()
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
        
#        xil = list()
#        zl = list()
#        Pl = list()
#
#        for part in particles:
#            xil.append(part[:self.lxi].reshape(self.lxi,1))
#            zl.append(part[self.lxi:zend].reshape(self.kf.lz,1))
#            Pl.append(part[zend:Pend].reshape(self.kf.lz,self.kf.lz))
        
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
#        Mz = list()
#        for part in smooth_particles:
#            Mz.append(part[Pend:Mend].reshape(self.kf.lz, self.kf.lz))
        Mz = smooth_particles[:,Pend:Mend].reshape((N, self.kf.lz, self.kf.lz))
        return Mz
    
    def set_Mz(self, smooth_particles, Mz):
        N = len(smooth_particles)
        zend = self.lxi+self.kf.lz
        Pend = zend+self.kf.lz**2
        Mend = Pend + self.kf.lz**2
#        for i in xrange(N):
#            smooth_particles[i,Pend:Mend] = Mz[i].ravel()
            
        smooth_particles[:,Pend:Mend] = Mz.reshape((N, self.kf.lz**2))
    
#    def eval_1st_stage_weight(self, u,y):
##        eta_old = copy.deepcopy(self.get_nonlin_state())
##        lin_old = copy.deepcopy(self.get_lin_est())
##        t_old = self.t
#        self.prep_update(u)
#        noise = numpy.zeros_like(self.eta)
#        self.update(u, noise)
#        
#        yn = self.prep_measure(y)
#        logpy = self.measure(yn)
#        
#        # Restore state
##        self.set_lin_est(lin_old)
##        self.set_nonlin_state(eta_old)
##        self.t = t_old
#        
#        return logpy
    
    def set_params(self, params):
        self.params = numpy.copy(params).reshape((-1,1))

    def get_pred_dynamics_grad(self, particles, u, t):
        """ Override this method if (A, f, Q) depends on the parameters """
        return (None, None, None)
    
    def get_meas_dynamics_grad(self, particles, y, t):
        """ Override this method if (C, h, R) depends on the parameters """
        return (None, None, None)
    
    def calc_logprod_derivative(self, A, dA, B, dB):
        """ I = logdet(A)+Tr(inv(A)*B)
            dI/dx = Tr(inv(A)*(dA - dA*inv(A)*B + dB) """
            
        tmp = numpy.linalg.solve(A, B)
        tmp2 = dA + dB - dA.dot(tmp)
        return numpy.trace(numpy.linalg.solve(A,tmp2))



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
            (_tmp, ld) = numpy.linalg.slogdet(P0[i])
            tmp = numpy.linalg.solve(P0[i], l1)
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
        lpxi0_grad = self.eval_grad_eta0_logpdf(xil)
        lpz0 = 0.0
        for i in xrange(N):
            l1 = self.calc_l1(zl[i], Pl[i], z0[i], P0[i])
            (_tmp, ld) = numpy.linalg.slogdet(P0[i])
            tmp = numpy.linalg.solve(P0[i], l1)
            lpz0 -= 0.5*(ld + numpy.trace(tmp))
        
            # Calculate gradient
            for j in range(len(self.params)):
                tmp = z0_grad[i][j].dot((zl[i]-z0[i]).T)
                dl1 = -tmp -tmp.T
    
                lpz0_grad[j] -= 0.5*self.calc_logprod_derivative(P0[i], P0_grad[i], l1, dl1)
                
        return (lpxi0 + lpz0,
                lpxi0_grad + lpz0_grad)
    
    
    def calc_l2(self, xin, zn, Pn, z, P, Axi, Az, f, M):
        xn = numpy.vstack((xin, zn))
        A = numpy.vstack((Axi, Az))
        predict_err = xn - f - A.dot(z)
        
        l2 = predict_err.dot(predict_err.T) +A.dot(P).dot(A.T)
        
        tmp = -Axi.dot(M)
        l2[len(xin):,:len(xin)] += tmp.T
        l2[:len(xin),len(xin):] += tmp
        
        tmp2 = Pn - M.T.dot(P) - Az.dot(M)        

        l2[len(xin):,len(xin):] += tmp2
        return (l2, predict_err)
 
    def calc_l2_grad(self, l2, predict_err, z, P, Mz, f, f_grad, A, A_grad):
        
        diff_l2 = numpy.zeros((len(self.params), l2.shape[0], l2.shape[1]))
        
        for j in range(len(self.params)):
                 
            tmp = (f_grad[j] + A_grad[j].dot(z)).dot(predict_err.T)
            diff_l2[j,:,:] = -tmp - tmp.T
            tmp = A_grad[j].dot(P).dot(A.T)
            diff_l2[j,:,:] += tmp + tmp.T
            tmp = -A_grad[j].dot(Mz)
            diff_l2[j,:,self.lxi:] +=  tmp
            diff_l2[j, self.lxi:, :] += tmp.T

        return diff_l2   
       
    def eval_logp_xnext(self, particles, x_next, u, t):
        """ Calculate gradient of a term of the I2 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""
        # Calculate l2 according to (16)
        N = len(particles)
        lpxn = numpy.empty(N)
        
        (_xi, z, P) = self.get_states(particles)
        Mzl = self.get_Mz(particles)
        (xin, zn, Pn) = self.get_states(x_next)
        
        (A, f, Q) = self.calc_A_f_Q(particles, u)
        
        for i in xrange(N):
            Axi = A[i][:self.lxi]
            Az = A[i][self.lxi:]
            (l2, _pe) = self.calc_l2(xin[i], zn[i], Pn[i], z[i], P[i], Axi, Az, f[i], Mzl[i])
            (_tmp, ld) = numpy.linalg.slogdet(Q[i])
            tmp = numpy.linalg.solve(Q[i], l2)
            lpxn[i] = -0.5*(ld + numpy.trace(tmp))
      
        return lpxn



    def eval_logp_xnext_grad(self, particles, x_next, u, t):
        """ Calculate gradient of a term of the I2 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""
            
        N = len(particles)
        lpxn = numpy.empty(N)
        
        (_xi, z, P) = self.get_states(particles)
        Mzl = self.get_Mz(particles)
        (xin, zn, Pn) = self.get_states(x_next)
        
        (A, f, Q) = self.calc_A_f_Q(particles, u)
        (A_grad, f_grad, Q_grad) = self.calc_A_f_Q_grad(particles, u)
        
        lpxn_grad = numpy.zeros(self.params.shape)
        
        for i in xrange(N):
            Axi = A[i][:self.lxi]
            Az = A[i][self.lxi:]
            (l2, pe) = self.calc_l2(xin[i], zn[i], Pn[i], z[i], P[i], Axi, Az, f[i], Mzl[i])
            (_tmp, ld) = numpy.linalg.slogdet(Q[i])
            tmp = numpy.linalg.solve(Q[i], l2)
            lpxn[i] = -0.5*(ld + numpy.trace(tmp))
      
            l2_grad = self.calc_l2_grad(self, l2, pe, z[i], P[i], Mzl[i], f[i], f_grad[i], A[i], A_grad[i])
            
            for j in xrange(len(self.params)):
                lpxn_grad[j] -= 0.5*self.calc_logprod_derivative(self.Q, Q_grad, l2, l2_grad[j])
            
        return (lpxn, lpxn_grad)

    def calc_l3(self, y, z, P, C, h):
        meas_diff = self.kf.measurement_diff(y,z,C, h) 
        l3 = meas_diff.dot(meas_diff.T)
        l3 += C.dot(P).dot(C.T)
        return l3
    
    def calc_l3_grad(self, y, z, P, C, h, C_grad, h_grad):
        meas_diff = self.kf.measurement_diff(y,z,C, h) 
        l3 = meas_diff.dot(meas_diff.T)
        l3 += C.dot(P).dot(C.T)
        
        diff_l3 = numpy.zeros((len(self.params), l3.shape[0], l3.shape[1]))
        
        for j in xrange(len(self.params)):
            tmp2 = self.C_grad[j].dot(P).dot(C.T)
            tmp = self.C_grad[j].dot(z).dot(meas_diff.T)
            tmp += self.h_grad[j].dot(meas_diff.T)
            diff_l3[j] += -tmp -tmp.T + tmp2 + tmp2.T
        
        return diff_l3
    
    def eval_logp_y(self, particles, y, t):
        """ Calculate a term of the I3 integral approximation
        and its gradient as specified in [1]"""
        N = len(particles)
        (y, Cz, hz, Rz, _, _, _) = self.get_meas_dynamics_int(particles, y, t)
        (_xil, zl, Pl) = self.get_states(particles)
        logpy = 0.0
        for i in xrange(N):
        # Calculate l3 according to (19b)
            l3 = self.calc_l3(y, zl[i], Pl[i], Cz[i], hz[i])
        
            (_tmp, ld) = numpy.linalg.slogdet(Rz[i])
            tmp = numpy.linalg.solve(Rz[i], l3)
            logpy -= 0.5*(ld + numpy.trace(tmp))

        return logpy


        
    def eval_logp_y_grad(self, particles, y, t):
        """ Calculate a term of the I3 integral approximation
        and its gradient as specified in [1]"""
        
        N = len(particles)
        (y, Cz, hz, Rz, _, _, _) = self.get_meas_dynamics_int(particles, y, t)
        (C_grad, h_grad, R_grad) = self.calc_C_h_R_grad(particles, y, t)
        (_xil, zl, Pl) = self.get_states(particles)
        logpy = 0.0
        lpy_grad = numpy.zeros(self.params.shape)
        for i in xrange(N):
        # Calculate l3 according to (19b)
            l3 = self.calc_l3(y, zl[i], Pl[i], Cz[i], hz[i])
        
            (_tmp, ld) = numpy.linalg.slogdet(Rz[i])
            tmp = numpy.linalg.solve(Rz[i], l3)
            logpy -= 0.5*(ld + numpy.trace(tmp))


            # Calculate gradient
            l3_grad = self.calc_l3_grad(y, zl[i], Pl[i], Cz[i], hz[i], C_grad[i], h_grad[i])
            for j in range(len(self.params)):
                lpy_grad[i] -= 0.5*self.calc_logprod_derivative(Rz[i], R_grad[i], l3, l3_grad[j])

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
#        z_list = list()
#        P_list = list()
#        for i in xrange(N):
#            z_list.append(numpy.copy(self.z0).reshape((-1,1)))
#            P_list.append(numpy.copy(self.Pz0))
        z_list = numpy.repeat(self.z0.reshape((1,self.kf.lz,1)), N, 0)
        P_list =  numpy.repeat(self.Pz0.reshape((1,self.kf.lz,self.kf.lz)), N, 0)
        return (z_list, P_list)
    
    def eval_logp_xi0(self, xil):
        """ Calculate gradient of a term of the I1 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""    
            
        N = len(xil)
        return kalman.lognormpdf_vec(xil, N*(self.xi0,), N*(self.Pxi0,))
    
    # TODO, FIXME!
    def eval_logp_xi0_grad(self, xil):
        """ Evaluate logprob of the initial non-linear state eta,
            default implementation assumes all are equal, override this
            if another behavior is desired """
        return numpy.zeros(self.params.shape)
