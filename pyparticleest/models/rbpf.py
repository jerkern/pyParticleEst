import abc
import pyparticleest.interfaces as interfaces
import pyximport
pyximport.install(inplace=True)
import pyparticleest.utils.kalman as kalman
import numpy


class RBPFBase(interfaces.ParticleFiltering):
    """ Base class for Rao-Blackwellized particles """
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, lz, Az=None, fz=None, Qz=None,
                 C=None ,hz=None, R=None):
        
        self.kf = kalman.KalmanSmoother(lz, A=Az, C=C, 
                                        Q=Qz, R=R,
                                        f_k=fz, h_k=hz)
        
    def set_dynamics(self, Az=None, C=None, Qz=None, R=None, fz=None, hz=None):
        return self.kf.set_dynamics(Az, C, Qz, R, fz, hz)
    
    def get_nonlin_pred_dynamics(self, particles, u, t):
        """ Return matrices describing affine relation of next
            nonlinear state conditioned on current linear state
            
            xi_{t+1]} = A_xi * z_t + f_xi + v_xi, v_xi ~ N(0,Q_xi)
            
            Return (A_xi, f_xi, Q_xi) where each element is a list
            with the corresponding matrix for each particle. None indicates
            that the matrix is identical for all particles and the value stored
            in this class should be used instead
            """
        return (None, None, None)
    
    def get_nonlin_pred_dynamics_int(self, particles, u, t):
        (Axi, fxi, Qxi) = self.get_nonlin_pred_dynamics(particles, u=u, t=t)
        N = len(particles)
        Axi_identical = False
        fxi_identical = False
        Qxi_identical = False
        # This is probably not so nice performance-wise, but will
        # work initially to profile where the bottlenecks are.
        if (Axi == None):
            Axi=N*(self.Axi,)
            Axi_identical = True
        if (fxi == None):
            fxi=N*(self.fxi,)
            fxi_identical = True
        if (Qxi == None):
            Qxi= N*(self.Qxi,)
            Qxi_identical = True
        return (Axi, fxi, Qxi, Axi_identical, fxi_identical, Qxi_identical)
    
    def get_lin_pred_dynamics(self, particles, u, t):
        """ Return matrices describing affine relation of next
            nonlinear state conditioned on current linear state
            
            \z_{t+1]} = A_z * z_t + f_z + v_z, v_z ~ N(0,Q_z)
            
            conditioned on the value of xi_{t+1}. 
            (Not the same as the dynamics unconditioned on xi_{t+1})
            when for example there is a noise correlation between the 
            linear and nonlinear state dynamics) 
            """
        return (None, None, None)
    
    def get_lin_pred_dynamics_int(self, particles, u, t):
        N = len(particles)
        (Az, fz, Qz) = self.get_lin_pred_dynamics(particles, u=u, t=t)
        Az_identical = False
        fz_identical = False
        Qz_identical = False
        if (Az == None):
            #Az=numpy.repeat(self.kf.A[numpy.newaxis,:,:], N, axis=0)
            Az=N*(self.kf.A,)
            Az_identical = True
        if (fz == None):
            #fz=numpy.repeat(self.kf.f_k[numpy.newaxis,:,:], N, axis=0)
            fz=N*(self.kf.f_k,)
            fz_identical = True
        if (Qz == None):
            #Qz=numpy.repeat(self.kf.Q[numpy.newaxis,:,:], N, axis=0)
            Qz=N*(self.kf.Q,)
            Qz_identical = True
            
        return (Az, fz, Qz, Az_identical, fz_identical, Qz_identical)
    
    def get_meas_dynamics(self, particles, y, t):
        return (y, None, None, None)
    
    def get_meas_dynamics_int(self, particles, y, t):
        N=len(particles)
        (y, Cz, hz, Rz) = self.get_meas_dynamics(particles=particles, y=y, t=t)
        Cz_identical = False
        hz_identical = False
        Rz_identical = False
        if (Cz == None):
            if (self.kf.C == None and hz != None):
                Cz=N*(numpy.zeros((len(hz[0]), self.kf.lz)),)
            else:
                Cz=N*(self.kf.C,)
            #Cz=N*(self.kf.C,)
            Cz_identical = True
        if (hz == None):
            hz=N*(self.kf.h_k,)
            hz_identical = True
        if (Rz == None):
            Rz=N*(self.kf.R,)
            Rz_identical = True
        return (y, Cz, hz, Rz, Cz_identical, hz_identical, Rz_identical)
    
# This is not implemented  
#    def get_condlin_meas_dynamics(self, y, xi_next, particles):
#        return (y, None, None, None)
    
    def update(self, particles, u, t, noise):
        """ Update estimate using noise as input """
        # Calc (xi_{t+1} | xi_t, z_t, y_t)
        xin = self.calc_xi_next(particles=particles, u=u, t=t, noise=noise)
        # Calc (z_{t+1} | xi_{t+1}, y_t)
        self.cond_predict(particles=particles, xi_next=xin, u=u, t=t)

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

    def copy_ind(self, particles, new_ind=None):
        if (new_ind != None):
            return numpy.copy(particles[new_ind])
        else:
            return numpy.copy(particles)

    
class RBPSBase(RBPFBase, interfaces.FFBSi):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def get_rb_initial(self, xi_initial):
        pass
    
    def post_smoothing(self, st):
        """ Kalman smoothing of the linear states conditioned on the non-linear
            trajetory """
        T = st.traj.shape[0]
        M = st.traj.shape[1]

        particles = numpy.copy(st.traj[0])
        lx = particles.shape[1]
        straj = numpy.zeros((T,M, self.lxi+self.kf.lz+2*self.kf.lz**2))
        (xil, _zl, _Pl) = self.get_states(particles)
        (z0, P0) = self.get_rb_initial(xil)
        self.set_states(particles, xil, z0, P0)

        for i in xrange(T-1):
            if (st.y[i] != None):
                self.measure(particles, y=st.y[i], t=st.t[i])
            (xin, _zn, _Pn) = self.get_states(st.traj[i+1])
            #self.meas_xi_next(particles, xin, u=st.u[i], t=st.t[i])
            straj[i,:,:lx] = particles

            self.cond_predict(particles, xin, u=st.u[i], t=st.t[i])
            
        if (st.y[-1] != None):
            self.measure(particles, y=st.y[-1], t=st.t[-1])
        
        straj[-1,:,:lx] = particles
        
        # Backward smoothing
        for i in reversed(xrange(T-1)):
            (xin, zn, Pn) = self.get_states(straj[i+1])
            particles = straj[i]
            self.meas_xi_next(particles, xin, u=st.u[i], t=st.t[i])
            (xi, z, P) = self.get_states(particles)
            (Al, fl, Ql) = self.calc_cond_dynamics(particles, xin, u=st.u[i], t=st.t[i])
            for j in xrange(M):
                
                (zs, Ps, Ms) = self.kf.smooth(z[j], P[j], zn[j], Pn[j],
                                              Al[j], fl[j], Ql[j])
                self.set_states(straj[i,j:j+1,:], xi[j], zs[numpy.newaxis], Ps[numpy.newaxis])
                self.set_Mz(straj[i,j:j+1,:], Ms[numpy.newaxis])

        return straj

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