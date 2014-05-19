import numpy
import math
from pyparticleest.models.hierarchial import HierarchicalRSBase
import pyparticleest.filter as pf
import pyparticleest.utils.kalman as kalman
import matplotlib.pyplot as plt
import scipy.stats

def generate_dataset(steps, P0_xi, P0_z, Q_xi, Q_z, R_xi, R_z):
    xi = numpy.zeros((1, steps+1))
    z = numpy.zeros((2, steps+1))
    y = numpy.zeros((steps,2))
    xi[:,0] = numpy.random.normal(0.0, math.sqrt(P0_xi))
    z[:,0] = numpy.random.multivariate_normal(numpy.zeros((2,)), P0_z)
    for k in range(1,steps+1):
        xi[:,k] = xi[:,k-1] + numpy.random.normal(0.0, Q_xi)
        Ak = numpy.asarray(((math.cos(xi[:,k-1]), math.sin(xi[:,k-1])),
                            (-math.sin(xi[:,k-1]), math.cos(xi[:,k-1]))
                            ))
        z[:,k] = Ak.dot(z[:,k-1]) + numpy.random.multivariate_normal(numpy.zeros((2,)), Q_z)
        C = numpy.asarray(((math.cos(xi[:,k-1]), math.sin(xi[:,k-1]))))
        y[k-1,0] = xi[:,k] + numpy.random.normal(0.0, math.sqrt(R_xi))
        y[k-1,1] = C.dot(z[:,k]) + numpy.random.normal(0.0, math.sqrt(R_z))
        
    x = numpy.vstack((xi, z))
    return (x,y)

def wmean(logw, val):
    w = numpy.exp(logw)
    w = w / sum(w)
    w.reshape((1,-1))
    return w.dot(val)

class Model(HierarchicalRSBase):
    """ xi_{k+1} = xi_k + v_xi_k, v_xi_k ~ N(0,Q_xi)
        z_{k+1} = ((cos(xi_k) sin(xi_k)),
                   (-sin(xi_k) cos(xi_k)) * z_{k} + v_z_k, v_z_k ~ N(0, Q_z) 
        y_z_k = ((cos(xi_k) sin(xi_k))*z_k + e_k, e_k ~ N(0,R_z),
        y_xi_k = xi_k + e_xi_k, e_xi_k ~ N(0,R_xi)
        x(0) ~ N(0,P0) """
    
    def __init__(self, P0_xi, P0_z, Q_xi, Q_z, R_xi, R_z):
        self.P0_xi = numpy.copy(P0_xi)
        self.Q_xi= numpy.copy(Q_xi)
        self.R_xi= numpy.copy(R_xi)
        self.P0_z = numpy.copy(P0_z)
        self.R_z=numpy.copy(R_z)
        fz = numpy.zeros((2,1))
        hz = numpy.zeros((1,1))
        self.pn_count = 0
        super(Model, self).__init__(len_xi=1, len_z=2, fz=fz, Qz=Q_z, hz=hz, R=R_z)
        
    def create_initial_estimate(self, N):
        particles = numpy.empty((N,), dtype=numpy.ndarray)
               
        for i in xrange(N):
            particles[i] = numpy.empty(7)
            particles[i][0] = numpy.random.normal(0.0, math.sqrt(self.P0_xi))
            particles[i][1:3] = numpy.zeros((1, 2))
            particles[i][3:] = numpy.copy(self.P0_z).ravel()  
        return particles
    
    def get_rb_initial(self, xi0):
        return (numpy.zeros((self.kf.lz,1)),
                numpy.copy(self.P0_z))
        
    def sample_process_noise(self, particles, u, t):
        """ Return process noise for input u """
        N = len(particles)
        return numpy.random.normal(0.0, math.sqrt(self.Q_xi), (N,)) 

    def calc_xi_next(self, particles, u, t, noise):
        N = len(particles)
        xi_next = numpy.empty(N)
        for i in xrange(N):
            xi_next[i] =  particles[i][0] + noise[i]
        return xi_next
    
    def next_pdf_xi(self, particles, next_xi, u, t):
        self.pn_count = self.pn_count + len(particles)
        tmp = numpy.vstack(particles)
        xi = tmp[:,0:1,numpy.newaxis]
        return scipy.stats.norm.logpdf((next_xi-xi).ravel(), 0.0, math.sqrt(self.Q_xi))
#        N = len(particles)
#        lpxi = numpy.empty(N, dtype=float)
#        for i in xrange(N):
#            # Predict z_{t+1}
#            lpxi[i] = kalman.lognormpdf(next_xi, particles[i][0:1], numpy.asarray(((self.Q_xi,),)))
#        return lpxi
    
    def next_pdf_xi_max(self, particles, u, t):
        return numpy.asarray((scipy.stats.norm.logpdf(0.0, 0.0, math.sqrt(self.Q_xi)),)*len(particles))
    
    def measure_nonlin(self, particles, y, t):
        N = len(particles)
        lpy = numpy.empty((N,))
        m = numpy.zeros((1,1))
        for i in xrange(N):
            lpy[i] = kalman.lognormpdf(y[0]-particles[i][0], self.R_xi)
        return lpy
    
    def get_lin_pred_dynamics(self, particles, u, t):
        """ Return matrices describing affine relation of next
            nonlinear state conditioned on current linear state
            
            \z_{t+1]} = A_z * z_t + f_z + v_z, v_z ~ N(0,Q_z)
            
            conditioned on the value of xi_{t+1}. 
            (Not the same as the dynamics unconditioned on xi_{t+1})
            when for example there is a noise correlation between the 
            linear and nonlinear state dynamics) 
            """
        N = len(particles)
        Az = numpy.empty((N,2,2))
        for i in xrange(N):
            Az[i] = numpy.asarray(((math.cos(particles[i][0]), math.sin(particles[i][0])),
                                  (-math.sin(particles[i][0]), math.cos(particles[i][0])))
                                  )
#        Az = list()
#        for i in xrange(N):
#
#            Az.append(numpy.asarray(((math.cos(particles[i][0]), math.sin(particles[i][0])),
#                                  (-math.sin(particles[i][0]), math.cos(particles[i][0])))))
            
        return (Az, None, None)
    
    def get_lin_meas_dynamics(self, particles, y, t):
        N = len(particles)
        Cz = numpy.empty((N,1,2))
        for i in xrange(N):
            Cz[i] = numpy.asarray(((math.cos(particles[i][0]), math.sin(particles[i][0])),))
#        Cz = list()
#        for i in xrange(N):
#            Cz.append(numpy.asarray(((math.cos(particles[i][0]), math.sin(particles[i][0])),)))
        return (y[1], Cz, None, None)
    
    def set_states(self, particles, xi_list, z_list, P_list):
        """ Set the estimate of the Rao-Blackwellized states """
        N = len(particles)
        for i in xrange(N):
            particles[i][0] = xi_list[i].ravel()
            particles[i][1:3] = z_list[i].ravel()
            particles[i][3:] = P_list[i].ravel()
 
    def get_states(self, particles):
        """ Return the estimate of the Rao-Blackwellized states.
            Must return two variables, the first a list containing all the
            expected values, the second a list of the corresponding covariance
            matrices"""
        N = len(particles)
        # This seems to create a bug!!!
#        xil = numpy.empty((N,1,1))
#        zl = numpy.empty((N,2,1))
#        Pl = numpy.empty((N,2,2))
#        for i in xrange(N):
#            xil[i] = particles[i][0:1].reshape(-1,1)
#            zl[i] = particles[i][1:3].reshape(-1,1)
#            Pl[i] = particles[i][3:].reshape(2,2)
        xil = list()
        zl = list()
        Pl = list()
        N = len(particles)
        for i in xrange(N):
            xil.append(particles[i][0:1].reshape(-1,1))
            zl.append(particles[i][1:3].reshape(-1,1))
            Pl.append(particles[i][3:].reshape(2,2))
        
        return (xil, zl, Pl)
    
if __name__ == '__main__':
    steps = 100
    num = 50
    nums=10
    P0_xi = 1.0
    P0_z = numpy.eye(2)
    Q_xi = 0.01*1.0
    Q_z = 0.01*numpy.eye(2)
    R_xi = 0.1*numpy.eye(1)
    R_z = 0.1*numpy.eye(1)
    (x, y) = generate_dataset(steps, P0_xi, P0_z, Q_xi, Q_z, R_xi, R_z)
    
    model = Model(P0_xi, P0_z, Q_xi, Q_z, R_xi, R_z)
    traj = pf.ParticleTrajectory(model, num)
    for k in range(len(y)):
        traj.forward(u=None, y=y[k])
    straj = traj.perform_smoothing(M=nums, method='rsas')
    print("pn_count = %d", traj.pf.model.pn_count)
    if (True):
        plt.plot(range(steps+1), x[0,:],'r-')
        plt.plot(range(steps+1), x[1,:],'g-')
        plt.plot(range(steps+1), x[2,:],'b-')
        #plt.plot(range(1,steps+1), y, 'bx')
        
        xi_vals = numpy.empty((num, steps+1))
        mvals = numpy.empty((steps+1, 3))
        for k in range(steps+1):
            part_exp = numpy.vstack(traj.traj[k].pa.part)
            xi_vals[:,k] = part_exp[:,0]
            mvals[k,:] = wmean(traj.traj[k].pa.w, 
                               part_exp[:,:3])
            plt.plot((k,)*num, xi_vals[:,k], 'r.', markersize=1.0)
            
        for j in xrange(nums):
            tmp = straj.traj[:,j]
            xi = tmp[:,0]
            z = tmp[:,1:]
            plt.plot(range(steps+1), xi[:,0],'r--')
            plt.plot(range(steps+1), z[:,0],'g--')
            plt.plot(range(steps+1), z[:,1],'b--')
            
    #    plt.plot(range(steps+1), mvals[:,0], 'r.')
    #    plt.plot(range(steps+1), mvals[:,1], 'g.')
    #    plt.plot(range(steps+1), mvals[:,2], 'b.')
        plt.show()
    