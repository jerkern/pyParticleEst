import numpy
import math
from pyparticleest.models.mixed_nl_gaussian import MixedNLGaussian
import pyparticleest.pf as pf
import matplotlib.pyplot as plt

def generate_dataset(steps, P0_xi, P0_z, Qxi, Qz, Qxiz, R):
    Q = numpy.vstack((numpy.hstack((Qxi, Qxiz)),
                      numpy.hstack((Qxiz.T, Qz))))
    xi = numpy.zeros((1, steps+1))
    z = numpy.zeros((1, steps+1))
    y = numpy.zeros((steps,1))
    xi[:,0] = numpy.random.normal(0.0, numpy.sqrt(P0_xi))
    z[:,0] = numpy.random.normal(0.0, numpy.sqrt(P0_z))
    for k in range(1,steps+1):
        noise = numpy.random.multivariate_normal(numpy.zeros((2,)), Q)
        xi[:,k] = xi[:,k-1] + z[:,k-1] + noise[0]
        z[:,k] = z[:,k-1] + noise[1]
        y[k-1,0] = xi[:,k] + numpy.random.normal(0.0, numpy.sqrt(R))
        
    x = numpy.vstack((xi, z))
    return (x,y)

def wmean(logw, val):
    w = numpy.exp(logw)
    w = w / sum(w)
    w.reshape((1,-1))
    return w.dot(val)

class Model(MixedNLGaussian):
    """ xi_{k+1} = xi_k + z_k + v_xi_k, v_xi ~ N(0,Q_xi)
        z_{k+1} = z_{k} + v_z, v_z_k ~ N(0, Q_z) 
        y_k = xi_k + +z_k + e_k, e_k ~ N(0,R_z),
        (v_xi v_z).T ~ N(0, ((Q_xi, Qxiz), (Qxiz.T Qz)) """
    
    def __init__(self, P0_xi, P0_z, Q_xi, Q_z, Q_xiz, R):
        fz = numpy.zeros((1,1))
        h = numpy.zeros((1,1))
        Axi = numpy.eye(1)
        Az = numpy.eye(1)
        self.pn_count = 0
        C = numpy.zeros((1,1))
        self.P0_xi = numpy.copy(P0_xi)
        self.P0_z = numpy.copy(P0_z)
        self.z0 = numpy.zeros((1,))
        super(Model, self).__init__(fxi=None, fz=fz, 
                                    Axi=Axi,Az=Az,
                                    Qxi=Q_xi, Qxiz=Q_xiz, Qz=Q_z,
                                    h=h, C=C, R=R)
        
    def create_initial_estimate(self, N):
        particles = numpy.empty((N,), dtype=numpy.ndarray)
               
        for i in xrange(N):
            particles[i] = numpy.empty(3)
            particles[i][0] = numpy.random.normal(0.0, math.sqrt(self.P0_xi))
            particles[i][1] = numpy.copy(self.z0)
            particles[i][2:] = numpy.copy(self.P0_z).ravel()  
        return particles
    
    def get_rb_initial(self, xi0):
        return (numpy.copy(self.z0),
                numpy.copy(self.P0_z))
        
    def get_nonlin_pred_dynamics(self, particles, u):
            N = len(particles)
            tmp = numpy.vstack(particles)
            fxi = tmp[:,0].tolist()
            return (None, fxi, None)
        
    def get_meas_dynamics(self, y, particles):
            N = len(particles)
            tmp = numpy.vstack(particles)
            h = tmp[:,0].tolist()
            return (y, None, h, None)

    def set_states(self, particles, xi_list, z_list, P_list):
        """ Set the estimate of the Rao-Blackwellized states """
        N = len(particles)
        for i in xrange(N):
            particles[i][0:1] = xi_list[i].ravel()
            particles[i][1:2] = z_list[i].ravel()
            particles[i][2:] = P_list[i].ravel()
 
    def get_states(self, particles):
        """ Return the estimate of the Rao-Blackwellized states.
            Must return two variables, the first a list containing all the
            expected values, the second a list of the corresponding covariance
            matrices"""
        N = len(particles)
        xil = list()
        zl = list()
        Pl = list()
        N = len(particles)
        for i in xrange(N):
            xil.append(particles[i][0].reshape(-1,1))
            zl.append(particles[i][1].reshape(-1,1))
            Pl.append(particles[i][2:].reshape(1,1))
        
        return (xil, zl, Pl)
    
if __name__ == '__main__':
    steps = 30
    num = 100
    nums = 10
    P0_xi = numpy.eye(1)
    P0_z = numpy.eye(1)
    Q_xi = 1.0*numpy.eye(1)
    Q_z = 1.0*numpy.eye(1)
    Q_xiz = 0.0*numpy.eye(1)
    R = 0.1*numpy.eye(1)
    (x, y) = generate_dataset(steps, P0_xi, P0_z, Q_xi, Q_z, Q_xiz, R)
    
    model = Model(P0_xi, P0_z, Q_xi, Q_z, Q_xiz, R)
    traj = pf.ParticleTrajectory(model, num)

    for k in range(len(y)):
        traj.forward(u=None, y=y[k])
    straj = traj.perform_smoothing(M=nums, method='rsas')
    straj.constrained_smoothing()
        
#    print("pn_count = %d", traj.pf.model.pn_count)
    if (True):
        plt.plot(range(steps+1), x[0,:],'r-')
        plt.plot(range(steps+1), x[1,:],'b-')
        plt.plot(range(1,steps+1), y, 'rx')
        
        xi_vals = numpy.empty((num, steps+1))
        z_vals = numpy.empty((num, steps+1))
        mvals = numpy.empty((steps+1, 2))
        for k in range(steps+1):
            part_exp = numpy.vstack(traj.traj[k].pa.part)
            xi_vals[:,k] = part_exp[:,0]
            z_vals[:,k] = part_exp[:,1]
            mvals[k,:] = wmean(traj.traj[k].pa.w, 
                               part_exp[:,:2])
            plt.plot((k,)*num, xi_vals[:,k], 'r.', markersize=1.0)
            plt.plot((k,)*num, z_vals[:,k], 'b.', markersize=1.0)

            
        for j in xrange(nums):
            tmp = numpy.hstack(straj.traj[:,j])
            xi = numpy.vstack(tmp[0,:,0])
            z = numpy.vstack(tmp[1,:,0])
            plt.plot(range(steps+1), xi[:,0],'r--')
            plt.plot(range(steps+1), z[:,0],'b--')
            
        for j in xrange(nums):
            tmp = numpy.vstack(straj.straj[:,j])
            xi = tmp[:,0]
            z = tmp[:,1]
            plt.plot(range(steps+1), z,'k--')
            
            
        plt.plot(range(steps+1), mvals[:,0], 'ro', markersize=3.0)
        plt.plot(range(steps+1), mvals[:,1], 'bo', markersize=3.0)

        plt.show()
    