import numpy
import math
import pyparticleest.kalman as kalman
import pyparticleest.part_utils
import pyparticleest.pf as pf
import matplotlib.pyplot as plt
import scipy.linalg

def generate_dataset(steps, P0, Q, R):
    x = numpy.zeros((steps+1,))
    y = numpy.zeros((steps,))
    x[0] = numpy.random.normal(0.0, P0)
    for k in range(1,steps+1):
        x[k] = math.sin(x[k-1]) + numpy.random.normal(0.0, Q)
        y[k-1] = x[k] + numpy.random.normal(0.0, R)
        
    return (x,y)

def wmean(logw, val):
    w = numpy.exp(logw)
    w = w / sum(w)
    return numpy.sum(w*val.ravel())

def calc_stuff(out, y, particles, N, R):
    Rcho = scipy.linalg.cho_factor(R)
    for k in xrange(N):
        out[k] = kalman.lognormpdf_cho(particles[k].reshape(-1,1)-y, Rcho)
    return out

class Model(pyparticleest.part_utils.ParticleFilteringInterface):
    """ x_{k+1} = x_k + v_k, v_k ~ N(0,Q)
        y_k = x_k + e_k, e_k ~ N(0,R),
        x(0) ~ N(0,P0) """
    
    def __init__(self, P0, Q, R):
        self.P0 = numpy.copy(P0)
        self.Q= numpy.copy(Q)
        self.R=numpy.copy(R)
    
    def create_initial_estimate(self, N):
        return numpy.random.normal(0.0, self.P0, (N,)) 
        
    def sample_process_noise(self, particles, u, t):
        """ Return process noise for input u """
        N = len(particles)
        return numpy.random.normal(0.0, self.Q, (N,)) 
    
    def update(self, particles, u, t, noise):
        """ Update estimate using 'data' as input """
        particles[:] = numpy.sin(particles) + noise
   
    def measure(self, particles, y, t):
        """ Return the log-pdf value of the measurement """
        logyprob = numpy.empty(len(particles), dtype=float)
        N = len(particles)
        return calc_stuff(logyprob, y, particles, N, self.R)

if __name__ == '__main__':
    steps = 100
    num = 100
    P0 = 1.0
    Q = 1.0
    R = 0.01*numpy.asarray(((1.0,),))
    (x, y) = generate_dataset(steps, P0, Q, R)
    
    model = Model(P0, Q, R)
    traj = pf.ParticleTrajectory(model, num)
    for k in range(len(y)):
        traj.forward(u=None, y=y[k])
    plt.plot(range(steps+1), x,'r-')
    plt.plot(range(1,steps+1), y, 'bx')
    
    vals = numpy.empty((num, steps+1))
    mvals = numpy.empty((steps+1))
    for k in range(steps+1):
        vals[:,k] = numpy.copy(traj.traj[k].pa.part)
        mvals[k] = wmean(traj.traj[k].pa.w, 
                          traj.traj[k].pa.part)
#        plt.plot((k,)*num, vals[:,k], 'k.', markersize=1.0)
        
    plt.plot(range(steps+1), mvals, 'k-')
    plt.show()
    