import numpy
import math
import pyparticleest.filter as pf
import matplotlib.pyplot as plt
from pyparticleest.models.ltv import LTV

def generate_dataset(steps, z0, P0, Q, R):
    x = numpy.zeros((steps+1,2,1))
    y = numpy.zeros((steps+1,2,1))
    A = numpy.asarray(((1.0, 1.0), (0.0, 1.0)))
    C = numpy.asarray(((1.0, 0.0), (0.0, 0.0)))
    x[0] = numpy.random.multivariate_normal(z0, P0).reshape((-1,1))
    y[0] = C.dot(x[0]) + numpy.random.multivariate_normal((0.0, 0.0), R).reshape((-1,1))
    
    for k in range(0,steps):
        C = numpy.asarray(((math.cos(k+1), 0.0), 
                           (math.sin(k+1), 0.0))).reshape((2,-1))
                          
        x[k+1] = A.dot(x[k]) + numpy.random.multivariate_normal((0.0, 0.0), Q).reshape((-1,1))
        y[k+1] = C.dot(x[k+1]) + numpy.random.multivariate_normal((0.0, 0.0), R).reshape((-1,1))
        
    return (x,y)
   
#def calc_stuff(out, y, particles, N, R):
#    for k in xrange(N):
#        out[k] = kalman.lognormpdf(particles[k].reshape(-1,1), y, R)
#    return out

class Model(LTV):
    
    def __init__(self, z0, P0, Q, R):
        A = numpy.asarray(((1.0, 1.0), (0.0, 1.0)))
        C = numpy.asarray(((0.0, 0.0), (0.0, 0.0)))
        super(Model, self).__init__(A=A, C=C, 
                                    z0=z0, P0=P0,
                                    Q=Q, R=R)
    
    
    def get_meas_dynamics(self, y, t):
        C = numpy.asarray(((math.cos(t), 0.0), 
                           (math.sin(t), 0.0))).reshape((2,2))
        return (y, C, None, None)
    

def callback(params, Q):
    print "params = %s" % numpy.exp(params)
    

def callback_sim(estimator):
    #vals = numpy.empty((num, steps+1))

    plt.figure(1)
    plt.clf()
#    mvals = numpy.empty((steps+1))
#    for k in range(steps+1):
#        #vals[:,k] = numpy.copy(estimator.pt.traj[k].pa.part)
#        mvals[k] = wmean(estimator.pt.traj[k].pa.w, 
#                          estimator.pt.traj[k].pa.part)
#        #plt.plot((k,)*num, vals[:,k], 'k.', markersize=0.5)
#    plt.plot(range(steps+1), mvals, 'k-')
        
    for k in range(estimator.straj.traj.shape[1]):
        plt.plot(range(steps+1), estimator.straj.traj[:,k], 'g-')
    
    plt.plot(range(steps+1), x,'r-')
    #plt.plot(range(steps+1), y, 'bx')    
    plt.draw()
    plt.show()

if __name__ == '__main__':
    steps = 50
    num = 1
    M = 1
    z0 = numpy.asarray((1.0, 2.0))
    P0 = 10.0*numpy.eye(2)
    Q = 1.0*numpy.eye(2)
    R = 0.1*numpy.eye(2)
    (x, y) = generate_dataset(steps, z0, P0, Q, R)
    model = Model(z0, P0, Q, R)
    traj = pf.ParticleTrajectory(model, num)
    traj.measure(y=y[0])
    for k in range(1,len(y)):
        traj.forward(u=None, y=y[k])

    straj = traj.perform_smoothing(M, method='full')
    plt.plot(range(steps+1),x[:,0,0],'r-')
    plt.plot(range(steps+1),x[:,1,0],'g-')
    plt.plot(range(steps+1),y[:,0,0],'bx')
    plt.plot(range(steps+1),y[:,1,0],'bo')
    plt.plot(range(steps+1),straj.traj[:,0,0],'r--')
    plt.plot(range(steps+1),straj.traj[:,0,1],'g--')
    plt.show()
    
    
    
    