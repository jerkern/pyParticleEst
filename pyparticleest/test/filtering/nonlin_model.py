import numpy
import math
import pyparticleest.models.nlg
import pyparticleest.filter as pf
import matplotlib.pyplot as plt

def generate_dataset(steps, P0, Q, R):
    x = numpy.zeros((steps+1,))
    y = numpy.zeros((steps,))
    x[0] = numpy.random.normal(0.0, P0)
    for k in range(1,steps+1):
        x[k] = math.sin(x[k-1]) + numpy.random.normal(0.0, math.sqrt(Q))
        y[k-1] = x[k] + numpy.random.normal(0.0, math.sqrt(R))
        
    return (x,y)

def wmean(logw, val):
    w = numpy.exp(logw)
    w = w / sum(w)
    return numpy.sum(w*val.ravel())


class Model(pyparticleest.models.nlg.NonlinearGaussianInitialGaussian):
    """ x_{k+1} = sin(x_k) + v_k, v_k ~ N(0,Q)
        y_k = x_k + e_k, e_k ~ N(0,R),
        x(0) ~ N(0,P0) """
    
    def __init__(self, P0, Q, R):
        x0 = numpy.zeros((1,1))
        super(Model, self).__init__(x0=x0, 
                                    Px0=numpy.asarray(P0).reshape((1,1)),
                                    Q=numpy.asarray(Q).reshape((1,1)),
                                    R=numpy.asarray(R).reshape((1,1)))
    
    def get_f(self, particles, u, t):
        return numpy.sin(particles)   
    
    def get_g(self, particles, t):
        return particles
    

if __name__ == '__main__':
    steps = 100
    num = 100
    P0 = 1.0
    Q = 1.0
    R = 0.0001*numpy.asarray(((1.0,),))
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
        vals[:,k] = traj.traj[k].pa.part.ravel()
        mvals[k] = wmean(traj.traj[k].pa.w, 
                          traj.traj[k].pa.part)
        plt.plot((k,)*num, vals[:,k], 'k.', markersize=1.0)
        
    plt.plot(range(steps+1), mvals, 'k-')
    plt.show()
    