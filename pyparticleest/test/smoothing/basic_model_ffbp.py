import numpy
import pyparticleest.models.nlg
import pyparticleest.filter as pf
import matplotlib.pyplot as plt

def generate_dataset(steps, P0, Q, R):
    x = numpy.zeros((steps+1,))
    y = numpy.zeros((steps,))
    x[0] = 2.0+0.0*numpy.random.normal(0.0, P0)
    for k in range(1,steps+1):
        x[k] = x[k-1] +  numpy.random.normal(0.0, Q)
        y[k-1] = x[k] + numpy.random.normal(0.0, R)
        
    return (x,y)

def wmean(logw, val):
    w = numpy.exp(logw)
    w = w / sum(w)
    return numpy.sum(w*val.ravel())

class Model(pyparticleest.models.nlg.NonlinearGaussianInitialGaussian):
    """ x_{k+1} = x_k + v_k, v_k ~ N(0,Q)
        y_k = x_k + e_k, e_k ~ N(0,R),
        x(0) ~ N(0,P0) """
    
    def __init__(self, P0, Q, R):
        x0 = numpy.zeros((1,1))
        super(Model, self).__init__(x0=x0, 
                                    Px0=numpy.asarray(P0).reshape((1,1)),
                                    Q=numpy.asarray(Q).reshape((1,1)),
                                    R=numpy.asarray(R).reshape((1,1)))
    
   
    def get_f(self, particles, u, t):
        return particles   
    
    def get_g(self, particles, t):
        return particles

if __name__ == '__main__':
    steps = 80
    num = 40
    M = 20
    P0 = 1.0
    Q = 1.0
    R = numpy.asarray(((1.0,),))
    numpy.random.seed(0)
    (x, y) = generate_dataset(steps, P0, Q, R)
    
    model = Model(P0, Q, R)
    traj = pf.ParticleTrajectory(model, num)
    for k in range(len(y)):
        traj.forward(u=None, y=y[k])
    plt.plot(range(steps+1), x,'r-')
    plt.plot(range(1,steps+1), y, 'bx')
    straj = traj.perform_smoothing(M, method='bp', smoother_options={'R': 50})
    plt.plot(range(steps+1), straj.straj[:,:,0], 'g.')
    plt.show()
