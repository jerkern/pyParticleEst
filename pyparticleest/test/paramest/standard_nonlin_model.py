import numpy
import math
import pyximport
pyximport.install(inplace=True)
import pyparticleest.utils.kalman as kalman
import pyparticleest.interfaces as interfaces
import pyparticleest.paramest.paramest as param_est
import pyparticleest.paramest.interfaces as pestint
import matplotlib.pyplot as plt


def generate_dataset(steps, P0, Q, R):
    x = numpy.zeros((steps+1,))
    y = numpy.zeros((steps+1,))
    x[0] = numpy.random.multivariate_normal((0.0,), P0)
    y[0] = 0.05*x[0]**2 + numpy.random.multivariate_normal((0.0,), R)
    for k in range(0,steps):
        x[k+1] = 0.5*x[k] + 25.0*x[k]/(1+x[k]**2) + 8*math.cos(1.2*k) + numpy.random.multivariate_normal((0.0,), Q)
        y[k+1] = 0.05*x[k+1]**2 + numpy.random.multivariate_normal((0.0,), R)
        
    return (x,y)

def wmean(logw, val):
    w = numpy.exp(logw)
    w = w / sum(w)
    return numpy.sum(w*val.ravel())
    
class Model(interfaces.FFBSiRS, pestint.ParamEstInterface):
    """ x_{k+1} = x_k + v_k, v_k ~ N(0,Q)
        y_k = x_k + e_k, e_k ~ N(0,R),
        x(0) ~ N(0,P0) """
    
    def __init__(self, P0, Q, R):
        self.P0 = numpy.copy(P0)
        self.Q= numpy.copy(Q)
        self.R=numpy.copy(R)
        self.logxn_max = kalman.lognormpdf_scalar(numpy.zeros((1,)), self.Q)
    
    def create_initial_estimate(self, N):
        return numpy.random.normal(0.0, numpy.sqrt(self.P0), (N,)) 
        
    def sample_process_noise(self, particles, u, t):
        """ Return process noise for input u """
        N = len(particles)
        return numpy.random.normal(0.0, numpy.sqrt(self.Q), (N,)) 
    
    def update(self, particles, u, noise, t):
        """ Update estimate using 'data' as input """
        particles[:] = 0.5*particles + 25.0*particles/(1+particles**2) + 8*math.cos(1.2*t)  + noise
   
    def measure(self, particles, y, t):
        """ Return the log-pdf value of the measurement """
        return kalman.lognormpdf_scalar(0.05*particles**2-y, self.R)
    
    def logp_xnext(self, particles, next_part, u, t):
        """ Return the log-pdf value for the possible future state 'next' given input u """
        pn = 0.5*particles + 25.0*particles/(1+particles**2) + 8*math.cos(1.2*t)
        return kalman.lognormpdf_scalar(pn-next_part.ravel(), self.Q)
    
    def logp_xnext_max(self, particles, u, t):
        return self.logxn_max
    
    def sample_smooth(self, particles, next_part, u, y, t):
        """ Update ev. Rao-Blackwellized states conditioned on "next_part" """
        return particles
    
    def set_params(self, params):
        """ New set of parameters for which the integral approximation terms will be evaluated"""
        self.Q=math.exp(params[0])*numpy.eye(1)
        self.R=math.exp(params[1])*numpy.eye(1)

    def eval_logp_x0(self, particles, t):
        """ Calculate gradient of a term of the I1 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""
        return kalman.lognormpdf_scalar(particles, self.P0)
    
    def copy_ind(self, particles, new_ind=None):
        if (new_ind != None):
            return numpy.copy(particles[new_ind])
        else:
            return numpy.copy(particles)

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
    numpy.random.seed(1)
    steps = 1499
    iterations = numpy.asarray(range(200))
    num = numpy.ceil(500 + 4500.0/(iterations[-1]**3)*iterations**3).astype(int)
    M = numpy.ceil(50 + 450.0/(iterations[-1]**3)*iterations**3).astype(int)
    P0 = 5.0*numpy.eye(1)
    Q = 1.0*numpy.eye(1)
    R = 0.1*numpy.eye(1)
    (x, y) = generate_dataset(steps, P0, Q, R)
    theta0 = numpy.log(numpy.asarray((2.0, 2.0)))
    model = Model(P0, Q, R)
    estimator = param_est.ParamEstimation(model, u=None, y=y)
    callback(theta0, None)
    estimator.maximize(theta0, num, M, smoother='rsas',meas_first=True, max_iter=len(iterations),
                       callback=callback)
#     plt.ion()
#     estimator.maximize(theta0, num, M, smoother='full',meas_first=True, max_iter=len(iterations),
#                        callback_sim=callback_sim, callback=callback)
#     plt.ioff()
#    traj = pf.ParticleTrajectory(model, num)
#    traj.measure(y[0])
#    for k in range(1,len(y)):
#        traj.forward(u=None, y=y[k])
#
#    straj = traj.perform_smoothing(M, method='rs')
    
    
    