""" Parameter estimation methods"""
import abc
import PF
import PS
import numpy
import scipy.optimize
 
class ParamEstInterface(object):
    """ Interface s for particles to be used with the parameter estimation
        algorithm presented in [1]
        [1] - 'System identification of nonlinear state-space models' by 
              Schon, Wills and Ninness """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def set_params(self, params):
        """ New set of parameters for which the integral approximation terms will be evaluated"""
        pass

    @abc.abstractmethod
    def eval_logp_x0(self, z0, P0, diff_z0, diff_P0):
        """ Calculate gradient of a term of the I1 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""
        pass

    @abc.abstractmethod
    def eval_logp_xnext(self, x_next):
        """ Calculate gradient of a term of the I2 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""
        pass

    @abc.abstractmethod    
    def eval_logp_y(self, y):
        """ Calculate gradient of a term of the I3 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""
        pass
    
def set_traj_params(straj, params):
    for i in range(len(straj)):
            for j in range(len(straj[i].traj)):
                straj[i].traj[j].set_params(params)

class ParamEstimation(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, u, y):
        
        if (u != None):
            self.u = u
        else:
            self.u = [None] * len(y)
        self.y = y
        self.pt = None
        self.straj = None
        self.params = None
    
    @abc.abstractmethod
    def create_initial_estimate(self, params, num):
        pass
    
    def set_params(self, params):
        self.params = numpy.copy(params)
        if (self.straj != None):
            set_traj_params(self.straj, params)
    
    def simulate(self, num_part, num_traj):
        
        particles = self.create_initial_estimate(params=self.params, num=num_part)
        
        # Create a particle approximation object from our particles
        pa = PF.ParticleApproximation(particles=particles)
    
        # Initialise a particle filter with our particle approximation of the initial state,
        # set the resampling threshold to 0.67 (effective particles / total particles )
        self.pt = PF.ParticleTrajectory(pa,0.67)
        
        # Run particle filter
        for i in range(len(self.y)):
            # Run PF using noise corrupted input signal
            self.pt.update(self.u[i])
        
            # Use noise corrupted measurements
            self.pt.measure(self.y[i])
            
        # Use the filtered estimates above to created smoothed estimates
        self.straj = PS.do_smoothing(self.pt, num_traj)   # Do sampled smoothing
        for i in range(len(self.straj)):
            (z0, P0) = self.straj[i].traj[0].get_z0_initial()
            self.straj[i].constrained_smoothing(z0, P0)
            
    def maximize(self, param0, num_part, num_traj, max_iter=1000, tol=0.001):
        
        def fval(params):
            set_traj_params(self.straj, params)
            (log_py, d_log_py) = self.eval_logp_y()
            (log_px0, d_log_px0) = self.eval_logp_x0()
            (log_pxnext, d_log_pxnext) = self.eval_logp_xnext()
            return (-1.0*(log_py + log_px0 + log_pxnext), -1.0*(d_log_py + d_log_px0 + d_log_pxnext).ravel())
        
        params = numpy.copy(param0)
        Q = -numpy.Inf
        for _i in xrange(max_iter):
            Q_old = Q
            self.set_params(params)
            self.simulate(num_part, num_traj)
            #res = scipy.optimize.minimize(fun=fval, x0=params, method='nelder-mead', jac=fgrad)
            res = scipy.optimize.minimize(fun=fval, x0=params, method='BFGS', jac=True)
            params = res.x
            (Q, Q_grad) = fval(params)
            Q = -Q
            Q_grad = -Q_grad
            print params, Q
            if (numpy.abs(Q - Q_old) < tol):
                break
        return params
    
    def eval_prob(self):
        (log_py, d_log_py) = self.eval_logp_y()
        (log_px0, d_log_px0) = self.eval_logp_x0()
        (log_pxnext, d_log_pxnext) = self.eval_logp_xnext()
        return (log_px0 + log_pxnext + log_py, d_log_py + d_log_px0 + d_log_pxnext) 
       
    def eval_logp_x0(self):
        logp_x0 = 0.0
        grad_logpx0 = numpy.zeros((len(self.params.shape),1))
        M = len(self.straj)
        for traj in self.straj:
            (z0, P0) = traj.traj[0].get_z0_initial()
            (grad_z0, grad_P0) = traj.traj[0].get_grad_z0_initial()
            (val, grad) = traj.traj[0].eval_logp_x0(z0, P0, grad_z0, grad_P0)
            logp_x0 += val
            grad_logpx0 += grad
        return (logp_x0/M, grad_logpx0/M)
    
    def eval_logp_y(self):
        logp_y = 0.0
        grad_logpy = numpy.zeros((len(self.params.shape),1))
        M = len(self.straj)
        for traj in self.straj:
            for i in range(len(traj.traj)):
                if (traj.y[i] != None):
                    (val, grad) = traj.traj[i].eval_logp_y(traj.y[i])
                    logp_y += val
                    grad_logpy += grad
        return (logp_y/M, grad_logpy/M)
    
    def eval_logp_xnext(self):
        logp_xnext = 0.0
        grad_logpxn = numpy.zeros((len(self.params.shape),1))
        M = len(self.straj)
        for traj in self.straj:
            for i in range(len(traj.traj)-1):
                (val, grad) = traj.traj[i].eval_logp_xnext(traj.traj[i+1])
                logp_xnext += val
                grad_logpxn += grad
        return (logp_xnext/M, grad_logpxn/M)
