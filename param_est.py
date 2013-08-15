""" Parameter estimation methods"""
import abc
import PF
import PS
 
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
    def eval_logp_x1(self, z0, P0):
        """ Calculate a term of the I1 integral approximation
        and its gradient as specified in [1]"""
        pass

    @abc.abstractmethod
    def eval_logp_xnext(self, x_next):
        """ Calculate a term of the I2 integral approximation
        and its gradient as specified in [1]"""
        pass

    @abc.abstractmethod    
    def eval_logp_y(self, y):
        """ Calculate a term of the I3 integral approximation
        and its gradient as specified in [1]"""
        pass
    

class ParamEstimation(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, u, y):
        
        self.u = u
        self.y = y
        self.pt = None
        self.straj = None
        self.params = None
    
    @abc.abstractmethod
    def create_initial_estimate(self, params, num):
        pass
    
    def set_params(self, params):
        self.params = params.ravel()
    
    def simulate(self, num_part, num_traj):
        
        (particles, z0, P0) = self.create_initial_estimate(params=self.params, num=num_part)
        
        # Create a particle approximation object from our particles
        pa = PF.ParticleApproximation(particles=particles)
    
        # Initialise a particle filter with our particle approximation of the initial state,
        # set the resampling threshold to 0.67 (effective particles / total particles )
        self.pt = PF.ParticleTrajectory(pa,0.67)
        
        # Run particle filter
        for i in range(self.u.shape[1]):
            # Run PF using noise corrupted input signal
            self.pt.update(self.u[:,i].reshape(-1,1))
        
            # Use noise corrupted measurements
            self.pt.measure(self.y[:,i].reshape(-1,1))
            
        # Use the filtered estimates above to created smoothed estimates
        self.straj = PS.do_smoothing(self.pt, num_traj)   # Do sampled smoothing
        for i in range(len(self.straj)):
            self.straj[i].constrained_smoothing(z0, P0)
       
    def eval_logp_y(self):
        logp_y = 0.0
        for traj in self.straj:
            for i in range(len(traj.traj)):
                if (traj.y[i] != None):
                    logp_y += traj.traj[i].eval_logp_y(traj.y[i])
        return -logp_y
