""" Parameter estimation methods"""
import abc
import pyparticleest.pf as pf
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
    def eval_logp_x0(self, particles):
        """ Calculate gradient of a term of the I1 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""
        pass
    
    @abc.abstractmethod
    def eval_logp_xnext(self, particles, particles_next, u, t, Mz=None):
        """ Calculate gradient of a term of the I2 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""
        pass
    
    @abc.abstractmethod    
    def eval_logp_y(self, particles, y):
        """ Calculate gradient of a term of the I3 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""
        pass
    

    
class ParamEstInterface_GradientSearch(ParamEstInterface):
    """ Interface s for particles to be used with the parameter estimation
        algorithm presented in [1]
        [1] - 'System identification of nonlinear state-space models' by 
              Schon, Wills and Ninness """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def eval_logp_x0_grad(self, particles):
        """ Calculate gradient of a term of the I1 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""
        pass

    @abc.abstractmethod
    def eval_logp_xnext_grad(self, particles, particles_next, u):
        """ Calculate gradient of a term of the I2 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""
        pass
    
    @abc.abstractmethod    
    def eval_logp_y_grad(self, particles, y):
        """ Calculate gradient of a term of the I3 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""
        pass
    

class ParamEstimation(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, model, u, y):
        
        if (u != None):
            self.u = u
        else:
            self.u = [None] * len(y)
        self.y = y
        self.pt = None
        self.straj = None
        self.params = None
        self.model = model
    
    def set_params(self, params):
        self.params = numpy.copy(params)
        self.model.set_params(self.params)
    
    def simulate(self, num_part, num_traj, filter='PF', smoother='rs', res=0.67, meas_first=False):
        resamplings=0
    
        # Initialise a particle filter with our particle approximation of the initial state,
        # set the resampling threshold to 0.67 (effective particles / total particles )
        self.pt = pf.ParticleTrajectory(self.model, num_part, res,filter=filter)
        
        # Run particle filter
        if (meas_first):
            self.pt.measure(self.y[0])
        else:
            if (self.pt.forward(self.u[0], self.y[0])):
                resamplings = resamplings + 1
        for i in range(1,len(self.y)):
            # Run PF using noise corrupted input signal
            if (self.pt.forward(self.u[i], self.y[i])):
                resamplings = resamplings + 1
            
        # Use the filtered estimates above to created smoothed estimates
        self.straj = self.pt.perform_smoothing(num_traj, method=smoother)
        if hasattr(self.model, 'get_rb_initial'): 
            self.straj.constrained_smoothing()
        else:
            self.straj.straj = self.straj.traj
        return resamplings
            
    def maximize(self, param0, num_part, num_traj, max_iter=1000, tol=0.001, 
                 callback=None, callback_sim=None, bounds=None, meas_first=False,
                 smoother='normal'):
        
        def fval(params_val):
            self.model.set_params(params_val)
#            (log_py, d_log_py) = self.eval_logp_y()
#            (log_pxnext, d_log_pxnext) = self.eval_logp_xnext()
#            (log_px0, d_log_px0) = self.eval_logp_x0()
#            val =  (-1.0*(log_py + log_px0 + log_pxnext), -1.0*(d_log_py + d_log_px0 + d_log_pxnext).ravel())
            #print "fval %s : %f" % (params_val, val[0])
            log_py = self.eval_logp_y()
            log_pxnext = self.eval_logp_xnext()
            log_px0 = self.eval_logp_x0()
            val =  -1.0*(log_py + log_px0 + log_pxnext)
            return val

        params_local = numpy.copy(param0)
        Q = -numpy.Inf
        for _i in xrange(max_iter):
            Q_old = Q
            self.set_params(params_local)
            if (numpy.isscalar(num_part)):
                nump = num_part
            else:
                if (_i < len(num_part)):
                    nump = num_part[_i]
                else:
                    nump = num_part[-1]
            if (numpy.isscalar(num_traj)):
                numt = num_traj
            else:
                if (_i < len(num_traj)):
                    numt = num_traj[_i]
                else:
                    numt = num_traj[-1]
            
            self.simulate(nump, numt, smoother=smoother, meas_first=meas_first)
            if (callback_sim != None):
                callback_sim(self)
            #res = scipy.optimize.minimize(fun=fval, x0=params, method='nelder-mead', jac=fgrad)
            
            res = scipy.optimize.minimize(fun=fval, x0=params_local, method='l-bfgs-b', jac=False, 
                                          options=dict({'maxiter':10, 'maxfun':100}), bounds=bounds,)
            
            params_local = res.x

            #(Q, Q_grad) = fval(params_local)
            Q = fval(params_local)
            Q = -Q
            #Q_grad = -Q_grad
            if (callback != None):
                callback(params=params_local, Q=Q)
            if (numpy.abs(Q - Q_old) < tol):
                break
        return (params_local, Q)
    
    def eval_prob(self):
        log_py = self.eval_logp_y()
        log_px0= self.eval_logp_x0()
        log_pxnext = self.eval_logp_xnext()
        return log_px0 + log_pxnext + log_py
       
    def eval_logp_x0(self):
        M = self.straj.straj.shape[1]
        logp_x0 = self.model.eval_logp_x0(numpy.vstack(self.straj.straj[0]))
        return numpy.sum(logp_x0)/M
    
    def eval_logp_y(self, ind=None, traj_ind=None):
        logp_y = 0.0
        M = self.straj.straj.shape[1]
        T = len(self.straj)
        for t in xrange(T):
            if (self.straj.y[t] != None):
                val = self.model.eval_logp_y(numpy.vstack(self.straj.straj[t]), self.straj.y[t])
                logp_y += numpy.sum(val)
        
        return logp_y/M
    
    def eval_logp_xnext(self, ind=None, traj_ind=None):
        logp_xnext = 0.0
        M = self.straj.straj.shape[1]
        T = len(self.straj)
        for t in xrange(T-1):
            if (self.straj.Mz != None):
                val = self.model.eval_logp_xnext(numpy.vstack(self.straj.straj[t]),
                                                 numpy.vstack(self.straj.straj[t+1]),
                                                 self.straj.u[t], self.straj.t[t],
                                                 self.straj.Mz[t])
            else:
                val = self.model.eval_logp_xnext(numpy.vstack(self.straj.straj[t]),
                                 numpy.vstack(self.straj.straj[t+1]),
                                 self.straj.u[t], self.straj.t[t])
            logp_xnext += numpy.sum(val)
        return logp_xnext/M
