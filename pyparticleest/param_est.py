""" Parameter estimation methods"""
import abc
import pyparticleest.pf as pf
import numpy
import scipy.optimize
import matplotlib.pyplot as plt

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
    def eval_logp_x0(self, particles, t):
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
    def eval_logp_y(self, particles, y, t):
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
    def eval_logp_x0_val_grad(self, particles):
        """ Calculate gradient of a term of the I1 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""
        pass

    @abc.abstractmethod
    def eval_logp_xnext_val_grad(self, particles, particles_next, u, t):
        """ Calculate gradient of a term of the I2 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with 
            respect to the corresponding parameter"""
        pass
    
    @abc.abstractmethod    
    def eval_logp_y_val_grad(self, particles, y, t):
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
    
    def simulate(self, num_part, num_traj, filter='PF', smoother='full', res=0.67, meas_first=False):
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
            if (self.pt.forward(self.u[i-1], self.y[i])):
                resamplings = resamplings + 1
            
        # Use the filtered estimates above to created smoothed estimates
        self.straj = self.pt.perform_smoothing(num_traj, method=smoother)
        return resamplings
            
    def maximize(self, param0, num_part, num_traj, max_iter=1000, tol=0.001, 
                 callback=None, callback_sim=None, bounds=None, meas_first=False,
                 smoother='full'):
        
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
        M = self.straj.traj.shape[1]
        logp_x0 = self.model.eval_logp_x0(self.straj.traj[0],
                                          self.straj.t[0])
        return numpy.sum(logp_x0)/M
    
    def eval_logp_y(self, ind=None, traj_ind=None):
        logp_y = 0.0
        M = self.straj.traj.shape[1]
        T = len(self.straj)
        for t in xrange(T):
            if (self.straj.y[t] != None):
                val = self.model.eval_logp_y(self.straj.traj[t],
                                             self.straj.y[t],
                                             self.straj.t[t])
                logp_y += numpy.sum(val)
        
        return logp_y/M
    
    def eval_logp_xnext(self, ind=None, traj_ind=None):
        logp_xnext = 0.0
        M = self.straj.traj.shape[1]
        T = len(self.straj)
        for t in xrange(T-1):
            val = self.model.eval_logp_xnext(self.straj.traj[t],
                                             self.straj.traj[t+1],
                                             self.straj.u[t],
                                             self.straj.t[t])
            logp_xnext += numpy.sum(val)
        return logp_xnext/M
    
    def eval_logp_x0_val_grad(self):
        M = self.straj.traj.shape[1]
        (logp_x0, logp_x0_grad) = self.model.eval_logp_x0_val_grad(self.straj.traj[0],
                                                                   self.straj.t[0])
        return (logp_x0/M, logp_x0_grad/M)
    
    def eval_logp_y_val_grad(self, ind=None, traj_ind=None):
        logp_y_grad = numpy.zeros((len(self.model.params)))
        logp_y = 0.0
        M = self.straj.traj.shape[1]
        T = len(self.straj)
        for t in xrange(T):
            if (self.straj.y[t] != None):
                (val, grad) = self.model.eval_logp_y_val_grad(self.straj.traj[t],
                                                              self.straj.y[t],
                                                              self.straj.t[t])
                logp_y += val
                logp_y_grad += grad
        return (logp_y/M, logp_y_grad/M)
    
    def eval_logp_xnext_val_grad(self, ind=None, traj_ind=None):
        logp_xnext_grad = numpy.zeros((len(self.model.params)))
        logp_xnext = 0.0
        M = self.straj.traj.shape[1]
        T = len(self.straj)
        for t in xrange(T-1):
            (val, grad) = self.model.eval_logp_xnext_val_grad(self.straj.traj[t],
                                                              self.straj.traj[t+1],
                                                              self.straj.u[t],
                                                              self.straj.t[t])
            logp_xnext += val
            logp_xnext_grad += grad
             
        return (logp_xnext/M, logp_xnext_grad/M)
    
class GradPlot():
    def __init__(self, params, vals, diff):
        self.params = params
        self.vals = vals
        self.diff = diff

    def plot(self, fig_id):
        fig = plt.figure(fig_id)
        fig.clf()
        plt.plot(self.params, self.vals)
        if (self.diff != None):
            for k in range(len(self.params)):
                if (k % 10 == 1):
                    self.draw_gradient(self.params[k], self.vals[k], self.params[k]-self.params[k-1], self.diff[k])
                
        plt.show()
        
    def draw_gradient(self, x, y, dx, dydx):
        plt.plot((x-dx, x+dx), (y-dydx*dx, y+dydx*dx), 'r')
    
    
class GradientTest(ParamEstimation):
    
    def test(self, param_id, param_vals, num=100, nums=1):
        self.simulate(num_part=num, num_traj=nums)
        param_steps = len(param_vals)
        logpy = numpy.zeros((param_steps,))
        grad_logpy = numpy.zeros((param_steps, len(self.params)))
        logpxn = numpy.zeros((param_steps,))
        grad_logpxn = numpy.zeros((param_steps, len(self.params)))
        logpx0 = numpy.zeros((param_steps,))
        grad_logpx0 = numpy.zeros((param_steps, len(self.params)))
        for k in range(param_steps):    
            tmp = numpy.copy(self.params)
            tmp[param_id] = param_vals[k]
            self.set_params(tmp)
            logpy[k] = self.eval_logp_y()
            logpxn[k]  = self.eval_logp_xnext()
            logpx0[k] = self.eval_logp_x0()
            grad_logpy[k] = self.eval_logp_y_val_grad()[1]
            grad_logpxn[k]  = self.eval_logp_xnext_val_grad()[1]
            grad_logpx0[k] = self.eval_logp_x0_val_grad()[1]

        self.plot_y = GradPlot(param_vals, logpy, grad_logpy[:,param_id])
        self.plot_xn = GradPlot(param_vals, logpxn, grad_logpxn[:,param_id])
        self.plot_x0 = GradPlot(param_vals, logpx0, grad_logpx0[:,param_id])

