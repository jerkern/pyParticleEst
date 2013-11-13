""" Parameter estimation methods"""
import abc
import PF
import PS
import numpy
import scipy.optimize

from multiprocessing import Pool
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
    
def set_traj_params(straj, params, ind=None):
    for i in range(len(straj)):
        if (ind == None):
            ind = range(len(straj[i].traj))
            for j in ind:
                straj[i].traj[j].set_params(params)
                
estimator = None

def calc_single_traj(i):
    (log_p0, d_log_p0) = estimator.eval_logp_x0(traj_ind=(i,))
    (log_py, d_log_py) = estimator.eval_logp_y(traj_ind=(i,))
    (log_pxn, d_log_pxn) = estimator.eval_logp_xnext(traj_ind=(i,))
    return (log_p0 + log_py + log_pxn, (d_log_p0 + d_log_py + d_log_pxn).tolist())
    
def calc_py(ind):
    return estimator.eval_logp_y(ind)

def calc_pxnext(ind):
    return estimator.eval_logp_xnext(ind)

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
    
    def simulate(self, num_part, num_traj, update_before_predict=True):
        
        particles = self.create_initial_estimate(params=self.params, num=num_part)
        
        # Create a particle approximation object from our particles
        pa = PF.ParticleApproximation(particles=particles)
    
        # Initialise a particle filter with our particle approximation of the initial state,
        # set the resampling threshold to 0.67 (effective particles / total particles )
        self.pt = PF.ParticleTrajectory(pa,0.67)
        
        if (update_before_predict):
           
            # Run particle filter
            for i in range(len(self.y)):
                # Run PF using noise corrupted input signal
                self.pt.update(self.u[i])
            
                # Use noise corrupted measurements
                self.pt.measure(self.y[i])
        else:
                        # Run particle filter
            for i in range(len(self.y)):
                # Use noise corrupted measurements
                self.pt.measure(self.y[i])
                # Run PF using noise corrupted input signal
                self.pt.update(self.u[i])
            
                
            
        # Use the filtered estimates above to created smoothed estimates
        self.straj = PS.do_smoothing(self.pt, num_traj)   # Do sampled smoothing
        for i in range(len(self.straj)):
            (z0, P0) = self.straj[i].traj[0].get_z0_initial()
            self.straj[i].constrained_smoothing(z0, P0)
            
    def maximize(self, param0, num_part, num_traj, max_iter=1000, tol=0.001, 
                 update_before_predict=True, callback=None, callback_sim=None):
        
        def fval_traj_pooled(params_val):
            
            print "params_val = %s" % params_val
            # Needed for the pool'ed functions
            global estimator
            estimator = self
            self.set_params(params_val)
            pool = Pool()
            res = pool.map(calc_single_traj, range(len(self.straj)))
            pool.close()
            pool.join()
            
            log_pval = numpy.zeros((1,))
            d_log_pval = numpy.zeros((len(params_val),))
            
            for i in res:
                log_pval += i[0]
                d_log_pval += numpy.asarray(i[1])
            
            print "pval = %s, d_pval = %s" % (log_pval, d_log_pval)
            return (-1.0*log_pval, -1.0*d_log_pval.ravel())
        
        def fval_pooled(params_val):
            # Stolen from,
            # http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
            def chunks(l, n):
                return [l[i:i+n] for i in range(0, len(l), n)]

            set_traj_params(self.straj, params_val)
            
            # Needed for the pool'ed functions
            global estimator
            estimator = self
            pool = Pool()
            N = len(self.straj[0].traj)
            ind_chunks= chunks(range(N), 20)
            res_py = pool.map(calc_py, ind_chunks)
            
            ind_chunks= chunks(range(N-1), 20)
            res_pxnext = pool.map(calc_pxnext, ind_chunks)
            pool.close()
            pool.join()
            (log_pval, d_log_pval) = self.eval_logp_x0()
            
            for i in res_py:
                log_pval += i[0]
                d_log_pval += i[1]
                
            for i in res_pxnext:
                log_pval += i[0]
                d_log_pval += i[1]
            
            return (-1.0*log_pval, -1.0*d_log_pval.ravel())
            
        def fval(params_val):
            set_traj_params(self.straj, params_val)
            (log_py, d_log_py) = self.eval_logp_y()
            (log_pxnext, d_log_pxnext) = self.eval_logp_xnext()
            (log_px0, d_log_px0) = self.eval_logp_x0()
            val =  (-1.0*(log_py + log_px0 + log_pxnext), -1.0*(d_log_py + d_log_px0 + d_log_pxnext).ravel())
            #print "fval %s : %f" % (params_val, val[0])
            return val

        params_local = numpy.copy(param0)
        Q = -numpy.Inf
        for _i in xrange(max_iter):
            Q_old = Q
            self.set_params(params_local)
            self.simulate(num_part, num_traj, update_before_predict)
            if (callback_sim != None):
                callback_sim(self)
            #res = scipy.optimize.minimize(fun=fval, x0=params, method='nelder-mead', jac=fgrad)
            
            res = scipy.optimize.minimize(fun=fval, x0=params_local, method='l-bfgs-b', jac=True, options=dict({'maxiter':10}))
            
            params_local = res.x

            (Q, Q_grad) = fval(params_local)
            Q = -Q
            Q_grad = -Q_grad
            if (callback != None):
                callback(params=params_local, Q=Q)
            if (numpy.abs(Q - Q_old) < tol):
                break
        return params_local
    
    def eval_prob(self):
        (log_py, d_log_py) = self.eval_logp_y()
        (log_px0, d_log_px0) = self.eval_logp_x0()
        (log_pxnext, d_log_pxnext) = self.eval_logp_xnext()
        return (log_px0 + log_pxnext + log_py, d_log_py + d_log_px0 + d_log_pxnext) 
       
    def eval_logp_x0(self, traj_ind=None):
        logp_x0 = 0.0
        grad_logpx0 = numpy.zeros((len(self.params),))
        M = len(self.straj)
        if (traj_ind == None):
            traj_ind = range(len(self.straj))
        for k in traj_ind:
            (z0, P0) = self.straj[k].traj[0].get_z0_initial()
            (grad_z0, grad_P0) = self.straj[k].traj[0].get_grad_z0_initial()
            (val, grad) = self.straj[k].traj[0].eval_logp_x0(z0, P0, grad_z0, grad_P0)
            logp_x0 += val
            grad_logpx0 += grad.ravel()
        return (logp_x0/M, grad_logpx0/M)
    
    def eval_logp_y(self, ind=None, traj_ind=None):
        logp_y = 0.0
        grad_logpy = numpy.zeros((len(self.params),))
        M = len(self.straj)
        if (traj_ind == None):
            traj_ind = range(len(self.straj))
        for k in traj_ind:
            if (ind == None):
                ind = range(len(self.straj[k].traj))
            for i in ind:
                if (self.straj[k].y[i] != None):
                    (val, grad) = self.straj[k].traj[i].eval_logp_y(self.straj[k].y[i])
                    logp_y += val
                    grad_logpy += grad.ravel()
        
        return (logp_y/M, grad_logpy/M)
    
    def eval_logp_xnext(self, ind=None, traj_ind=None):
        logp_xnext = 0.0
        grad_logpxn = numpy.zeros((len(self.params),))
        M = len(self.straj)
        if (traj_ind == None):
            traj_ind = range(len(self.straj))
        for k in traj_ind:
            if (ind == None):
                ind = range(len(self.straj[k].traj)-1)
            for i in ind:
                (val, grad) = self.straj[k].traj[i].eval_logp_xnext(self.straj[k].traj[i+1])
                logp_xnext += val
                grad_logpxn += grad.ravel()
        return (logp_xnext/M, grad_logpxn/M)
