'''
Interfaces required for using the parameter estimation methods

@author: Jerker Nordh
'''
import abc
import numpy
import scipy.optimize

class ParamEst(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def maximize(self, straj):
        pass

class ParamEstIntFullTraj(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def set_params(self, params):
        """
        New set of parameters for which the integral approximation terms will be evaluated

        Args:
         - params (array-like): new parameter values
        """
        pass

    @abc.abstractmethod
    def eval_logp_x0(self, particles, t):
        """
        Calculate term of the I1 integral approximation as specified in [1].
        Eg. Evaluate log p(x_0) or sum log p(x_0)

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - t (float): time stamp

        Returns: (array-like) or (float) """
        pass

    @abc.abstractmethod
    def eval_logp_xnext_fulltraj(self, straj, ut, tt):
        pass

    @abc.abstractmethod
    def eval_logp_y_fulltraj(self, straj, yt, tt):
        pass

class ParamEstInterface(ParamEstIntFullTraj):
    """ Interface s for particles to be used with the parameter estimation
        algorithm presented in [1]
        [1] - 'System identification of nonlinear state-space models' by Schon, Wills and Ninness """
    __metaclass__ = abc.ABCMeta


    def eval_logp_xnext_fulltraj(self, straj, ut, tt):
        logp_xnext = 0.0
        sest = straj.get_smoothed_estimates()
        M = sest.shape[1]
        T = straj.traj.shape[0]
        for i in range(T - 1):
            val = self.eval_logp_xnext(sest[i],
                                       sest[i + 1],
                                       ut[i], tt[i])
            logp_xnext += numpy.sum(val)
        return logp_xnext / M

    def eval_logp_y_fulltraj(self, straj, yt, tt):
        logp_y = 0.0
        sest = straj.get_smoothed_estimates()
        M = sest.shape[1]
        T = straj.traj.shape[0]
        for i in range(T):
            if (yt[i] is not None):
                val = self.eval_logp_y(sest[i], yt[i], tt[i])
                logp_y += numpy.sum(val)

        return logp_y / M

    def eval_logp_xnext(self, particles, particles_next, u, t):
        """
        Calculate gradient of a term of the I2 integral approximation
        as specified in [1].

        Eg. Evaluate log p(x_{t+1}|x_t) or sum log p(x_{t+1}|x_t)

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - x_next (array-like): future states
         - t (float): time stamp

        Returns: (array-like) or (float)
        """
        # Here we can just reuse the method used in the particle smoothing as default
        return self.logp_xnext(particles, particles_next, u, t)

    def eval_logp_y(self, particles, y, t):
        """
        Calculate gradient of a term of the I3 integral approximation
        as specified in [1].

        Eg. Evaluate log p(y_t|x_t) or sum log p(y_t|x_t)

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - y (array-like): measurement
         - t (float): time stamp

        Returns: (array-like) or (float)
        """

        # Default implementation, doesn't work for classes were the measure updates
        # the internal state of the particle (e.g Rao-Blackwellized models)
        return self.measure(particles, y, t)



class ParamEstInterface_GradientSearchFullTraj(ParamEstInterface):
    @abc.abstractmethod
    def eval_logp_y_val_grad_fulltraj(self, straj, yt, tt):
        pass

    @abc.abstractmethod
    def eval_logp_xnext_val_grad_fulltraj(self, straj, ut, tt):
        pass


class ParamEstInterface_GradientSearch(ParamEstInterface_GradientSearchFullTraj):
    """ Interface s for particles to be used with the parameter estimation
        algorithm presented in [1] using analytic gradients
    """
    __metaclass__ = abc.ABCMeta

    def eval_logp_y_val_grad_fulltraj(self, straj, yt, tt):
        logp_y_grad = numpy.zeros((len(self.params)))
        logp_y = 0.0
        sest = straj.get_smoothed_estimates()
        M = sest.shape[1]
        T = len(straj)
        for t in range(T):
            if (straj.y[t] is not None):
                (val, grad) = self.eval_logp_y_val_grad(sest[t],
                                                        straj.y[t],
                                                        straj.t[t])
                logp_y += val
                logp_y_grad += grad
        return (logp_y / M, logp_y_grad / M)

    def eval_logp_xnext_val_grad_fulltraj(self, straj, ut, tt):
        logp_xnext_grad = numpy.zeros((len(self.params)))
        logp_xnext = 0.0
        sest = straj.get_smoothed_estimates()
        M = sest.shape[1]
        T = len(straj)
        for t in range(T - 1):
            (val, grad) = self.eval_logp_xnext_val_grad(sest[t],
                                                        sest[t + 1],
                                                        straj.u[t],
                                                        straj.t[t])
            logp_xnext += val
            logp_xnext_grad += grad

        return (logp_xnext / M, logp_xnext_grad / M)

    @abc.abstractmethod
    def eval_logp_x0_val_grad(self, particles, t):
        """
        Calculate term of the I1 integral approximation as specified in [1].
        Eg. Evaluate log p(x_0) or sum log p(x_0)

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - t (float): time stamp

        The gradient is an array where each element is the derivative with
        respect to the corresponding parameter

        Returns: ((array-like) or (float), array-like) (value, gradient)
        """
        pass

    @abc.abstractmethod
    def eval_logp_xnext_val_grad(self, particles, particles_next, u, t):
        """
        Calculate gradient of a term of the I2 integral approximation
        as specified in [1].

        Eg. Evaluate log p(x_{t+1}|x_t) or sum log p(x_{t+1}|x_t)

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - particles_next (array-like): future states
         - u (array-like): input signal
         - t (float): time stamp

        The gradient is an array where each element is the derivative with
        respect to the corresponding parameter

        Returns: ((array-like) or (float), array-like) (value, gradient)
        """
        pass

    @abc.abstractmethod
    def eval_logp_y_val_grad(self, particles, y, t):
        """
        Calculate gradient of a term of the I3 integral approximation
        as specified in [1].

        Eg. Evaluate log p(y_t|x_t) or sum log p(y_t|x_t)

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - y (array-like): measurement
         - t (float): time stamp


        The gradient is an array where each element is the derivative with
        respect to the corresponding parameter

        Returns: ((array-like) or (float), array-like) (value, gradient)
        """
        pass


class ParamEstBaseNumeric(ParamEstIntFullTraj):
    def __init__(self, param_bounds=None, **kwargs):
        self.param_bounds = param_bounds
        super(ParamEstBaseNumeric, self).__init__(**kwargs)

    def set_param_bounds(self, bounds):
        self.param_bounds = bounds

    def maximize(self, straj):
        def fval(params_val):
            """ internal function """
            self.set_params(params_val)
            log_py = self.eval_logp_y_fulltraj(straj,
                                               straj.y,
                                               straj.t)
            log_pxnext = self.eval_logp_xnext_fulltraj(straj,
                                                       straj.u,
                                                       straj.t)
            tmp = self.eval_logp_x0(straj.traj[0].pa.part,
                                    straj.t[0])
            log_px0 = numpy.mean(tmp)

            val = -1.0 * (log_py + log_px0 + log_pxnext)
            return val

        res = scipy.optimize.minimize(fun=fval, x0=self.params, method='l-bfgs-b', jac=False,
                                      options=dict({'maxiter':10, 'maxfun':100}),
                                      bounds=self.param_bounds,)
        return res.x

class ParamEstBaseNumericGrad(ParamEstInterface_GradientSearchFullTraj):
    def __init__(self, param_bounds=None, **kwargs):
        self.param_bounds = param_bounds
        super(ParamEstBaseNumericGrad, self).__init__(**kwargs)

    def set_param_bounds(self, bounds):
        self.param_bounds = bounds

    def maximize(self, straj):

        def fval_grad(params_val):
            """ internal function """
            self.set_params(params_val)
            (logp_y, grad_logp_y) = self.eval_logp_y_val_grad_fulltraj(straj,
                                                                       straj.y,
                                                                       straj.t)
            (logp_xnext, grad_logp_xnext) = self.eval_logp_xnext_val_grad_fulltraj(straj,
                                                                                   straj.u,
                                                                                   straj.t)

            (tmp1, tmp2) = self.eval_logp_x0_val_grad(straj.traj[0].pa.part,
                                                      straj.t[0])
            (logp_x0, grad_logp_x0) = (numpy.mean(tmp1), numpy.mean(tmp2))
            val = -1.0 * (logp_y + logp_x0 + logp_xnext)
            grad = -1.0 * (grad_logp_y + grad_logp_xnext + grad_logp_x0)
            return (val, grad)

        res = scipy.optimize.minimize(fun=fval_grad, x0=self.params, method='l-bfgs-b', jac=True,
                                      options=dict({'maxiter':10, 'maxfun':100}),
                                      bounds=self.param_bounds,)

        return res.x
