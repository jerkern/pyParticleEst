""" Parameter estimation classes

@author: Jerker Nordh
"""
from pyparticleest.simulator import Simulator
import numpy
import scipy.optimize
import matplotlib.pyplot as plt


class ParamEstimation(Simulator):
    """
    Extension of the Simulator class to iterative perform particle smoothing
    combined with a gradienst search algorithms for maximizing the likelihood
    of the parameter estimates
    """

    def maximize(self, param0, num_part, num_traj, max_iter=1000, tol=0.001,
                 callback=None, callback_sim=None, bounds=None, meas_first=False,
                 smoother='full', smoother_options=None, analytic_gradient=False):
        """
        Find the maximum likelihood estimate of the paremeters using an
        EM-algorihms combined with a gradient search algorithms

        Args:
         - param0 (array-like): Initial parameter estimate
         - num_part (int/array-like): Number of particle to use in the forward filter
           if array each iteration takes the next element from the array when setting
           up the filter
         - num_traj (int/array-like): Number of smoothed trajectories to create
           if array each iteration takes the next element from the array when setting
           up the smoother
         - max_iter (int): Max number of EM-iterations to perform
         - tol (float): When the different in loglikelihood between two iterations
           is less that his value the algorithm is terminated
         - callback (function): Callback after each EM-iteration with new estimate
         - callback_sim (function): Callback after each simulation
         - bounds (array-like): Hard bounds on parameter estimates
         - meas_first (bool): If true, first measurement occurs before the first
           time update
         - smoother (string): Which particle smoother to use
         - smoother_options (dict): Extra options for the smoother
         - analytic_gradient (bool): Use analytic gradient (requires that the model
           implements ParamEstInterface_GradientSearch)
        """


        def fval(params_val):
            """ internal function """
            self.model.set_params(params_val)
            log_py = self.eval_logp_y()
            log_pxnext = self.eval_logp_xnext()
            log_px0 = self.eval_logp_x0()
            val = -1.0 * (log_py + log_px0 + log_pxnext)
            return val

        def fval_grad(params_val):
            """ internal function """
            self.model.set_params(params_val)
            (logp_y, grad_logp_y) = self.eval_logp_y_val_grad()
            (logp_xnext, grad_logp_xnext) = self.eval_logp_xnext_val_grad()
            (logp_x0, grad_logp_x0) = self.eval_logp_x0_val_grad()
            val = -1.0 * (logp_y + logp_x0 + logp_xnext)
            grad = -1.0 * (grad_logp_y + grad_logp_xnext + grad_logp_x0)
            return (val, grad)

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

            self.simulate(nump, numt, smoother=smoother, smoother_options=smoother_options,
                          meas_first=meas_first)
            if (callback_sim != None):
                callback_sim(self)
            # res = scipy.optimize.minimize(fun=fval, x0=params, method='nelder-mead', jac=fgrad)
            if (analytic_gradient):
                res = scipy.optimize.minimize(fun=fval_grad, x0=params_local, method='l-bfgs-b', jac=True,
                                              options=dict({'maxiter':10, 'maxfun':100}), bounds=bounds,)
            else:
                res = scipy.optimize.minimize(fun=fval, x0=params_local, method='l-bfgs-b', jac=False,
                                              options=dict({'maxiter':10, 'maxfun':100}), bounds=bounds,)

            params_local = res.x

            # (Q, Q_grad) = fval(params_local)
            Q = fval(params_local)
            Q = -Q
            # Q_grad = -Q_grad
            if (callback != None):
                callback(params=params_local, Q=Q)
            if (numpy.abs(Q - Q_old) < tol):
                break
        return (params_local, Q)

    def eval_prob(self):
        """ internal helper function """
        log_py = self.eval_logp_y()
        log_px0 = self.eval_logp_x0()
        log_pxnext = self.eval_logp_xnext()
        return log_px0 + log_pxnext + log_py

    def eval_logp_x0(self):
        """ internal helper function """
        M = self.straj.traj.shape[1]
        logp_x0 = self.model.eval_logp_x0(self.straj.traj[0],
                                          self.straj.t[0])
        return numpy.sum(logp_x0) / M

    def eval_logp_y(self, ind=None, traj_ind=None):
        """ internal helper function """
        logp_y = 0.0
        M = self.straj.traj.shape[1]
        T = len(self.straj)
        for t in xrange(T):
            if (self.straj.y[t] != None):
                val = self.model.eval_logp_y(self.straj.traj[t],
                                             self.straj.y[t],
                                             self.straj.t[t])
                logp_y += numpy.sum(val)

        return logp_y / M

    def eval_logp_xnext(self, ind=None, traj_ind=None):
        """ internal helper function """
        logp_xnext = 0.0
        M = self.straj.traj.shape[1]
        T = len(self.straj)
        for t in xrange(T - 1):
            val = self.model.eval_logp_xnext(self.straj.traj[t],
                                             self.straj.traj[t + 1],
                                             self.straj.u[t],
                                             self.straj.t[t])
            logp_xnext += numpy.sum(val)
        return logp_xnext / M

    def eval_logp_x0_val_grad(self):
        """ internal helper function """
        M = self.straj.traj.shape[1]
        (logp_x0, logp_x0_grad) = self.model.eval_logp_x0_val_grad(self.straj.traj[0],
                                                                   self.straj.t[0])
        return (logp_x0 / M, logp_x0_grad / M)

    def eval_logp_y_val_grad(self, ind=None, traj_ind=None):
        """ internal helper function """
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
        return (logp_y / M, logp_y_grad / M)

    def eval_logp_xnext_val_grad(self, ind=None, traj_ind=None):
        """ internal helper function """
        logp_xnext_grad = numpy.zeros((len(self.model.params)))
        logp_xnext = 0.0
        M = self.straj.traj.shape[1]
        T = len(self.straj)
        for t in xrange(T - 1):
            (val, grad) = self.model.eval_logp_xnext_val_grad(self.straj.traj[t],
                                                              self.straj.traj[t + 1],
                                                              self.straj.u[t],
                                                              self.straj.t[t])
            logp_xnext += val
            logp_xnext_grad += grad

        return (logp_xnext / M, logp_xnext_grad / M)

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
                    self.draw_gradient(self.params[k], self.vals[k], self.params[k] - self.params[k - 1], self.diff[k])


    def draw_gradient(self, x, y, dx, dydx):
        plt.plot((x - dx, x + dx), (y - dydx * dx, y + dydx * dx), 'r')


class GradientTest(ParamEstimation):

    def test(self, param_id, param_vals, num=100, nums=1, analytic_grad=True):
        self.simulate(num_part=num, num_traj=nums)
        param_steps = len(param_vals)
        logpy = numpy.zeros((param_steps,))
        logpxn = numpy.zeros((param_steps,))
        logpx0 = numpy.zeros((param_steps,))
        if (analytic_grad):
            grad_logpy = numpy.zeros((param_steps, len(self.params)))
            grad_logpxn = numpy.zeros((param_steps, len(self.params)))
            grad_logpx0 = numpy.zeros((param_steps, len(self.params)))
        for k in range(param_steps):
            tmp = numpy.copy(self.params)
            tmp[param_id] = param_vals[k]
            self.set_params(tmp)
            logpy[k] = self.eval_logp_y()
            logpxn[k] = self.eval_logp_xnext()
            logpx0[k] = self.eval_logp_x0()
            if (analytic_grad):
                grad_logpy[k] = self.eval_logp_y_val_grad()[1]
                grad_logpxn[k] = self.eval_logp_xnext_val_grad()[1]
                grad_logpx0[k] = self.eval_logp_x0_val_grad()[1]
                self.plot_y = GradPlot(param_vals, logpy, grad_logpy[:, param_id])
                self.plot_xn = GradPlot(param_vals, logpxn, grad_logpxn[:, param_id])
                self.plot_x0 = GradPlot(param_vals, logpx0, grad_logpx0[:, param_id])
            else:
                self.plot_y = GradPlot(param_vals, logpy, None)
                self.plot_xn = GradPlot(param_vals, logpxn, None)
                self.plot_x0 = GradPlot(param_vals, logpx0, None)



