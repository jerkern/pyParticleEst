""" Parameter estimation classes

@author: Jerker Nordh
"""
from pyparticleest.simulator import Simulator
import numpy

from builtins import range


class ParamEstimation(Simulator):
    """
    Extension of the Simulator class to iterative perform particle smoothing
    combined with a gradienst search algorithms for maximizing the likelihood
    of the parameter estimates
    """

    def maximize(self, param0, num_part, num_traj, max_iter=1000, tol=0.001,
                 callback=None, callback_sim=None, meas_first=False,
                 filter='pf', smoother='full', smoother_options=None):
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

        params_local = numpy.copy(param0)
        Q = -numpy.Inf
        for i in range(max_iter):
            Q_old = Q
            self.set_params(params_local)
            if (numpy.isscalar(num_part)):
                nump = num_part
            else:
                if (i < len(num_part)):
                    nump = num_part[i]
                else:
                    nump = num_part[-1]
            if (numpy.isscalar(num_traj)):
                numt = num_traj
            else:
                if (i < len(num_traj)):
                    numt = num_traj[i]
                else:
                    numt = num_traj[-1]

            self.simulate(nump, numt, filter=filter, smoother=smoother,
                          smoother_options=smoother_options, meas_first=meas_first)
            if (callback_sim is not None):
                callback_sim(self)

            params_local = self.model.maximize(self.straj)
            # res = scipy.optimize.minimize(fun=fval, x0=params, method='nelder-mead', jac=fgrad)

            # FIXME: Q value not accesible (not needed?)
            # Should the callback define when we terminate the EM-algorithm?
            # (Q, Q_grad) = fval(params_local)
            #Q = fval(params_local)
            #Q = -Q
            # Q_grad = -Q_grad
            if (callback is not None):
                callback(params=params_local, Q=-numpy.Inf, cur_iter=i)  # , Q=Q)
#            if (numpy.abs(Q - Q_old) < tol):
#                break
        # return (params_local, Q)
        return (params_local, -numpy.Inf)


def alpha_gen(it):
    offset = 100
    if (it <= offset):
        return 1
    else:
        return (it - offset) ** (-0.51)


class ParamEstimationSAEM(Simulator):
    """
    Extension of the Simulator class to iterative perform particle smoothing
    combined with a gradienst search algorithms for maximizing the likelihood
    of the parameter estimates
    """

    def maximize(self, param0, num_part, num_traj, max_iter=1000, tol=0.001,
                 callback=None, callback_sim=None, meas_first=False,
                 filter='pf', filter_options=None,
                 smoother='full', smoother_options=None,
                 alpha_gen=alpha_gen):
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

        params_local = numpy.copy(param0)
        alltrajs = None
        weights = None

        for i in range(max_iter):
            self.set_params(params_local)

            if (numpy.isscalar(num_part)):
                nump = num_part
            else:
                if (i < len(num_part)):
                    nump = num_part[i]
                else:
                    nump = num_part[-1]
            if (numpy.isscalar(num_traj)):
                numt = num_traj
            else:
                if (i < len(num_traj)):
                    numt = num_traj[i]
                else:
                    numt = num_traj[-1]

            self.simulate(nump, numt, filter=filter, filter_options=filter_options,
                          smoother=smoother, smoother_options=smoother_options,
                          meas_first=meas_first)

            w = numpy.ones((numt,)) / numt
            newtrajs = numpy.copy(self.straj.traj)
            alpha = alpha_gen(i)
            if (weights is None):
                weights = w
            else:
                weights = numpy.concatenate(((1.0 - alpha) * weights, alpha * w))

            if (callback_sim is not None):
                callback_sim(self)

            if (alltrajs is None):
                alltrajs = numpy.copy(newtrajs)
            else:
                alltrajs = numpy.concatenate((alltrajs, newtrajs), axis=1)

            zero_ind = (weights == 0.0)
            weights = weights[~zero_ind]
            alltrajs = alltrajs[:, ~zero_ind]
            params_local = self.model.maximize_weighted(self.straj, alltrajs, weights)

            if (callback is not None):
                callback(params=params_local, Q=-numpy.Inf, cur_iter=i)  # , Q=Q)
        return (params_local, -numpy.Inf)


class ParamEstimationPSAEM(Simulator):
    """
    Extension of the Simulator class to iterative perform particle smoothing
    combined with a gradienst search algorithms for maximizing the likelihood
    of the parameter estimates
    """

    def maximize(self, param0, num_part, max_iter=1000, tol=0.001,
                 callback=None, callback_sim=None, meas_first=False,
                 filter='cpfas', filter_options=None,
                 alpha_gen=alpha_gen, discard_eps=0.0, discard_percentile=0,
                 M=1, smoother='ancestor', raoblackwell=False, max_traj=0):
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

        params_local = numpy.copy(param0)
        alltrajs = None
        weights = None

        def default_callback(params, Q, cur_iter):
            if (cur_iter >= max_iter):
                return True

        if (callback is None):
            callback = default_callback

        ind = numpy.asarray(range(num_part), dtype=numpy.int)
        i = 0
        while (True):
            i += 1
            self.set_params(params_local)

            self.simulate(num_part, M, filter=filter, filter_options=filter_options,
                          smoother=smoother, meas_first=meas_first)

            if (raoblackwell == True):
                tmp = self.straj.calculate_ancestors(self.pt, ind)
                w = numpy.exp(self.pt.traj[-1].pa.w)
                w = numpy.copy(w / numpy.sum(w))
                N = tmp[0].pa.part.shape[0]
                T = len(tmp)
                D = tmp[0].pa.part.shape[1]
                newtrajs = numpy.empty((T, N, D))

            else:
                newtrajs = self.get_smoothed_estimates()
                N = newtrajs.shape[1]
                w = numpy.ones((N,)) / float(N)

            alpha = alpha_gen(i - 1)
            if (weights is None):
                weights = w
            else:
                weights = numpy.concatenate(((1.0 - alpha) * weights, alpha * w))

            filter_options['cond_traj'] = numpy.copy(self.straj.traj)
            if (callback_sim is not None):
                callback_sim(self)

            if (alltrajs is None):
                alltrajs = numpy.copy(newtrajs)
            else:
                alltrajs = numpy.concatenate((alltrajs, newtrajs), axis=1)

            # Discard at max the lowest 'discard_percentile' of the weights
            tmp = numpy.percentile(weights, discard_percentile)
            wlow = numpy.max(numpy.hstack((weights[weights < tmp], 0.0)))
            threshold = min(discard_eps, wlow)
            zero_ind = (weights <= threshold)
            weights = weights[~zero_ind]
            alltrajs = alltrajs[:, ~zero_ind]

            if (max_traj > 0 and len(weights) > max_traj):
                weights = weights[-max_traj:]
                alltrajs = alltrajs[:, -max_traj:]
            # Make sure weights sum to one
            weights /= numpy.sum(weights)

            params_local = self.model.maximize_weighted(self.straj, alltrajs, weights)

            if (callback is not None):
                rval = callback(params=params_local, Q=-numpy.Inf, cur_iter=i)
                if (rval):
                    break
        return (params_local, -numpy.Inf)


class ParamEstimationPSAEM2(Simulator):
    """
    Extension of the Simulator class to iterative perform particle smoothing
    combined with a gradienst search algorithms for maximizing the likelihood
    of the parameter estimates
    """

    def maximize(self, param0, num_part, max_iter=1000, tol=0.001,
                 callback=None, callback_sim=None, meas_first=False,
                 filter='cpfas', filter_options=None,
                 smoother='full', smoother_options=None, alpha_gen=alpha_gen,
                 discard_eps=0.0, discard_percentile=0):
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

        params_local = numpy.copy(param0)
        alltrajs = None
        weights = numpy.empty((max_iter * num_part,))

        datalen = 0
        for i in range(max_iter):
            self.set_params(params_local)

            self.simulate(num_part, 1, filter=filter, filter_options=filter_options,
                          smoother=smoother, smoother_options=smoother_options,
                          meas_first=meas_first)

            tmp = numpy.copy(self.straj.traj)
            T = len(tmp)
            N = tmp[0].pa.part.shape[0]
            D = tmp[0].pa.part.shape[1]

            newtrajs = numpy.empty((T, N, D))

            for t in range(T):
                newtrajs[t] = tmp[t].pa.part

            w = 1.0

            alpha = alpha_gen(i)
            weights[:datalen] *= (1.0 - alpha)
            weights[datalen:datalen + 1] = alpha * w

#            weights[datalen:datalen + 1] = alpha * w

            filter_options['cond_traj'] = numpy.copy(self.straj.traj)
            if (callback_sim is not None):
                callback_sim(self)

            if (alltrajs is None):
                alltrajs = numpy.copy(newtrajs)
            else:
                alltrajs = numpy.concatenate((alltrajs[:, :datalen], newtrajs), axis=1)

            datalen += 1

            # Discard at max the lowest 'discard_percentile' of the weights
            tmp = numpy.percentile(weights[:datalen], discard_percentile)
            wlow = numpy.max(numpy.hstack(
                (weights[:datalen][weights[:datalen] < tmp], 0.0)))
            threshold = min(discard_eps, wlow)

            zero_ind = (weights[:datalen] <= threshold)
            zerolen = numpy.count_nonzero(zero_ind)
            weights[:datalen - zerolen] = weights[:datalen][~zero_ind]
            alltrajs[:, :datalen - zerolen] = alltrajs[:,
                                                       :datalen][:, ~zero_ind]
            datalen -= zerolen
            weights[:datalen] /= numpy.sum(weights[:datalen])
            params_local = self.model.maximize_weighted(self.straj, alltrajs[:, :datalen], weights[:datalen])
#            params_local = self.model.maximize_weighted(self.straj, alltrajs[:, -1:], numpy.asarray((1.0,)))

            if (callback is not None):
                callback(params=params_local, Q=-numpy.Inf,
                         cur_iter=i + 1)  # , Q=Q)
        return (params_local, -numpy.Inf)
