'''
Created on Mar 27, 2015

@author: ajn
'''
import numpy
import matplotlib.pyplot as plt
from pyparticleest.paramest.paramest import ParamEstimation

class GradPlot():
    def __init__(self, params, vals, diff):
        self.params = params
        self.vals = vals
        self.diff = diff

    def plot(self, fig_id):
        fig = plt.figure(fig_id)
        fig.clf()
        plt.plot(self.params, self.vals)
        if (self.diff is not None):
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
            logpy[k] = self.model.eval_logp_y_fulltraj(self.straj,
                                                       self.straj.y,
                                                       self.straj.t)
            logpxn[k] = self.model.eval_logp_xnext_fulltraj(self.straj,
                                                             self.straj.u,
                                                             self.straj.t)
            tmp = self.model.eval_logp_x0(self.straj.traj[0].pa.part,
                                          self.straj.t[0])
            logpx0[k] = numpy.mean(tmp)

            if (analytic_grad):
                (_, grad_logp_y) = self.model.eval_logp_y_val_grad_fulltraj(self.straj,
                                                                             self.straj.y,
                                                                             self.straj.t)
                (_, grad_logp_xnext) = self.model.eval_logp_xnext_val_grad_fulltraj(self.straj,
                                                                                     self.straj.u,
                                                                                     self.straj.t)
                (tmp1, tmp2) = self.model.eval_logp_x0_val_grad(self.straj.traj[0].pa.part,
                                                                self.straj.t[0])
                (_, grad_logp_x0) = (numpy.mean(tmp1), numpy.mean(tmp2))

                grad_logpy[k] = grad_logp_y
                grad_logpxn[k] = grad_logp_xnext
                grad_logpx0[k] = grad_logp_x0
                self.plot_y = GradPlot(param_vals, logpy, grad_logpy[:, param_id])
                self.plot_xn = GradPlot(param_vals, logpxn, grad_logpxn[:, param_id])
                self.plot_x0 = GradPlot(param_vals, logpx0, grad_logpx0[:, param_id])
            else:
                self.plot_y = GradPlot(param_vals, logpy, None)
                self.plot_xn = GradPlot(param_vals, logpxn, None)
                self.plot_x0 = GradPlot(param_vals, logpx0, None)

