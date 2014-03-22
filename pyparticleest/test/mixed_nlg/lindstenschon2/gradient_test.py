'''
Created on Nov 11, 2013

@author: ajn
'''

import pyparticleest.param_est as param_est
import numpy
import math
import matplotlib.pyplot as plt
import pyparticleest.test.mixed_nlg.lindstenschon2.particle_ls2 as particle_ls2

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
    
    
class GradientTest(param_est.ParamEstimation):
    
    def test(self, param_id, param_vals, num=100, nums=1):
        self.simulate(num_part=num, num_traj=nums)
        param_steps = len(param_vals)
        logpy = numpy.zeros((param_steps,))
        #grad_lpy = numpy.zeros((len(self.params), param_steps))
        logpxn = numpy.zeros((param_steps,))
        #grad_lpxn = numpy.zeros((len(self.params), param_steps))
        logpx0 = numpy.zeros((param_steps,))
        #grad_lpx0 = numpy.zeros((len(self.params), param_steps))
        for k in range(param_steps):    
            tmp = numpy.copy(self.params)
            tmp[param_id] = param_vals[k]
            self.set_params(tmp)
            logpy[k] = self.eval_logp_y()
            logpxn[k]  = self.eval_logp_xnext()
            logpx0[k] = self.eval_logp_x0()

#        self.plot_y = GradPlot(param_vals, logpy, grad_lpy[param_id,:])
#        self.plot_xn = GradPlot(param_vals, logpxn, grad_lpxn[param_id,:])
#        self.plot_x0 = GradPlot(param_vals, logpx0, grad_lpx0[param_id,:])
        self.plot_y = GradPlot(param_vals, logpy, None)
        self.plot_xn = GradPlot(param_vals, logpxn, None)
        self.plot_x0 = GradPlot(param_vals, logpx0, None)

    
if __name__ == '__main__':
    num = 50
    nums = 4
    
    theta_true = numpy.array((1.0, 1.0, 0.3, 0.968, 0.315))

    # How many steps forward in time should our simulation run
    steps = 200
    sims = 1

    # Create arrays for storing some values for later plotting    
    vals = numpy.zeros((2, num+1, steps+1))

    estimate = numpy.zeros((5,sims))

    # How many steps forward in time should our simulation run
    steps = 200

    (y, e, z) = particle_ls2.generate_dataset(theta_true, steps)
    
    model = particle_ls2.ParticleLS2(theta_true)
    # Create an array for our particles 
    gt = GradientTest(model=model, u=None, y=y)
    gt.set_params(theta_true)
    
    param_id = 4
    param_steps = 101
    tval = theta_true[param_id]
    param_vals = numpy.linspace(tval-math.fabs(tval), tval+math.fabs(tval), param_steps)
    gt.test(param_id, param_vals, nums=nums)

    gt.plot_y.plot(1)
    gt.plot_xn.plot(2)
    gt.plot_x0.plot(3)