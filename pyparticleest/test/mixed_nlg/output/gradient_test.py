'''
Created on Sep 17, 2013

@author: ajn
'''
import param_est
import numpy
import matplotlib.pyplot as plt

import particle_param_output
from test.mixed_nlg.output.particle_param_output import ParticleParamOutput as PartModel # Our model definition

class GradPlot():
    def __init__(self, params, vals, diff):
        self.params = params
        self.vals = vals
        self.diff = diff

    def plot(self, fig_id):
        fig = plt.figure(fig_id)
        fig.clf()
        plt.plot(self.params, self.vals)
        for k in range(len(self.params)):
            if (k % 10 == 1):
                self.draw_gradient(self.params[k], self.vals[k], self.params[k]-self.params[k-1], self.diff[:,k])
                
        plt.show()
        
    def draw_gradient(self, x, y, dx, dydx):
        plt.plot((x-dx, x+dx), (y-dydx*dx, y+dydx*dx), 'r')
        


z0 = numpy.array([[0.0], [0.0]])
P0 = 1000.0*numpy.eye(2)
Q_in = numpy.diag([ 0.1, 0.1])
B = numpy.array([[0.0, 0.0], [1.0, -1.0]])
Qz = numpy.diag([ 0.0000001, 0.0000001])+B.dot(Q_in.dot(B.T))
R = numpy.array([[1.0]])
    
class GradientTest(param_est.ParamEstimation):

    def create_initial_estimate(self, params, num):
        self.params = params
        particles = numpy.empty(num, PartModel)
        for k in range(len(particles)):
            particles[k] = PartModel(x0=z0, P0=P0, Qz=Qz, R=R, params=params)
        return particles
    
    def test(self, param_id, param_vals, num=100, nums=1):
        self.simulate(num_part=num, num_traj=nums)
        param_steps = len(param_vals)
        logpy = numpy.zeros((param_steps,))
        grad_lpy = numpy.zeros((len(self.params), param_steps))
        logpxn = numpy.zeros((param_steps,))
        grad_lpxn = numpy.zeros((len(self.params), param_steps))
        logpx0 = numpy.zeros((param_steps,))
        grad_lpx0 = numpy.zeros((len(self.params), param_steps))
        for k in range(param_steps):    
            tmp = numpy.copy(self.params)
            tmp[param_id] = param_vals[k]
            
            self.set_params(tmp.ravel())
            logpy[k] = self.eval_logp_y()
            logpxn[k] = self.eval_logp_xnext()
            logpx0[k] = self.eval_logp_x0()
            grad_lpy[:,k] = self.eval_grad_logp_y()
            grad_lpxn[:,k] = self.eval_grad_logp_xnext()
            grad_lpx0[:,k] = self.eval_grad_logp_x0()

        self.plot_y = GradPlot(param_vals, logpy, grad_lpy)
        self.plot_xn = GradPlot(param_vals, logpxn, grad_lpxn)
        self.plot_x0 = GradPlot(param_vals, logpx0, grad_lpx0)

    
if __name__ == '__main__':
    
    num = 1
    nums=1
    sims = 1
    
    c_true = 2.5
    c_guess = numpy.array((2.5,))
    
    # How many steps forward in time should our simulation run
    steps = 32
    
    # Create a random input vector to drive our "correct state"
    mean = -numpy.hstack((-1.0*numpy.ones(steps/4), 1.0*numpy.ones(steps/2),-1.0*numpy.ones(steps/4) ))
    diff_vec = numpy.random.normal(mean, .1, steps)
    uvec = numpy.vstack((1.0-diff_vec/2, 1.0+diff_vec/2))

    # Create reference
    (ulist, ylist, states) = particle_param_output.generate_refernce(z0, P0, Qz, R, uvec, steps, c_true)

    # Create an array for our particles 
    gt = GradientTest(u=ulist, y=ylist)
    gt.set_params(numpy.array((c_true,)).reshape((-1,1)))
    
    param_steps = 101
    param_vals = numpy.linspace(-10.0, 10.0, param_steps)
    gt.test(0, param_vals)

    gt.plot_y.plot(1)
    gt.plot_xn.plot(2)
    gt.plot_x0.plot(3)
    