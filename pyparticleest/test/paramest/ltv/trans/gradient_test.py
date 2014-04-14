'''
Created on Sep 17, 2013

@author: ajn
'''
import pyparticleest.param_est as param_est
import numpy
import matplotlib.pyplot as plt

from pyparticleest.test.ltv.trans.particle_param_trans import ParticleParamTrans as PartModel # Our model definition
import pyparticleest.test.ltv.trans.particle_param_trans as particle_param_trans
class GradPlot():
    def __init__(self, params, vals):
        self.params = params
        self.vals = vals

    def plot(self, fig_id):
        fig = plt.figure(fig_id)
        fig.clf()
        plt.plot(self.params, self.vals)
#        for k in range(len(self.params)):
#            if (k % 10 == 1):
#                self.draw_gradient(self.params[k], self.vals[k], self.params[k]-self.params[k-1], self.diff[:,k])
                
        plt.show()
        
    def draw_gradient(self, x, y, dx, dydx):
        plt.plot((x-dx, x+dx), (y-dydx*dx, y+dydx*dx), 'r')
        
z0 = numpy.array([0.0, 0.0]).reshape((-1,1))
P0 = 100*numpy.eye(2)    
    
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

        self.plot_y = GradPlot(param_vals, logpy)
        self.plot_xn = GradPlot(param_vals, logpxn)
        self.plot_x0 = GradPlot(param_vals, logpx0)

    def plot_estimate(self, states, y):
        svals = numpy.zeros((2, 1, steps+1))
        for i in range(steps+1):
            for j in range(len(self.straj)):
                svals[:,j,i]=self.straj.traj[i,j,:2].ravel()
                
        plt.figure()
        y = numpy.asarray(y)
        y = y.ravel()
        for j in range(1):
            plt.plot(range(steps+1),svals[0,j,:],'g-')
            plt.plot(range(steps+1),svals[1,j,:],'r-')
            plt.plot(range(steps+1),states[0,:],'go')
            plt.plot(range(steps+1),states[1,:],'ro')
            plt.plot(range(1,steps+1),y,'bx')
            
        plt.show()
        plt.draw()
    
    
if __name__ == '__main__':
    
    num = 1
    
    theta_true = 0.1
    theta_guess = 0.1

    
    steps = 200
    
    # Create a reference which we will try to estimate using a RBPS
    (ylist, states) = particle_param_trans.generate_reference(z0, P0, theta_true, steps)
    model = PartModel(z0=z0,P0=P0, params=(theta_true,))

    # Create an array for our particles 
    gt = GradientTest(model, u=None, y=ylist)
    gt.set_params(numpy.array((theta_guess,)))
    
    param_steps = 101
    param_vals = numpy.linspace(-0.1, 0.3, param_steps)
    gt.test(0, param_vals, num=num)

    gt.plot_y.plot(1)
    gt.plot_xn.plot(2)
    gt.plot_x0.plot(3)
    
#    gt.simulate(1, 1)
#    gt.plot_estimate(states=states, y=ylist)