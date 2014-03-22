'''
Created on Sep 17, 2013

@author: ajn
'''
import pyparticleest.param_est as param_est
import numpy
import math
import matplotlib.pyplot as plt

import pyparticleest.test.mixed_nlg.trans.particle_param_trans  as PartModel # Our model definition

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
    nums = 5
    theta_true = 0.3
    R = 0.1*numpy.eye(1)
    Qxi = 0.1*numpy.eye(1)
    Qz = 0.1*numpy.eye(1)

    # How many steps forward in time should our simulation run
    steps = 200

    (x, y) = PartModel.generate_data(theta_true, Qxi, Qz, R, steps)

    # Create an array for our particles 
    model = PartModel.ParticleParamTrans((theta_true,), 
                                         R=R,
                                         Qxi=Qxi,
                                         Qz=Qz)
    gt = GradientTest(model, u=None, y=y)
    gt.set_params(numpy.array((theta_true,)))
    #gt.simulate(num, nums)
    param_steps = 21
    param_vals = numpy.linspace(-1, 1, param_steps)
    gt.test(0, param_vals, num=num, nums=nums)
     
    svals = numpy.zeros((2, nums, steps+1))
    vals = numpy.zeros((2, num, steps+1))
    t = numpy.asarray(range(steps+1))
    
    for i in range(steps+1):
        (xil, zl, Pl) = model.get_states(gt.straj.straj[i])
        svals[0,:,i] = numpy.vstack(xil).ravel()
        svals[1,:,i] = numpy.vstack(zl).ravel()
        (xil, zl, Pl) = model.get_states(gt.pt.traj[i].pa.part)
        vals[0,:,i]=numpy.vstack(xil).ravel()
        vals[1,:,i]=numpy.vstack(zl).ravel()
    

    for j in range(num):   
        plt.plot(range(steps+1),vals[0,j,:],'r.',markersize=1.0)
        plt.plot(range(steps+1),vals[1,j,:],'b.',markersize=1.0)
    for j in range(nums):
        plt.plot(range(steps+1),svals[0,j,:],'r--')
        plt.plot(range(steps+1),svals[1,j,:],'b--')
    
    tmp = numpy.hstack(x)
    plt.plot(t, tmp[0,:],'k-')
    plt.plot(t, tmp[1,:],'k-')
    #plt.plot(range(steps+1),svals_mean[1,:],'--', markersize=1.0, color='1.0')
    #plt.savefig('rbps_fail_theta.eps', bbox_inches='tight')
    plt.show()

    plt.draw()
    
    #gt.plot_y.plot(1)
    gt.plot_xn.plot(1)
    #gt.plot_x0.plot(3)
    