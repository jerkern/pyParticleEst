'''
Created on Sep 17, 2013

@author: ajn
'''
import param_est
import numpy
import matplotlib.pyplot as plt

from test.mixed_nlg.trans.particle_param_trans import ParticleParamTrans as PartModel # Our model definition

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
        
    
    
class GradientTest(param_est.ParamEstimation):
    
    def create_initial_estimate(self, params, num):
        self.params = params
        particles = numpy.empty(num, PartModel)
        e0 = numpy.array([0.0,])
        z0 = numpy.array([0.0,])
        P0 = numpy.eye(1)
        for k in range(len(particles)):
            particles[k] = PartModel(eta0=e0, z0=z0, P0=P0, params=params)
        return (particles, z0, P0)
    
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
            
            self.set_params(tmp.reshape((-1,1)))
            (logpy[k], grad_lpy[:,k]) = self.eval_logp_y()
            (logpxn[k], grad_lpxn[:,k]) = self.eval_logp_xnext()
            (logpx0[k], grad_lpx0[:,k]) = self.eval_logp_x0()

        self.plot_y = GradPlot(param_vals, logpy, grad_lpy)
        self.plot_xn = GradPlot(param_vals, logpxn, grad_lpxn)
        self.plot_x0 = GradPlot(param_vals, logpx0, grad_lpx0)

    
if __name__ == '__main__':
    
    num = 1
    
    theta_true = 0.1
    R = numpy.array([[0.1]])
    Q = numpy.array([ 0.1, 0.1])
    e0 = numpy.array([0.0, ])
    z0 = numpy.array([0.0, ])
    P0 = numpy.eye(1)
    
    # Create a reference which we will try to estimate using a RBPS
    correct = PartModel(eta0=e0, z0=z0,P0=P0, params=(theta_true,))

    # How many steps forward in time should our simulation run
    steps = 20
    

    # Create arrays for storing some values for later plotting    
    vals = numpy.zeros((2, num+1, steps+1))
    yvec = numpy.zeros((1, steps))


    # Create reference
    for i in range(steps):
        
        # Extract linear states
        vals[0,num,i]=correct.kf.z.reshape(-1)
        # Extract non-linear state
        vals[1,num,i]=correct.eta[0,0]

        # Drive the correct particle using the true input
        noise = correct.sample_process_noise()
        correct.update(u=None, noise=noise)
        correct.kf.z += numpy.random.normal(0.0, Q[1])
        # use the correct particle to generate the true measurement
        yvec[0,i] = correct.eta[0,0]

    
    # Store values for last time-step aswell    
    vals[0,num,steps]=correct.kf.z.reshape(-1)
    vals[1,num,steps]=correct.eta[0,0]

    y_noise = yvec.T.tolist()
    for i in range(len(y_noise)):
        y_noise[i][0] += numpy.random.normal(0.0,R)
    
    # Create an array for our particles 
    gt = GradientTest(u=None, y=y_noise)
    gt.set_params(numpy.array((theta_true,)).reshape((-1,1)))
    
    param_steps = 51
    param_vals = numpy.linspace(-1.0, 1.0, param_steps)
    gt.test(0, param_vals)

    gt.plot_y.plot(1)
    gt.plot_xn.plot(2)
    gt.plot_x0.plot(3)
    