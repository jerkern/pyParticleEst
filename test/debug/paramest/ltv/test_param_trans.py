#!/usr/bin/python

import numpy
import pyparticleest.paramest.paramest as paramest
import pyparticleest.paramest.gradienttest as gradienttest
import pyparticleest.paramest.interfaces as pestinf

import matplotlib.pyplot as plt
from pyparticleest.models.ltv import LTV

R = numpy.array([[0.1]])
Q = numpy.diag([ 0.1, 0.1])
gradient_test = True

def generate_reference(z0, P0, theta_true, steps):
    A = numpy.asarray(((1.0, theta_true), (0.0, 1.0)))
    C = numpy.array([[1.0, 0.0]])

    states = numpy.zeros((steps + 1, 2))
    y = numpy.zeros((steps, 1))
    x0 = numpy.random.multivariate_normal(z0.ravel(), P0)
    states[0] = numpy.copy(x0)
    for i in range(steps):

        # Calc linear states
        x = states[i].reshape((-1, 1))
        xn = A.dot(x) + numpy.random.multivariate_normal(numpy.zeros(x0.shape), Q).reshape((-1, 1))
        states[i + 1] = xn.ravel()

        # use the correct particle to generate the true measurement
        y[i] = C.dot(xn).ravel() + numpy.random.multivariate_normal((0.0,), R).reshape((-1, 1))

    return (y, states)

class ParticleParamTrans(LTV, pestinf.ParamEstBaseNumericGrad,
                         pestinf.ParamEstInterface_GradientSearch):
    """ Implement a simple system by extending the MixedNLGaussian class """
    def __init__(self, z0, P0, params):
        """ Define all model variables """
        self.params = numpy.copy(params)
        A = numpy.array([[1.0, params[0]], [0.0, 1.0]])
        self.A_grad = numpy.array([[0.0, 1.0], [0.0, 0.0]])[numpy.newaxis]
        C = numpy.array([[1.0, 0.0]])
        z0 = numpy.copy(z0).reshape((-1, 1))
        # Linear states handled by base-class
        super(ParticleParamTrans, self).__init__(z0=z0, P0=P0, A=A,
                                                C=C, R=R, Q=Q,)

    def set_params(self, params):
        """ New set of parameters """
        # Update all needed matrices and derivates with respect
        # to the new parameter set
        self.params = numpy.copy(params)
        A = numpy.array([[1.0, params[0]], [0.0, 1.0]])
        self.kf.set_dynamics(A=A)

    def get_pred_dynamics_grad(self, u, t):
        return (self.A_grad, None, None)

z0 = numpy.array([0.0, 1.0, ])
P0 = numpy.eye(2)

def callback(params, Q):
    print "Q=%f, params=%s" % (Q, params)
    return

if __name__ == '__main__':

    num = 1
    nums = 1

    theta_true = 0.1

    # How many steps forward in time should our simulation run
    steps = 200
    model = ParticleParamTrans(z0=z0, P0=P0, params=(theta_true,))

    def callback_sim(pe):
        plt.clf()
        plt.plot(range(steps + 1), x[:, 0], 'r-')
        plt.plot(range(steps + 1), x[:, 1], 'b-')
        sest = pe.straj.get_smoothed_estimates()
        plt.plot(range(steps + 1), sest[:, 0, 0], 'r--')
        plt.plot(range(steps + 1), sest[:, 0, 1], 'b--')
        plt.plot(range(steps + 1), sest[:, 0, 0] - numpy.sqrt(sest[:, 0, 2]), 'r--')
        plt.plot(range(steps + 1), sest[:, 0, 0] + numpy.sqrt(sest[:, 0, 2]), 'r--')
        plt.plot(range(steps + 1), sest[:, 0, 1] - numpy.sqrt(sest[:, 0, 5]), 'b--')
        plt.plot(range(steps + 1), sest[:, 0, 1] + numpy.sqrt(sest[:, 0, 5]), 'b--')
        plt.draw()

    if (gradient_test):
        (y, x) = generate_reference(z0, P0, theta_true, steps)
        gt = gradienttest.GradientTest(model, u=None, y=y)

        gt.set_params(numpy.array((theta_true,)))

        param_steps = 101
        param_vals = numpy.linspace(-0.1, 0.3, param_steps)
        gt.test(0, param_vals, num=num, analytic_grad=True)

        plt.figure(1)
        callback_sim(gt)


        gt.plot_y.plot(2)
        gt.plot_xn.plot(3)
        gt.plot_x0.plot(4)
        plt.show()
    else:
        max_iter = 200
        sims = 100
        tol = 0.0
        plt.ion()
        theta_guess = 0.3
        # theta_guess = theta_true
        # Create arrays for storing some values for later plotting
        vals = numpy.zeros((2, num + 1, steps + 1))
        yvec = numpy.zeros((1, steps))

        estimate = numpy.zeros((1, sims))

        plt.ion()
        fig1 = plt.figure()
        fig2 = plt.figure()

        for k in range(sims):
            print k
            # Create reference
            (y, x) = generate_reference(z0, P0, theta_true, steps)

            plt.figure(fig1.number)
            plt.clf()


            # Create an array for our particles
            pe = paramest.ParamEstimation(model=model, u=None, y=y)
            pe.set_params(numpy.array((theta_guess,)).reshape((-1, 1)))

            # ParamEstimator.simulate(num_part=num, num_traj=nums)
            print "maximization start"
#            (param, Qval) = pe.maximize(param0=numpy.array((theta_guess,)), num_part=num, num_traj=nums,
#                                        max_iter=max_iter, callback_sim=callback_sim, tol=tol, callback=callback,
#                                        analytic_gradient=False)
            (param, Qval) = pe.maximize(param0=numpy.array((theta_guess,)),
                                        num_part=num, num_traj=nums,
                                        max_iter=max_iter, tol=tol)
            estimate[0, k] = param

            plt.figure(fig2.number)
            plt.clf()
            plt.hist(estimate[0, :(k + 1)].T)
            fig2.show()
            plt.show()
            plt.draw()



    plt.ioff()
    if (not gradient_test):
        print "mean: %f" % numpy.mean(estimate)
        print "stdd: %f" % numpy.std(estimate)
        plt.hist(estimate.T)
        plt.show()
        plt.draw()
    print "exit"
