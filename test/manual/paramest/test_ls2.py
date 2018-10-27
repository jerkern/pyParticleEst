'''
Created on Nov 11, 2013

@author: Jerker Nordh
'''

import numpy
import math
import matplotlib.pyplot as plt
import pyparticleest.paramest.paramest as param_est
import pyparticleest.paramest.interfaces as pestinf
import pyparticleest.paramest.gradienttest as gradienttest
import sys

import pyparticleest.models.mlnlg as mlnlg

def sign(x):
    if (x < 0.0):
        return -1.0
    else:
        return 1.0

def calc_h(eta):
    return numpy.asarray(((0.1 * eta[0, 0] * math.fabs(eta[0, 0])),
                          0.0)).reshape((-1, 1))

def generate_dataset(params, length):
    Ae = numpy.array([[params[1], 0.0, 0.0]])
    Az = numpy.asarray(((1.0, params[2], 0.0),
                        (0.0,
                         params[3] * math.cos(params[4]),
                         - params[3] * math.sin(params[4])),
                         (0.0,
                         params[3] * math.sin(params[4]),
                         params[3] * math.cos(params[4]))))

    C = numpy.array([[0.0, 0.0, 0.0], [1.0, -1.0, 1.0]])

    e_vec = numpy.zeros((1, length))
    z_vec = numpy.zeros((3, length))

    e = numpy.asarray(((numpy.random.normal(0.0, 1.0),),))
    z = numpy.zeros((3, 1))

    e_vec[:, 0] = e.ravel()
    z_vec[:, 0] = z.ravel()

    y = numpy.zeros((2, length))
    h = calc_h(e)
    y[:, 0] = (h + C.dot(z)).ravel()

    for i in range(1, length):
        e = params[0] * numpy.arctan(e) + Ae.dot(z) + numpy.random.normal(0.0, math.sqrt(0.01))

        wz = numpy.random.multivariate_normal(numpy.zeros((3,)), 0.01 * numpy.eye(3, 3)).ravel().reshape((-1, 1))

        z = Az.dot(z) + wz
        h = calc_h(e)
        y[:, i] = (h + C.dot(z)).ravel()
        e_vec[:, i] = e.ravel()
        z_vec[:, i] = z.ravel()

    return (y.T.tolist(), e_vec, z_vec)

class ParticleLS2(mlnlg.MixedNLGaussianSampledInitialGaussian,
                  pestinf.ParamEstBaseNumericGrad,
                  pestinf.ParamEstInterface_GradientSearch):
    """ Implement a simple system by extending the MixedNLGaussian class """
    def __init__(self, params):
        """ Define all model variables """
        Axi = numpy.array([[params[1], 0.0, 0.0]])
        Az = numpy.asarray(((1.0, params[2], 0.0),
                            (0.0,
                             params[3] * math.cos(params[4]),
                             - params[3] * math.sin(params[4])),
                             (0.0,
                             params[3] * math.sin(params[4]),
                             params[3] * math.cos(params[4]))))

        C = numpy.array([[0.0, 0.0, 0.0], [1.0, -1.0, 1.0]])
        Qxi = numpy.diag([ 0.01, ])
        Qz = numpy.diag([ 0.01, 0.01, 0.01])
        R = numpy.diag([0.1, 0.1])
        xi0 = numpy.asarray((0.0,)).reshape((-1, 1))
        Pxi0 = numpy.eye(1)
        z0 = numpy.zeros((3,))
        Pz0 = 0.0 * numpy.eye(3)

        # Linear states handled by base-class
        super(ParticleLS2, self).__init__(xi0=xi0, z0=z0, Pz0=Pz0, Pxi0=Pxi0,
                                         Az=Az, C=C, Axi=Axi,
                                         R=R, Qxi=Qxi, Qz=Qz,
                                         params=params)


    def get_nonlin_pred_dynamics(self, particles, u, t):
        xil = particles[:, 0]
        fxil = self.params[0] * numpy.arctan(xil)
        return (None, fxil[:, numpy.newaxis, numpy.newaxis], None)

    def get_meas_dynamics(self, particles, y, t):
        N = len(particles)
        xil = numpy.vstack(particles)[:, 0]
        h = numpy.zeros((N, 2, 1))
        h[:, 0, 0] = 0.1 * numpy.fabs(xil) * xil
        return (numpy.asarray(y).reshape((-1, 1)), None, h, None)

    # Override this method since there is no uncertainty in z0
    def eval_logp_x0(self, particles, t):
        return self.eval_logp_xi0(particles[:, :self.lxi])

    def eval_logp_x0_val_grad(self, particles, t):
        return (self.eval_logp_xi0(particles[:, :self.lxi]),
                self.eval_logp_xi0_grad(particles[:, :self.lxi]))

    def get_pred_dynamics_grad(self, particles, u, t):
        N = len(particles)
        xil = particles[:, 0]
        f_grad = numpy.zeros((N, 5, 4, 1))
        f_grad[:, 0, 0, 0] = numpy.arctan(xil)

        return (numpy.repeat(self.A_grad[numpy.newaxis], N, 0), f_grad, None)


    def set_params(self, params):
        """ New set of parameters """
        # Update all needed matrices and derivates with respect
        # to the new parameter set
        self.params = numpy.copy(params)
        Axi = numpy.array([[params[1], 0.0, 0.0]])

        Az = numpy.asarray(((1.0, params[2], 0.0),
                            (0.0,
                             params[3] * math.cos(params[4]),
                             - params[3] * math.sin(params[4])),
                             (0.0,
                             params[3] * math.sin(params[4]),
                             params[3] * math.cos(params[4]))))

        self.A_grad = numpy.vstack((
            numpy.zeros((4, 3))[numpy.newaxis],
            numpy.asarray(((1.0, 0.0, 0.0),
                           (0.0, 0.0, 0.0),
                           (0.0, 0.0, 0.0),
                           (0.0, 0.0, 0.0)))[numpy.newaxis],
            numpy.asarray(((0.0, 0.0, 0.0),
                           (0.0, 1.0, 0.0),
                           (0.0, 0.0, 0.0),
                           (0.0, 0.0, 0.0)))[numpy.newaxis],
            numpy.asarray(((0.0, 0.0, 0.0),
                           (0.0, 0.0, 0.0),
                           (0.0, math.cos(params[4]), -math.sin(params[4])),
                           (0.0, math.sin(params[4]), math.cos(params[4]))))[numpy.newaxis],
            numpy.asarray(((0.0, 0.0, 0.0),
                           (0.0, 0.0, 0.0),
                           (0.0, -params[3] * math.sin(params[4]), -params[3] * math.cos(params[4])),
                           (0.0, params[3] * math.cos(params[4]), -params[3] * math.sin(params[4]))))[numpy.newaxis]
            ))
        self.set_dynamics(Axi=Axi, Az=Az)

if __name__ == '__main__':

    if (len(sys.argv) < 2 or sys.argv[1].lower() == 'gui'):
        num = 50
        nums = 5

        theta_true = numpy.array((1.0, 1.0, 0.3, 0.968, 0.315))


        # How many steps forward in time should our simulation run
        steps = 200
        sims = 1

        # Create arrays for storing some values for later plotting
        vals = numpy.zeros((2, num + 1, steps + 1))

        estimate = numpy.zeros((5, sims))

        plt.ion()
        fig1 = plt.figure()
        fig2 = plt.figure()


        max_iter = 1000

        for k in range(sims):
            print(k)
            theta_guess = numpy.array((numpy.random.uniform(0.0, 2.0),
                                       numpy.random.uniform(0.0, 2.0),
                                       numpy.random.uniform(0.0, 0.6),
                                       numpy.random.uniform(0.0, 1.0),
                                       numpy.random.uniform(0.0, math.pi / 2.0)))

            # theta_guess = numpy.copy(theta_true)

            # Create reference
            (y, e, z) = generate_dataset(theta_true, steps)
            # Store values for last time-step aswell

            print("estimation start")

            plt.figure(fig1.number)
            plt.clf()
            x = numpy.asarray(range(steps + 1))
            plt.plot(x[1:], numpy.asarray(y)[:, :], '.')
            fig1.show()
            plt.draw()

            params_it = numpy.zeros((max_iter, len(theta_guess)))
            Q_it = numpy.zeros((max_iter))
            it = 0

            def callback(params, Q, cur_iter):
                global it
                params_it[it, :] = params
                Q_it[it] = Q
                it = it + 1
                plt.figure(fig2.number)
                plt.clf()
                plt.plot(range(it), params_it[:it, 0], 'b-')
                plt.plot((0.0, it), (theta_true[0], theta_true[0]), 'b--')
                plt.plot(range(it), params_it[:it, 1], 'r-')
                plt.plot((0.0, it), (theta_true[1], theta_true[1]), 'r--')
                plt.plot(range(it), params_it[:it, 2], 'g-')
                plt.plot((0.0, it), (theta_true[2], theta_true[2]), 'g--')
                plt.plot(range(it), params_it[:it, 3], 'c-')
                plt.plot((0.0, it), (theta_true[3], theta_true[3]), 'c--')
                plt.plot(range(it), params_it[:it, 4], 'k-')
                plt.plot((0.0, it), (theta_true[4], theta_true[4]), 'k--')
                plt.show()
                plt.draw()
                plt.pause(0.0001)
                return (cur_iter > max_iter)

            # Create an array for our particles
            model = ParticleLS2(theta_guess)
            ParamEstimator = param_est.ParamEstimation(model=model, u=None, y=y)
            ParamEstimator.set_params(theta_guess)
            # ParamEstimator.simulate(num, nums, False)

            (param, Q) = ParamEstimator.maximize(param0=theta_guess, num_part=num, num_traj=nums, max_iter=max_iter,
                                                 callback=callback, smoother='rsas', tol=0.0)

            svals = numpy.zeros((4, nums, steps + 1))

            fig3 = plt.figure()
            fig4 = plt.figure()
            fig5 = plt.figure()
            fig6 = plt.figure()


            sest = ParamEstimator.straj.get_smoothed_estimates()

            for i in range(steps + 1):
                for j in range(nums):
                    svals[0, j, i] = sest[i, j, 0]
                    svals[1:, j, i] = sest[i, j, 1:4]

            plt.figure(fig3.number)
            plt.clf()
            # TODO, does these smoothed estimates really look ok??
            for j in range(nums):
                plt.plot(range(steps + 1), svals[0, j, :], 'g-')
                # plt.plot(range(steps+1),svals[1,j,:],'r-')
            plt.plot(x[:-1], e.T, 'rx')
            # plt.plot(x[:-1], e,'r-')
            fig3.show()

            plt.figure(fig4.number)
            plt.clf()
            # TODO, does these smoothed estimates really look ok??
            for j in range(nums):
                plt.plot(range(steps + 1), svals[1, j, :], 'g-')
                # plt.plot(range(steps+1),svals[1,j,:],'r-')
            plt.plot(x[:-1], z[0, :], 'rx')
            # plt.plot(x[:-1], e,'r-')
            fig4.show()

            plt.figure(fig5.number)
            plt.clf()
            # TODO, does these smoothed estimates really look ok??
            for j in range(nums):
                plt.plot(range(steps + 1), svals[2, j, :], 'g-')
                # plt.plot(range(steps+1),svals[1,j,:],'r-')
            plt.plot(x[:-1], z[1, :], 'rx')
            # plt.plot(x[:-1], e,'r-')
            fig5.show()

            plt.figure(fig6.number)
            plt.clf()
            # TODO, does these smoothed estimates really look ok??
            for j in range(nums):
                plt.plot(range(steps + 1), svals[2, j, :], 'g-')
                # plt.plot(range(steps+1),svals[1,j,:],'r-')
            plt.plot(x[:-1], z[2, :], 'rx')
            # plt.plot(x[:-1], e,'r-')
            fig6.show()


            plt.draw()


            print("maximization start")

            estimate[:, k] = param

    #        plt.figure(fig2.number)
    #        plt.clf()
    #        plt.hist(estimate[:,:(k+1)].T)
    #        fig2.show()
    #        plt.show()
    #        plt.draw()




    #    plt.hist(estimate.T)
        plt.ioff()
        plt.show()
        plt.draw()
        print("exit")
    elif (sys.argv[1].lower() == 'nogui'):
        num = 50
        nums = 10

        theta_true = numpy.array((1.0, 1.0, 0.3, 0.968, 0.315))


        # How many steps forward in time should our simulation run
        steps = 200
        sims = 20

        # Create arrays for storing some values for later plotting
        vals = numpy.zeros((2, num + 1, steps + 1))

        estimate = numpy.zeros((5, sims))

        max_iter = 1000

        for k in range(sims):
            theta_guess = numpy.array((numpy.random.uniform(0.0, 2.0),
                                       numpy.random.uniform(0.0, 2.0),
                                       numpy.random.uniform(0.0, 0.6),
                                       numpy.random.uniform(0.0, 1.0),
                                       numpy.random.uniform(0.0, math.pi / 2.0)))

            # Create reference
            # numpy.random.seed(k)
            (y, e, z) = generate_dataset(theta_true, steps)

            # Create an array for our particles
            model = ParticleLS2(theta_guess)
            ParamEstimator = param_est.ParamEstimation(model=model, u=None, y=y)
            ParamEstimator.set_params(theta_guess)
            # ParamEstimator.simulate(num, nums, False)

            (param, Q) = ParamEstimator.maximize(param0=theta_guess, num_part=num, num_traj=nums, max_iter=max_iter,
                                                 callback=None, smoother='rsas', tol=0.0)

            print("{} {} {} {} {}") % tuple(round(param, 4))

        print("exit")
    elif (sys.argv[1].lower() == 'gradient'):
        num = 50
        nums = 5
        numpy.random.seed(4) # 3
        theta_true = numpy.array((1.0, 1.0, 0.3, 0.968, 0.315))

        # How many steps forward in time should our simulation run
        steps = 50
        sims = 1

        # Create arrays for storing some values for later plotting
        vals = numpy.zeros((2, num + 1, steps + 1))

        estimate = numpy.zeros((5, sims))

        (y, e, z) = generate_dataset(theta_true, steps)

        model = ParticleLS2(theta_true)
        # Create an array for our particles
        gt = gradienttest.GradientTest(model=model, u=None, y=y)
        gt.set_params(theta_true)

        param_id = 4
        param_steps = 101
        tval = theta_true[param_id]
        param_vals = numpy.linspace(tval - math.fabs(tval), tval + math.fabs(tval), param_steps)
        gt.test(param_id, param_vals, nums=nums)

        gt.plot_y.plot(1)
        gt.plot_xn.plot(2)
        gt.plot_x0.plot(3)
        plt.show()
    else:
        print('Unsupported option')


