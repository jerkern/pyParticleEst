import numpy
import math
import  pyparticleest.models.nlg as nlg
import pyparticleest.simulator as simulator
import matplotlib.pyplot as plt

def generate_dataset(steps, P0, Q, R):
    x = numpy.zeros((steps + 1,))
    y = numpy.zeros((steps + 1,))
    x[0] = numpy.random.multivariate_normal((0.0,), P0)
    y[0] = 0.05 * x[0] ** 2 + numpy.random.multivariate_normal((0.0,), R)
    for k in range(0, steps):
        x[k + 1] = 0.5 * x[k] + 25.0 * x[k] / (1 + x[k] ** 2) + 8 * math.cos(1.2 * k) + numpy.random.multivariate_normal((0.0,), Q)
        y[k + 1] = 0.05 * x[k + 1] ** 2 + numpy.random.multivariate_normal((0.0,), R)

    return (x, y)

def wmean(logw, val):
    w = numpy.exp(logw)
    w = w / sum(w)
    return numpy.sum(w * val.ravel())

class Model(nlg.NonlinearGaussianInitialGaussian):
    """ x_{k+1} = 0.5*x_k + 25.0*x_k/(1+x_k**2) + 8*math.cos(1.2*k) + v_k, v_k ~ N(0,Q)
        y_k = 0.05*x_k**2 + e_k, e_k ~ N(0,R),
        x(0) ~ N(0,P0) """

    def __init__(self, P0, Q, R):
        super(Model, self).__init__(Px0=P0, Q=Q, R=R)

    def calc_g(self, particles, t):
        return 0.05 * particles ** 2

    def calc_f(self, particles, u, t):
        return 0.5 * particles + 25.0 * particles / (1 + particles ** 2) + 8 * math.cos(1.2 * t)


if __name__ == '__main__':

    T = 80
    P0 = 5.0 * numpy.eye(1)
    Q = 1.0 * numpy.eye(1)
    R = 0.1 * numpy.eye(1)


    N = 100
    M = 10
    # If using many iterations to calculate average RMSE disable the plotting code
    iterations = 10 # 1000

    plt.ion()

    model = Model(P0, Q, R)
    rmse_filt = 0.0
    rmse_smooth = 0.0
    rmse2_filt = 0.0
    rmse2_smooth = 0.0
    for it in xrange(iterations):
        numpy.random.seed(it)
        (x, y) = generate_dataset(T, P0, Q, R)
        sim = simulator.Simulator(model, u=None, y=y)
        plt.clf()
        plt.plot(range(T + 1), x, 'r-', linewidth=2.0, label='True')
        sim.simulate(N, M, filter='PF', smoother='rsas', meas_first=True)

        (est_filt, w_filt) = sim.get_filtered_estimates()
        mean_filt = sim.get_filtered_mean()

        err = mean_filt.ravel() - x
        rmse_filt += numpy.sqrt(numpy.mean(err ** 2))

        tmp = 0.0
        plt.plot((0,) * N, est_filt[0, :, 0].ravel(), 'k.', markersize=0.5, label='Particles')
        for t in xrange(1, T + 1):
            plt.plot((t,) * N, est_filt[t, :, 0].ravel(), 'k.', markersize=0.5)
            tmp += numpy.sum(w_filt[t] * (est_filt[t, :, 0] - x[t]) ** 2)
        rmse2_filt += numpy.sqrt(tmp / T)

        if (M > 0):
            est_smooth = sim.get_smoothed_estimates()

            mean_smooth = sim.get_smoothed_mean()
            err = mean_smooth.ravel() - x
            rmse_smooth += numpy.sqrt(numpy.mean(err ** 2))

            tmp = 0.0
            for k in xrange(M):
                tmp += 1.0 / M * numpy.sum((est_smooth[:, k, 0].ravel() - x) ** 2)
            rmse2_smooth += numpy.sqrt(tmp / T)
        plt.ioff()
        plt.plot(range(T + 1), mean_filt[:, 0], 'g--', linewidth=2.0, label='Filter mean')
        plt.plot(range(T + 1), mean_smooth[:, 0], 'b--', linewidth=2.0, label='Smoother mean')
        plt.legend(loc=4, fontsize=24)
        plt.draw()
        plt.show()

    print "rmse filter = %f" % (rmse_filt / iterations)
    print "rmse smooth = %f" % (rmse_smooth / iterations)
    print "rmse2 filter = %f" % (rmse2_filt / iterations)
    print "rmse2 smooth = %f" % (rmse2_smooth / iterations)

