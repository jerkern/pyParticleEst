import numpy
import math
import pyparticleest.utils.kalman as kalman
import pyparticleest.interfaces as interfaces
import pyparticleest.paramest.paramest as param_est
import pyparticleest.paramest.interfaces as pestint
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

class Model(interfaces.ParticleFiltering, interfaces.FFBSiRS, pestint.ParamEstInterface):
    """ x_{k+1} = x_k + v_k, v_k ~ N(0,Q)
        y_k = x_k + e_k, e_k ~ N(0,R),
        x(0) ~ N(0,P0) """

    def __init__(self, P0, Q, R):
        self.P0 = numpy.copy(P0)
        self.Q = numpy.copy(Q)
        self.R = numpy.copy(R)
        self.logxn_max = kalman.lognormpdf_scalar(numpy.zeros((1,)), self.Q)

    def create_initial_estimate(self, N):
        return numpy.random.normal(0.0, numpy.sqrt(self.P0), (N,))

    def sample_process_noise(self, particles, u, t):
        """ Return process noise for input u """
        N = len(particles)
        return numpy.random.normal(0.0, numpy.sqrt(self.Q), (N,))

    def update(self, particles, u, noise, t):
        """ Update estimate using 'data' as input """
        particles[:] = 0.5 * particles + 25.0 * particles / (1 + particles ** 2) + 8 * math.cos(1.2 * t) + noise

    def measure(self, particles, y, t):
        """ Return the log-pdf value of the measurement """
        return kalman.lognormpdf_scalar(0.05 * particles ** 2 - y, self.R)

    def logp_xnext(self, particles, next_part, u, t):
        """ Return the log-pdf value for the possible future state 'next' given input u """
        pn = 0.5 * particles + 25.0 * particles / (1 + particles ** 2) + 8 * math.cos(1.2 * t)
        return kalman.lognormpdf_scalar(pn.ravel() - next_part.ravel(), self.Q)

    def logp_xnext_max(self, particles, u, t):
        return self.logxn_max

    def sample_smooth(self, particles, future_trajs, ut, yt, tt):
        """ Update ev. Rao-Blackwellized states conditioned on "next_part" """
        return particles.reshape((-1, 1))

    def set_params(self, params):
        """ New set of parameters for which the integral approximation terms will be evaluated"""
        self.Q = params[0] * numpy.eye(1)
        #self.R = params[1] * numpy.eye(1)

    def eval_logp_x0(self, particles, t):
        """ Calculate gradient of a term of the I1 integral approximation
            as specified in [1].
            The gradient is an array where each element is the derivative with
            respect to the corresponding parameter"""
        return kalman.lognormpdf_scalar(particles, self.P0)

    def copy_ind(self, particles, new_ind=None):
        if (new_ind != None):
            return numpy.copy(particles[new_ind])
        else:
            return numpy.copy(particles)

    def eval_logp_xnext_fulltraj(self, straj, ut, tt):
        M = straj.traj.shape[1]
        part = straj.traj
        cost = 8.0 * numpy.cos(1.2 * numpy.asarray(tt, dtype=float))
        xp = 0.5 * part + 25.0 * part / (1 + part ** 2) + numpy.repeat(cost.reshape(-1, 1, 1), repeats=M, axis=1)
        diff = part[1:] - xp[:-1]
        logp = kalman.lognormpdf_scalar(diff.ravel(), self.Q)
        return numpy.sum(logp) / M


    def eval_logp_y_fulltraj(self, straj, yt, tt):
        M = straj.traj.shape[1]
        yp = 0.05 * straj.traj ** 2
        diff = yp - numpy.repeat(numpy.asarray(yt, dtype=float).reshape((-1, 1, 1)),
                                 repeats=M, axis=1)
        return numpy.sum(kalman.lognormpdf_scalar(diff.ravel(), self.R)) / M

    def maximize_weighted(self, straj, alltrajs, weights):
        M = alltrajs.shape[1]

        tt = straj.t
        yt = straj.y
        part = alltrajs

        cost = 8.0 * numpy.cos(1.2 * numpy.asarray(tt, dtype=float))
        xp = 0.5 * part + 25.0 * part / (1 + part ** 2) + numpy.repeat(cost.reshape(-1, 1, 1), repeats=M, axis=1)
        diff = part[1:] - xp[:-1]

        werr = numpy.empty((len(alltrajs) - 1, M))

        for j in xrange(len(weights)):
            werr[:, j:j + 1] = weights[j] * diff[:, j] ** 2

        Q = numpy.mean(numpy.sum(werr, axis=1))

        yp = 0.05 * alltrajs ** 2
        diff = yp - numpy.repeat(numpy.asarray(yt, dtype=float).reshape((-1, 1, 1)),
                                 repeats=M, axis=1)

        werr = numpy.empty((len(alltrajs), M))

        for j in xrange(len(weights)):
            werr[:, j:j + 1] = weights[j] * diff[:, j] ** 2

        R = numpy.mean(numpy.sum(werr, axis=1))

        #newparams = numpy.asarray((Q, R))
        newparams = numpy.asarray((Q,))
        return newparams

def callback(params, Q):
    print "params = %s" % params

if __name__ == '__main__':
    numpy.random.seed(1)
    steps = 49 #1499

    num = 15
    P0 = 5.0 * numpy.eye(1)
    Q = 1.0 * numpy.eye(1)
    R = 0.1 * numpy.eye(1)
    (x, y) = generate_dataset(steps, P0, Q, R)

    t = numpy.arange(steps + 1)
    def callback_sim(estimator):

        plt.figure(1)
        plt.clf()
        sest = estimator.get_smoothed_estimates()
        for k in range(sest.shape[1]):
            plt.plot(range(steps + 1), sest[:, k], 'g-')
            plt.plot(range(steps + 1), 0.05 * sest[:, k] ** 2, 'gx')

        plt.plot(range(steps + 1), x, 'r-')
        plt.plot(range(steps + 1), y, 'bx')
        plt.draw()
        plt.show()

    #theta0 = numpy.asarray((2.0, 2.0))
    theta0 = numpy.asarray((2.0,))
    model = Model(P0, Q, R)
    estimator = param_est.ParamEstimationSAEM(model, u=None, y=y)
    callback(theta0, None)
    filter_options = {'cond_traj': numpy.zeros((steps + 1, 1, 1))}
    plt.ion()
    estimator.maximize(theta0, num, filter='cpfas', smoother='ancestor',
                       meas_first=True, max_iter=2000,
                       filter_options=filter_options,
                       callback=callback,
                       callback_sim=callback_sim)
    plt.ioff()
#     plt.ion()
#     estimator.maximize(theta0, num, M, smoother='full', meas_first=True, max_iter=len(iterations),
#                        callback_sim=callback_sim, callback=callback)
#     plt.ioff()
#    traj = pf.ParticleTrajectory(model, num)
#    traj.measure(y[0])
#    for k in range(1,len(y)):
#        traj.forward(u=None, y=y[k])
#
#    straj = traj.perform_smoothing(M, method='rs')


