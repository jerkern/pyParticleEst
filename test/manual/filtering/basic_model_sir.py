""" Particle filtering for a trivial model
    Also illustrates that the """

import numpy
import pyparticleest.utils.kalman as kalman
import pyparticleest.interfaces as interfaces
import matplotlib.pyplot as plt
import pyparticleest.simulator as simulator
import scipy.linalg

from builtins import range


def generate_dataset(steps, P0, Q, R):
    x = numpy.zeros((steps + 1,))
    y = numpy.zeros((steps,))
    x[0] = 2.0 + 0.0 * numpy.random.normal(0.0, P0)
    for k in range(1, steps + 1):
        x[k] = x[k - 1] + numpy.random.normal(0.0, Q)
        y[k - 1] = x[k] + numpy.random.normal(0.0, R)

    return (x, y)


class Model(interfaces.SIR):
    """ x_{k+1} = x_k + v_k, v_k ~ N(0,Q)
        y_k = x_k + e_k, e_k ~ N(0,R),
        x(0) ~ N(0,P0) """

    def __init__(self, P0, Q, R):
        self.P0 = numpy.copy(P0)
        self.Q = numpy.copy(Q)
        self.R = numpy.copy(R)

    def create_initial_estimate(self, N):
        return numpy.random.normal(0.0, self.P0, (N,)).reshape((-1, 1))

    def qsample(self, particles, u, y, t):
        pnext = numpy.empty_like(particles)
        err = y - particles
        C = numpy.eye(1)
        P = self.Q
        S = C.dot(P).dot(C.T) + self.R
        Pn = P - P.dot(C.T).dot(scipy.linalg.solve(S, C.dot(P)))
        for i in range(len(pnext)):
            m = particles[i] + P.dot(C.T).dot(scipy.linalg.solve(S,
                                                                 err[i].reshape((-1, 1)))).ravel()
            pnext[i] = numpy.random.multivariate_normal(m, Pn).ravel()

        return pnext

    def logp_q(self, particles, next_part, u, y, t):
        logpq = numpy.empty(len(particles), dtype=float)
        err = y - particles
        C = numpy.eye(1)
        P = self.Q
        S = C.dot(P).dot(C.T) + self.R
        Pn = P - P.dot(C.T).dot(scipy.linalg.solve(S, C.dot(P)))
        for i in range(len(logpq)):
            m = particles[i] + P.dot(C.T).dot(scipy.linalg.solve(S,
                                                                 err[i].reshape((-1, 1)))).ravel()
            logpq[i] = kalman.lognormpdf(
                m.reshape((-1, 1)) - next_part[i].reshape((-1, 1)), Pn).ravel()

        return logpq

    def logp_xnext(self, particles, next_part, u, t):
        logpxn = numpy.empty(len(particles), dtype=float)
        for k in range(len(particles)):
            logpxn[k] = kalman.lognormpdf(
                particles[k].reshape(-1, 1) - next_part[k].reshape(-1, 1), self.Q)
        return logpxn

    def measure(self, particles, y, t):
        """ Return the log-pdf value of the measurement """
        logyprob = numpy.empty(len(particles), dtype=float)
        for k in range(len(particles)):
            logyprob[k] = kalman.lognormpdf(
                particles[k].reshape(-1, 1) - y, self.R)
        return logyprob


if __name__ == '__main__':
    steps = 50
    num = 50
    P0 = 1.0
    Q = numpy.asarray(((1.0,),))
    R = numpy.asarray(((1.0,),))

    # Make realization deterministic
    numpy.random.seed(1)
    (x, y) = generate_dataset(steps, P0, Q, R)

    model = Model(P0, Q, R)
    sim = simulator.Simulator(model, u=None, y=y)
    sim.simulate(num, num, filter='sir', smoother='ancestor')

    plt.plot(range(steps + 1), x, 'r-')
    plt.plot(range(1, steps + 1), y, 'bx')

    (vals, _) = sim.get_filtered_estimates()

    plt.plot(range(steps + 1), vals[:, :, 0], 'k.', markersize=0.8)

    svals = sim.get_smoothed_estimates()

    # Plot "smoothed" trajectories to illustrate that the particle filter
    # suffers from degeneracy when considering the full trajectories
    plt.plot(range(steps + 1), svals[:, :, 0], 'b--')
    plt.plot(range(steps + 1), x, 'r-')
    plt.xlabel('t')
    plt.ylabel('x')

    plt.show()
