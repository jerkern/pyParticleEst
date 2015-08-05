""" Particle filtering for a trivial model
    Also illustrates that the """

import numpy
import pyparticleest.utils.kalman as kalman
import pyparticleest.interfaces as interfaces
import matplotlib.pyplot as plt
import pyparticleest.simulator as simulator

def generate_dataset(steps, P0, Q, R):
    x = numpy.zeros((steps + 1,))
    y = numpy.zeros((steps,))
    x[0] = 2.0 + 0.0 * numpy.random.normal(0.0, P0)
    for k in range(1, steps + 1):
        x[k] = x[k - 1] + numpy.random.normal(0.0, Q)
        y[k - 1] = x[k] + numpy.random.normal(0.0, R)

    return (x, y)

class Model(interfaces.FFProposeFromMeasure,
            interfaces.FFBSi):
    """ x_{k+1} = x_k + v_k, v_k ~ N(0,Q)
        y_k = x_k + e_k, e_k ~ N(0,R),
        x(0) ~ N(0,P0) """

    def __init__(self, P0, Q, R):
        self.P0 = numpy.copy(P0)
        self.Q = numpy.copy(Q)
        self.R = numpy.copy(R)

    def propose_from_y(self, N, y, t):
        return numpy.random.normal(y, numpy.sqrt(self.R), (N,)).reshape((-1, 1))

    def create_initial_estimate(self, N):
        return numpy.random.normal(0.0, numpy.sqrt(self.P0), (N,)).reshape((-1, 1))

    def logp_xnext(self, particles, next_part, u, t):
        diff = next_part - particles
        return kalman.lognormpdf_scalar(diff, self.Q)
#
#    def sample_process_noise(self, particles, u, t):
#        """ Return process noise for input u """
#        N = len(particles)
#        return numpy.random.normal(0.0, self.Q, (N,)).reshape((-1, 1))
#
#    def update(self, particles, u, t, noise):
#        """ Update estimate using 'data' as input """
#        particles += noise
#
#    def measure(self, particles, y, t):
#        """ Return the log-pdf value of the measurement """
#        logyprob = numpy.empty(len(particles), dtype=float)
#        for k in range(len(particles)):
#            logyprob[k] = kalman.lognormpdf(particles[k].reshape(-1, 1) - y, self.R)
#        return logyprob
#

    def sample_smooth(self, part, ptraj, anc, future_trajs, find, ut, yt, tt, cur_ind):
        return numpy.copy(part)

    def copy_ind(self, particles, new_ind=None):

        if (new_ind != None):
            return numpy.copy(particles[new_ind])
        else:
            return numpy.copy(particles)

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
    sim.simulate(num, num, filter='pfy', smoother='ancestor')

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
