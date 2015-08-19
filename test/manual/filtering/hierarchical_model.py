import numpy
import math
from pyparticleest.models.hierarchial import HierarchicalRSBase
import pyparticleest.simulator
import pyparticleest.utils.kalman as kalman
import matplotlib.pyplot as plt
import scipy.stats

def generate_dataset(steps, P0_xi, P0_z, Q_xi, Q_z, R_xi, R_z):
    xi = numpy.zeros((1, steps + 1))
    z = numpy.zeros((2, steps + 1))
    y = numpy.zeros((steps, 2))
    xi[:, 0] = numpy.random.normal(0.0, math.sqrt(P0_xi))
    z[:, 0] = numpy.random.multivariate_normal(numpy.zeros((2,)), P0_z)
    for k in range(1, steps + 1):
        xi[:, k] = xi[:, k - 1] + numpy.random.normal(0.0, Q_xi)
        Ak = numpy.asarray(((math.cos(xi[:, k - 1]), math.sin(xi[:, k - 1])),
                            (-math.sin(xi[:, k - 1]), math.cos(xi[:, k - 1]))
                            ))
        z[:, k] = Ak.dot(z[:, k - 1]) + numpy.random.multivariate_normal(numpy.zeros((2,)), Q_z)
        C = numpy.asarray(((math.cos(xi[:, k - 1]), math.sin(xi[:, k - 1]))))
        y[k - 1, 0] = xi[:, k] + numpy.random.normal(0.0, math.sqrt(R_xi))
        y[k - 1, 1] = C.dot(z[:, k]) + numpy.random.normal(0.0, math.sqrt(R_z))

    x = numpy.vstack((xi, z))
    return (x, y)


class Model(HierarchicalRSBase):
    """ xi_{k+1} = xi_k + v_xi_k, v_xi_k ~ N(0,Q_xi)
        z_{k+1} = ((cos(xi_k) sin(xi_k)),
                   (-sin(xi_k) cos(xi_k)) * z_{k} + v_z_k, v_z_k ~ N(0, Q_z) 
        y_z_k = ((cos(xi_k) sin(xi_k))*z_k + e_k, e_k ~ N(0,R_z),
        y_xi_k = xi_k + e_xi_k, e_xi_k ~ N(0,R_xi)
        x(0) ~ N(0,P0) """

    def __init__(self, P0_xi, P0_z, Q_xi, Q_z, R_xi, R_z):
        self.P0_xi = numpy.copy(P0_xi)
        self.Q_xi = numpy.copy(Q_xi)
        self.R_xi = numpy.copy(R_xi)
        self.P0_z = numpy.copy(P0_z)
        self.R_z = numpy.copy(R_z)
        fz = numpy.zeros((2, 1))
        hz = numpy.zeros((1, 1))
        self.pn_count = 0
        super(Model, self).__init__(len_xi=1, len_z=2, fz=fz, Qz=Q_z, hz=hz, R=R_z)

    def create_initial_estimate(self, N):
        particles = numpy.zeros((N, self.lxi + self.kf.lz + 2 * self.kf.lz ** 2))

        for i in xrange(N):
            particles[i, 0] = numpy.random.normal(0.0, math.sqrt(self.P0_xi))
            particles[i, 1:3] = numpy.zeros((1, 2))
            particles[i, 3:7] = numpy.copy(self.P0_z).ravel()
        return particles

    def get_rb_initial(self, xi0):
        N = len(xi0)
        z0 = numpy.zeros((self.kf.lz, 1))
        z_list = numpy.repeat(z0.reshape((1, self.kf.lz, 1)), N, 0)
        P_list = numpy.repeat(self.P0_z.reshape((1, self.kf.lz, self.kf.lz)), N, 0)
        return (z_list, P_list)

    def sample_process_noise(self, particles, u, t):
        """ Return process noise for input u """
        N = len(particles)
        return numpy.random.normal(0.0, math.sqrt(self.Q_xi), (N,))

    def calc_xi_next(self, particles, u, t, noise):
        N = len(particles)
        xi_next = numpy.empty(N)
        for i in xrange(N):
            xi_next[i] = particles[i][0] + noise[i]
        return xi_next

    def logp_xnext_xi(self, particles, next_xi, u, t):
        self.pn_count = self.pn_count + len(particles)
        xi = particles[:, :self.lxi]
        return scipy.stats.norm.logpdf((next_xi - xi).ravel(), 0.0, math.sqrt(self.Q_xi))

    def logp_xnext_xi_max(self, particles, u, t):
        return numpy.asarray((scipy.stats.norm.logpdf(0.0, 0.0, math.sqrt(self.Q_xi)),) * len(particles))

    def measure_nonlin(self, particles, y, t):
        N = len(particles)
        lpy = numpy.empty((N,))
        for i in xrange(N):
            lpy[i] = kalman.lognormpdf(y[0] - particles[i][0], self.R_xi)
        return lpy

    def get_lin_pred_dynamics(self, particles, u, t):
        """ Return matrices describing affine relation of next
            nonlinear state conditioned on current nonlinear state

            \z_{t+1]} = A_z * z_t + f_z + v_z, v_z ~ N(0,Q_z)

            conditioned on the value of xi_{t+1}. 
            (Not the same as the dynamics unconditioned on xi_{t+1})
            when for example there is a noise correlation between the 
            linear and nonlinear state dynamics) 
            """
        N = len(particles)
        Az = numpy.empty((N, 2, 2))
        for i in xrange(N):
            Az[i] = numpy.asarray(((math.cos(particles[i][0]), math.sin(particles[i][0])),
                                  (-math.sin(particles[i][0]), math.cos(particles[i][0])))
                                  )
        return (Az, None, None)

    def get_lin_meas_dynamics(self, particles, y, t):
        N = len(particles)
        Cz = numpy.empty((N, 1, 2))
        for i in xrange(N):
            Cz[i] = numpy.asarray(((math.cos(particles[i][0]), math.sin(particles[i][0])),))

        return (y[1], Cz, None, None)



if __name__ == '__main__':
    steps = 100
    num = 50
    nums = 10
    P0_xi = 1.0
    P0_z = numpy.eye(2)
    Q_xi = 0.01 * 1.0
    Q_z = 0.01 * numpy.eye(2)
    R_xi = 0.1 * numpy.eye(1)
    R_z = 0.1 * numpy.eye(1)
    (x, y) = generate_dataset(steps, P0_xi, P0_z, Q_xi, Q_z, R_xi, R_z)

    model = Model(P0_xi, P0_z, Q_xi, Q_z, R_xi, R_z)
    sim = pyparticleest.simulator.Simulator(model, u=None, y=y)
    sim.simulate(num, nums, smoother='rsas')

    plt.plot(range(steps + 1), x[0, :], 'r-')
    plt.plot(range(steps + 1), x[1, :], 'g-')
    plt.plot(range(steps + 1), x[2, :], 'b-')
    # plt.plot(range(1,steps+1), y, 'bx')

    xi_vals = numpy.empty((num, steps + 1))

    (parts, _) = sim.get_filtered_estimates()

    sest = sim.get_smoothed_estimates()

    for k in range(steps + 1):
        plt.plot((k,) * num, parts[k, :, 0], 'r.', markersize=1.0)

    for j in xrange(nums):
        plt.plot(range(steps + 1), sest[:, j, 0], 'r--')
        plt.plot(range(steps + 1), sest[:, j, 1], 'g--')
        plt.plot(range(steps + 1), sest[:, j, 2], 'b--')

    plt.show()
