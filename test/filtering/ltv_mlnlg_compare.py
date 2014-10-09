import numpy
from pyparticleest.models.mlnlg import MixedNLGaussianInitialGaussianProperBSi
from pyparticleest.models.ltv import LTV
import pyparticleest.filter as pf
import matplotlib.pyplot as plt

# R = numpy.array([[0.01]])
# Q = numpy.diag([ 0.01, 0.1])
R = numpy.array([[1.0]])
Q = numpy.diag([ 1.0, 1.0])
gradient_test = False
xi0_true = numpy.array([0.0, ])
z0_true = numpy.array([1.0, ])

def generate_reference(z0, P0, theta_true, steps):
    A = numpy.asarray(((1.0, theta_true), (0.0, 1.0)))
    C = numpy.array([[1.0, 0.0]])

    states = numpy.zeros((steps + 1, 2))
    y = numpy.zeros((steps + 1, 1))
    x0 = numpy.random.multivariate_normal(z0.ravel(), P0)
    states[0] = numpy.copy(x0)
    y[0] = C.dot(x0.reshape((-1, 1))).ravel() + numpy.random.multivariate_normal((0.0,), R).ravel()
    for i in range(steps):

        # Calc linear states
        x = states[i].reshape((-1, 1))
        xn = A.dot(x) + numpy.random.multivariate_normal(numpy.zeros(x0.shape), Q).reshape((-1, 1))
        states[i + 1] = xn.ravel()

        # use the correct particle to generate the true measurement
        y[i + 1] = C.dot(xn).ravel() + numpy.random.multivariate_normal((0.0,), R).ravel()

    return (y, states)

def wmean(logw, val):
    w = numpy.exp(logw)
    w = w / sum(w)
    return numpy.sum(w * val.ravel())

def wcov(logw, val):
    u = wmean(logw, val)
    w = numpy.exp(logw)
    w = w / numpy.sum(w)
    err = val - u
    cov = sum(w) / (sum(w) ** 2 - sum(w ** 2)) * sum(w * err * err)
    return cov

class ParticleLTV(LTV):
    """ Implement a simple system by extending the MixedNLGaussian class """
    def __init__(self, z0, P0, params):
        """ Define all model variables """
        self.params = numpy.copy(params)
        A = numpy.array([[1.0, params[0]], [0.0, 1.0]])
        C = numpy.array([[1.0, 0.0]])
        z0 = numpy.copy(z0).reshape((-1, 1))
        # Linear states handled by base-class
        super(ParticleLTV, self).__init__(z0=z0, P0=P0, A=A,
                                                C=C, R=R, Q=Q,)

class ParticleMLNLG(MixedNLGaussianInitialGaussianProperBSi):
    """ Implement a simple system by extending the MixedNLGaussian class """
    def __init__(self, params, R, Qxi, Qz):
        """ Define all model variables """
        C = numpy.array([[0.0, ]])
        self.params = numpy.copy(params)
        Axi = params[0] * numpy.eye(1.0)
        Az = numpy.eye(1.0)
        z0 = numpy.copy(z0_true)
        xi0 = numpy.copy(xi0_true)
        Pz0 = numpy.eye(1)
        Pxi0 = numpy.eye(1)
        # Linear states handled by base-class
        super(ParticleMLNLG, self).__init__(Az=Az, C=C, Axi=Axi,
                                            R=R, Qxi=Qxi, Qz=Qz,
                                            z0=z0, xi0=xi0,
                                            Pz0=Pz0, Pxi0=Pxi0)

    def get_nonlin_pred_dynamics(self, particles, u, t):
        xil = numpy.vstack(particles)[:, 0]
        fxil = xil[:, numpy.newaxis, numpy.newaxis]
        return (None, fxil, None)

    def get_meas_dynamics(self, y, particles, t):
        xil = numpy.vstack(particles)[:, 0]
        h = xil[:, numpy.newaxis, numpy.newaxis]
        return (numpy.asarray(y).reshape((-1, 1)), None, h, None)


if __name__ == '__main__':
    num = 50
    nums = 5
    theta_true = 0.1
    Qxi = Q[0, 0].reshape((1, 1))
    Qz = Q[1, 1].reshape((1, 1))
    P0 = numpy.eye(2)
    steps = 50
    x0 = numpy.vstack((xi0_true, z0_true))
    # numpy.random.seed(1)
    (y, x) = generate_reference(x0, P0, theta_true, steps)


    model_mlnlg = ParticleMLNLG((theta_true,), R=R, Qxi=Qxi, Qz=Qz)

    model_ltv = ParticleLTV(z0=x0, P0=P0, params=(theta_true,))

    # Filtering
    traj_ltv = pf.ParticleTrajectory(model_ltv, 1)
    traj_ltv.measure(y[0])
    for k in range(1, len(y)):
        traj_ltv.forward(u=None, y=y[k])

    traj_mlnlg = pf.ParticleTrajectory(model_mlnlg, num)
    traj_mlnlg.measure(y[0])
    for k in range(1, len(y)):
        traj_mlnlg.forward(u=None, y=y[k])

    plt.ion()
    plt.figure(1)
    plt.clf()
    plt.plot(range(steps + 1), x[:, 0], 'r-')
    plt.plot(range(steps + 1), x[:, 1], 'b-')

    plt.plot(range(steps + 1), x[:, 0], 'r-')
    plt.plot(range(steps + 1), x[:, 1], 'b-')

    avg1 = numpy.zeros(steps + 1)
    avg2 = numpy.zeros(steps + 1)
    cov1 = numpy.zeros(steps + 1)
    cov2 = numpy.zeros(steps + 1)

    for t in range(steps + 1):
        avg1[t] = wmean(traj_ltv.traj[t].pa.w, numpy.vstack(traj_ltv.traj[t].pa.part)[:, 0])
        avg2[t] = wmean(traj_ltv.traj[t].pa.w, numpy.vstack(traj_ltv.traj[t].pa.part)[:, 1])
        cov1[t] = wmean(traj_ltv.traj[t].pa.w, numpy.vstack(traj_ltv.traj[t].pa.part)[:, 2])
        cov2[t] = wmean(traj_ltv.traj[t].pa.w, numpy.vstack(traj_ltv.traj[t].pa.part)[:, 5])

    plt.plot(range(steps + 1), avg1, 'r--')
    plt.plot(range(steps + 1), avg2, 'b--')
    plt.plot(range(steps + 1), avg1 - numpy.sqrt(cov1), 'r-.')
    plt.plot(range(steps + 1), avg1 + numpy.sqrt(cov1), 'r-.')
    plt.plot(range(steps + 1), avg2 - numpy.sqrt(cov2), 'b-.')
    plt.plot(range(steps + 1), avg2 + numpy.sqrt(cov2), 'b-.')
    # print "filtering ltv: (%s/%s) (%s/%s)" % (avg1, cov1, avg2, cov2)
#
#    for j in xrange(nums):
#        plt.plot(range(steps+1), pe_mlnlg.straj.traj[:,j,0], 'g--')

    avg1 = numpy.zeros(steps + 1)
    avg2 = numpy.zeros(steps + 1)
    cov1 = numpy.zeros(steps + 1)
    cov2 = numpy.zeros(steps + 1)

    for t in range(steps + 1):
        avg1[t] = wmean(traj_mlnlg.traj[t].pa.w, traj_mlnlg.traj[t].pa.part[:, 0])
        avg2[t] = wmean(traj_mlnlg.traj[t].pa.w, traj_mlnlg.traj[t].pa.part[:, 1])
        cov1[t] = wcov(traj_mlnlg.traj[t].pa.w, traj_mlnlg.traj[t].pa.part[:, 0])
        cov2[t] = wmean(traj_mlnlg.traj[t].pa.w, traj_mlnlg.traj[t].pa.part[:, 2])
    plt.plot(range(steps + 1), avg1, 'g--')
    plt.plot(range(steps + 1), avg1 + numpy.sqrt(cov1), 'g-.')
    plt.plot(range(steps + 1), avg1 - numpy.sqrt(cov1), 'g-.')
    plt.plot(range(steps + 1), avg2, 'k--')
    plt.plot(range(steps + 1), avg2 + numpy.sqrt(cov2), 'k-.')
    plt.plot(range(steps + 1), avg2 - numpy.sqrt(cov2), 'k-.')
    # print "filtering mlnlg: (%s/%s) (%s/%s)" % (avg1, cov1, avg2, cov2)
    plt.show()
    plt.draw()

    # Smoothing
    straj_ltv = traj_ltv.perform_smoothing(1)
    straj_mlnlg = traj_mlnlg.perform_smoothing(M=nums, method='full')

    plt.figure(2)
    plt.clf()
    plt.plot(range(steps + 1), x[:, 0], 'r-')
    plt.plot(range(steps + 1), x[:, 1], 'b-')
    avg1 = straj_ltv.traj[:, 0, 0]
    avg2 = straj_ltv.traj[:, 0, 1]
    cov1 = straj_ltv.traj[:, 0, 2]
    cov2 = straj_ltv.traj[:, 0, 5]
    plt.plot(range(steps + 1), x[:, 0], 'r-')
    plt.plot(range(steps + 1), x[:, 1], 'b-')
    plt.plot(range(steps + 1), avg1, 'r--')
    plt.plot(range(steps + 1), avg2, 'b--')
    plt.plot(range(steps + 1), avg1 - numpy.sqrt(cov1), 'r-.')
    plt.plot(range(steps + 1), avg1 + numpy.sqrt(cov1), 'r-.')
    plt.plot(range(steps + 1), avg2 - numpy.sqrt(cov2), 'b-.')
    plt.plot(range(steps + 1), avg2 + numpy.sqrt(cov2), 'b-.')

    # print "smoothing ltv: (%s/%s) (%s/%s)" % (avg1, cov1, avg2, cov2)

    avg1 = numpy.mean(straj_mlnlg.traj[:, :, 0], 1)
    stdd = numpy.std(straj_mlnlg.traj[:, :, 0], 1)
    cov1 = stdd ** 2
    plt.plot(range(steps + 1), avg1, 'g--')
    plt.plot(range(steps + 1), avg1 + stdd, 'g-.')
    plt.plot(range(steps + 1), avg1 - stdd, 'g-.')
    plt.plot(range(steps + 1), numpy.mean(straj_mlnlg.traj[:, :, 1], 1), 'k--')
    plt.plot(range(steps + 1),
             numpy.mean(straj_mlnlg.traj[:, :, 1], 1) - numpy.sqrt(numpy.mean(straj_mlnlg.traj[:, :, 2])),
             'k-.')
    plt.plot(range(steps + 1),
             numpy.mean(straj_mlnlg.traj[:, :, 1], 1) + numpy.sqrt(numpy.mean(straj_mlnlg.traj[:, :, 2])),
             'k-.')
    plt.ioff()

    # print "smoothing mlnlg: (%s/%s) (%s/%s)" % (avg1, cov1, avg2, cov2)

    plt.show()

    plt.draw()
