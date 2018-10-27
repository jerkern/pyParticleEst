'''
Created on Jul 23, 2015

@author: ajn
'''
import unittest
import pyparticleest.models.nlg as nlg
import numpy
import numpy.testing as npt
import math


class Model(nlg.NonlinearGaussianInitialGaussian):
    """ x_{k+1} = sin(x_k) + v_k, v_k ~ N(0,Q)
        y_k = x_k + e_k, e_k ~ N(0,R),
        x(0) ~ N(0,P0) """

    def __init__(self, P0, Q, R):
        x0 = numpy.zeros((1, 1))
        super(Model, self).__init__(x0=x0,
                                    Px0=numpy.asarray(P0).reshape((1, 1)),
                                    Q=numpy.asarray(Q).reshape((1, 1)),
                                    R=numpy.asarray(R).reshape((1, 1)))

    def calc_f(self, particles, u, t):
        return numpy.sin(particles)

    def calc_g(self, particles, t):
        return particles


class Test(unittest.TestCase):

    def setUp(self):
        self.model = Model(1.0, 1.0, 1.0)

    def tearDown(self):
        pass

    def testUpdate(self):
        N = 10
        particles = self.model.create_initial_estimate(N)
        noise = self.model.sample_process_noise(particles, None, None)
        nextp = self.model.update(numpy.copy(particles), None, None, noise)

        npt.assert_array_equal(nextp, numpy.sin(particles) + noise)

    def testMeasure(self):
        N = 10
        particles = self.model.create_initial_estimate(N)
        y = 1.0
        logpy = self.model.measure(particles, y, None)
        logpy_correct = -0.5 * \
            math.log(2.0 * math.pi) - 0.5 * (y - particles.ravel()) ** 2

        npt.assert_array_equal(logpy, logpy_correct)

    def testInitial(self):
        N = 10000000
        particles = self.model.create_initial_estimate(N)
        m = numpy.mean(particles)
        s = numpy.std(particles)

        self.assertAlmostEqual(m, 0.0, 2)
        self.assertAlmostEqual(s, 1.0, 2)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
