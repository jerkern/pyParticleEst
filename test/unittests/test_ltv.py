'''
Created on Jul 23, 2015

@author: ajn
'''
import unittest
import pyparticleest.models.ltv as ltv
import numpy
import numpy.testing as npt
import math

class Model(ltv.LTV):
    """ x_{k+1} = sin(x_k) + v_k, v_k ~ N(0,Q)
        y_k = x_k + e_k, e_k ~ N(0,R),
        x(0) ~ N(0,P0) """

    def __init__(self, x0, P0, A, C, f, Q, R):
        super(Model, self).__init__(numpy.asarray(x0).reshape((1, 1)),
                                    numpy.asarray(P0).reshape((1, 1)),
                                    A=numpy.asarray(A).reshape((1, 1)),
                                    C=numpy.asarray(C).reshape((1, 1)),
                                    f=numpy.asarray(f).reshape((1, 1)),
                                    Q=numpy.asarray(Q).reshape((1, 1)),
                                    R=numpy.asarray(R).reshape((1, 1)))

class Test(unittest.TestCase):


    def setUp(self):
        self.A = 0.9
        self.C = 1.5
        self.f = 0.5
        self.R = 0.5
        self.Q = 2.0
        self.P0 = 3.0
        self.z0 = -0.5

        self.model = Model(self.z0, self.P0, self.A, self.C,
                           self.f, self.Q, self.R)

    def tearDown(self):
        pass


    def testUpdate(self):
        particles = self.model.create_initial_estimate(1)
        nextp = self.model.update(numpy.copy(particles), None, None, None)

        (zl, Pl) = self.model.get_states(particles)
        (nzl, nPl) = self.model.get_states(nextp)
        npt.assert_array_equal(numpy.asarray(nzl), self.A * numpy.asarray(zl) + self.f)
        npt.assert_array_equal(numpy.asarray(nPl), (self.A ** 2) * numpy.asarray(Pl) + self.Q)


    def testMeasure(self):
        particles = self.model.create_initial_estimate(1)
        (zl, Pl) = self.model.get_states(particles)
        y = 1.0
        # https://en.wikipedia.org/wiki/Kalman_filter
        S = self.C * self.P0 * self.C + self.R
        K = self.P0 * self.C / S
        xn = zl[0] + K * (y - self.C * zl[0])
        Pn = Pl[0] - K * self.C * Pl[0]

        partn = numpy.copy(particles)
        _ = self.model.measure(partn, numpy.asarray(y).reshape((-1, 1)), None)
        (nzl, nPl) = self.model.get_states(partn)

        npt.assert_array_almost_equal(xn[0].ravel(), nzl[0].ravel(), 10)
        npt.assert_array_almost_equal(Pn[0].ravel(), nPl[0].ravel(), 10)




if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
