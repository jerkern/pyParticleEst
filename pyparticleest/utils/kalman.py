#!/usr/bin/python
"""
A module with operations useful for Kalman filtering.
"""
import numpy as np
import math
import scipy.linalg

from builtins import range

l2pi = math.log(2 * math.pi)


def lognormpdf(err, S):
    """
    Calculate gaussian probability density of err, when err ~ N(0,sigma)
    """
    tmp = err.reshape(-1, 1)
    return -0.5 * (S.shape[0] * l2pi + np.linalg.slogdet(S)[1] + np.linalg.solve(S, tmp).T.dot(tmp))


def lognormpdf_cho(err, Schol):
    """
    Calculate gaussian probability density of err, when err ~ N(0,Schol*Scholt^T)
    """
    dim = len(err)
    ld = np.sum(np.log(np.diag(Schol[0]))) * 2
    return -0.5 * (dim * l2pi + ld + scipy.linalg.cho_solve(Schol, err, check_finite=False).T.dot(err))


def lognormpdf_cho_vec(err, Schol):
    """
    Calculate gaussian probability density of for all elements in the vector err
    , when err[i] ~ N(0,Schol*Scholt^T)
    """
    N = err.shape[0]
    dim = err.shape[1]
    ld = np.sum(np.log(np.diag(Schol[0]))) * 2
    res = np.ones((N,)) * (-0.5 * (dim * l2pi + ld))
    for i in range(N):
        res[i] += -0.5 * scipy.linalg.cho_solve(Schol, err[i], check_finite=False).T.dot(err[i])
    return res


def lognormpdf_vec(err, Sl):
    """
    Calculate gaussian probability density of all elements in err, when
    err[i] ~ N(0,Sl[i])
    """
    N = len(err)
    res = np.empty(N)

    for i in range(N):
        S = Sl[i]
        res[i] = -0.5 * (S.shape[0] * l2pi + np.linalg.slogdet(S)
                         [1] + np.linalg.solve(S, err[i]).T.dot(err[i]))
    return res


def lognormpdf_scalar(err, S):
    """
    Calculate gaussian probability density of all elements in err, when
    err[i] ~ N(0,S) and each element in err is a scalar
    """
    return -0.5 * (l2pi + math.log(S[0, 0]) + (err.ravel() ** 2) / S[0, 0])


class KalmanFilter(object):
    """
    A Kalman filter class, does filtering for systems of the type:
    z_{k+1} = A*z_{k}+f_k + v_k
    y_k = C*z_k +f_k e_k
    f_k - Additive (time-varying) constant
    h_k - Additive (time-varying) constant
    v_k ~ N(0,Q)
    e_k ~ N(0,R)
    """

    def __init__(self, lz, A=None, C=None, Q=None, R=None, f_k=None, h_k=None):

        self.A = None
        self.C = None
        self.R = None  # Measurement noise covariance
        self.Q = None  # Process noise covariance
        if (f_k is None):
            self.f_k = np.zeros((lz, 1))
        self.h_k = None
        self.lz = lz
        self.set_dynamics(A, C, Q, R, f_k, h_k)

    def set_dynamics(self, A=None, C=None, Q=None, R=None, f_k=None, h_k=None):
        if (A is not None):
            self.A = A
        if (C is not None):
            self.C = C
        if (Q is not None):
            self.Q = Q
        if (R is not None):
            self.R = R
        if (f_k is not None):
            self.f_k = f_k
        if (h_k is not None):
            self.h_k = h_k

    def time_update(self):
        """
        Do a time update, i.e. predict one step forward in time using the dynamics
        """

        # Calculate next state
        (self.z, self.P) = self.predict_full(A=self.A, f_k=self.f_k, Q=self.Q)

    def predict(self, z, P):
        """
        Calculate next state estimate without actually updating
        the internal variables
        """
        return self.predict_full(z, P, A=self.A, f_k=self.f_k, Q=self.Q)

    def predict_full_inplace(self, z, P, A, f_k, Q):
        """
        Update the estimates to time t+1, using the supplied matrices as the dynamics
        """
        z[:] = f_k + A.dot(z)  # Calculate the next state
        P[:, :] = A.dot(P).dot(A.T) + Q  # Calculate the estimated variance
        return (z, P)

    def predict_full(self, z, P, A, f_k, Q):
        """
        Calculate next state estimate without actually updating
        the internal variables, using the supplied matrices as the dynamics
        """
        z = f_k + A.dot(z)  # Calculate the next state
        P = A.dot(P).dot(A.T) + Q  # Calculate the estimated variance
        return (z, P)

    def measurement_diff(self, y, z, C, h_k=None):
        """
        Calculate different between measurement and predicted measurement
        """
        yhat = np.zeros_like(y)
        if (C is not None):
            yhat += C.dot(z)
        if (h_k is not None):
            yhat += h_k
        return (y - yhat)

    def measure(self, y, z, P):
        """
        Do a measurement update, i.e correct the current estimate
        with information from a new measurement
        """

        return self.measure_full(y, z, P, C=self.C, h_k=self.h_k, R=self.R)

    def measure_full(self, y, z, P, C, h_k, R):
        """
        Do a measurement update, i.e correct the current estimate
        with information from a new measurement
        """
        if (C is not None):
            S = C.dot(P).dot(C.T) + R
            Schol = scipy.linalg.cho_factor(S, check_finite=False)
            err = y - C.dot(z)
            if (h_k is not None):
                err -= h_k
            Sinv_err = scipy.linalg.cho_solve(Schol, err, check_finite=False)
            z[:] = z + P.dot(C.T).dot(Sinv_err)
            P[:, :] = P - P.dot(C.T).dot(scipy.linalg.cho_solve(Schol, C.dot(P), check_finite=False))
        else:
            if (h_k is not None):
                err = y - h_k
            else:
                err = y
            Schol = scipy.linalg.cho_factor(R, check_finite=False)
            Sinv_err = scipy.linalg.cho_solve(Schol, err, check_finite=False)

        # Return the probability of the received measurement
        dim = len(y)
        ld = np.sum(np.log(np.diag(Schol[0]))) * 2
        return -0.5 * (dim * l2pi + ld + err.T.dot(Sinv_err))

    def measure_full_scalar(self, y, z, P, C, h_k, R):
        """
        Do a measurement update, i.e correct the current estimate
        with information from a new measurement.

        Must be scalar measurement equation
        """
        if (C is not None):
            S = C.dot(P).dot(C.T) + R
            err = y - C.dot(z)
            if (h_k is not None):
                err -= h_k
            z[:] = z + P.dot(C.T).dot(err) / S[0, 0]
            tmp = C.dot(P)
            P[:, :] = P - tmp.T.dot(tmp) / S[0, 0]
        else:
            S = R
            if (h_k is not None):
                err = y - h_k
            else:
                err = y

        # Return the probability of the received measurement
        return lognormpdf_scalar(err, S)


class KalmanSmoother(KalmanFilter):
    """
    Forward/backward Kalman smoother

    Extends the KalmanFilter class and provides an additional method for smoothing
    backwards in time
    """

    def smooth(self, z, P, z_next, P_next, A, f, Q):
        """
        Create smoothed estimate using knowledge about x_{k+1} and P_{k+1} and
        the relation x_{k+1} = A*x_k + f_k +v_k, v_k ~ (0,Q)
        """

        (z_np, P_np) = self.predict_full(z, P, A, f, Q)
        J = P.dot(A.T.dot(np.linalg.inv(P_np)))
        z_smooth = z + J.dot(z_next - z_np)
        P_smooth = P + J.dot((P_next - P_np).dot(J.T))
        M_smooth = J.dot(P_next)
        return (z_smooth, P_smooth, M_smooth)
