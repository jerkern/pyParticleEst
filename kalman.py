#! /usr/bin/python
""" A module for handling Kalman filtering.
    Uses scipy.sparse for handling sparse matrices, works with dense matrices aswell """
import numpy as np
import math
import scipy.sparse as sp
import scipy.sparse.linalg as spln

def lognormpdf(x,mu,S):
    """ Calculate gaussian probability density of x, when x ~ N(mu,sigma) """
    nx = len(S)
    norm_coeff = -0.5*(nx*math.log(2*math.pi)+np.linalg.slogdet(S)[1])
    
    err = x-mu
    if (sp.issparse(S)):
        numerator = spln.spsolve(S, err).T.dot(err)
    else:
        numerator = np.linalg.solve(S, err).T.dot(err)

    return norm_coeff-numerator

class KalmanFilter(object):
    """ A Kalman filter class, does filtering for systems of the type:
        z_{k+1} = A*z_{k}+f_k + v_k
        y_k = C*z_k +f_k e_k
        f_k - Additive (time-varying) constant
        h_k - Additive (time-varying) constant
        v_k ~ N(0,Q)
        e_k ~ N(0,R)
        """
        
    def __init__(self, z0, P0, A=None, C=None, Q=None, R=None, f_k=None, h_k=None):
        """ z_{k+1} = A*z_{k}+f_k + v_k
        y_k = C*z_k + h_k + e_k 
        v_k ~ N(0,Q)
        e_k ~ N(0,R)
        x0 = x_0, P0 = P_0
        P is the variance of the estimate
        
        x0 and P0 are mandatory, the matrices can be overriden at each time instant
        if desired, for instance if the system dynamics are time-varying
        """
        self.z = np.copy(z0) # Estimated state
        self.P = np.copy(P0)     # Estimated covariance

        self.A = None
        self.C = None
        self.R = None      # Measurement noise covariance
        self.Q = None      # Process noise covariance
        self.f_k = None
        self.h_k = None
        
        self.set_dynamics(A, C, Q, R, f_k, h_k)
        
    def set_dynamics(self, A=None, C=None, Q=None, R=None, f_k=None, h_k=None):
        if (A != None):
            self.A = A
        if (C != None):
            self.C = C
        if (Q != None):
            self.Q = Q
        if (R != None):
            self.R = R
        if (f_k != None):
            self.f_k = f_k
        if (h_k != None):
            self.h_k = h_k
     
    def time_update(self):
        """ Do a time update, i.e. predict one step forward in time using the dynamics """
        
        # Calculate next state
        (self.z, self.P) = self.predict()  
        
    def predict(self):
        """ Calculate next state estimate without actually updating the internal variables """
        A = self.A
        z = A.dot(self.z)     # Calculate the next state
        if (self.f_k != None):
            # Calculate how u affects the states
            z += self.f_k                 
        P = A.dot(self.P).dot(A.T) + self.Q  # Calculate the estimated variance  
        return (z, P)
    
    def measurement_diff(self, y):
        yhat = self.C.dot(self.z)
        if (self.h_k != None):
            yhat += self.h_k
        return y-yhat
    
    def meas_update(self, y):
        """ Do a measurement update, i.e correct the current estimate with information from a new measurement """

        C = self.C
        
        S = C.dot(self.P).dot(C.T)+self.R

        if (sp.issparse(S)):
            Sd = S.todense() # Ok if dimension of S is small compared to other matrices 
            Sinv = np.linalg.inv(Sd)
            K = self.P.dot(C.T).dot(sp.csr_matrix(Sinv))
        else:
            Sinv = np.linalg.inv(S)
            K = self.P.dot(C.T).dot(Sinv)
        
        yhat = C.dot(self.z)
        if (self.h_k != None):
            yhat += self.h_k
        err = self.measurement_diff(y)
        self.z += K.dot(err)  
        self.P -= K.dot(C).dot(self.P)

        # Return the probability of the received measurement
        return lognormpdf(y, yhat, S)
    


class KalmanSmoother(KalmanFilter):
    """ Forward/backward Kalman smoother
    
        Extends the KalmanFilter class and provides an additional method for smoothing
        backwards in time """
    def smooth(self, x_next, P_next):
        """ Create smoothed estimate using knowledge about x_{k+1} and P_{k+1} and
            the relation x_{k+1} = A*x_k + f_k +v_k
            v_k ~ (0,Q)"""
        
        (x_np, P_np) = self.predict()
        tmp = self.P.dot(self.A.T.dot(np.linalg.inv(P_np)))
        x_smooth = self.z + tmp.dot(x_next-x_np)
        P_smooth = self.P + tmp.dot(P_next - P_np).dot(tmp.T)
        
        self.z = x_smooth
        self.P = P_smooth