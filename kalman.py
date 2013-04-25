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
    tmp = -0.5*(nx*math.log(2*math.pi)+np.linalg.slogdet(S)[1])
    
    err = x-mu
    if (sp.issparse(S)):
        numerator = spln.spsolve(S, err).T.dot(err)
    else:
        numerator = np.linalg.solve(S, err).T.dot(err)

    return tmp-numerator

class KalmanFilter(object):
    """ A Kalman filter class, does filtering for systems of the type:
        x_{k+1} = A*x_{k}+B*u_k + v_k
        y_k = C*x_k + e_k 
        v_k ~ N(0,Q)
        e_k ~ N(0,R)
        """
        
    def __init__(self, A, B, C, x0, P0, Q, R):
        """ x_{k+1} = A*x_{k}+B*u_k + v_k
        y_k = C*x_k + e_k 
        v_k ~ N(0,Q)
        e_k ~ N(0,R)
        x0 = x_0, P0 = P_0
        P is the variance of the estimate
        
        x0 and P0 are mandatory, the matrices can be overriden at each time instant
        if desired, for instance if the system dynamics are time-varying
        """
        self.A = A
        self.B = B
        self.C = C
        
        self.R = R      # Measurement noise covariance
        self.Q = Q      # Process noise covariance
        self.x_new = x0 # Estimated state
        self.P = P0     # Estimated covariance
        self.K = None   # Define self.K for later use
        
        
    def time_update(self, A = None, B = None, Q = None, u = None):
        """ Do a time update, i.e. predict one step forward in time using the input u """
        if (Q == None):
            Q = self.Q
        if (A == None):
            A = self.A
        if (B == None):
            B = self.B
        if (u == None):
            #k-diag is ones, but outside, so this is zero matrix
            u = (sp.eye(B.shape[1], 1, k=2)).tocsr()
        
        tmp = self.predict(A, B, Q, u)  # Calculate next state
        self.x_new = tmp[0]
        self.P = tmp[1]
        
    def predict(self, A, B, Q, u):
        """ Calculate next state estimate without actually updating the internal variables """
        tmp = B.dot(u)                  # Calcate how u affects the states
        x = A.dot(self.x_new) + tmp     # Calculate the next state
        P = A.dot(self.P).dot(A.T) + Q  # Calucate the estimated variane  
        return (x, P)
    
    def meas_update(self, y, C = None, R = None):
        """ Do a measurement update, i.e correct the current estimate with information from a new measurement """
        if (C == None):
            C = self.C
        if (R == None):
            R = self.R

        S = C.dot(self.P).dot(C.T)+R

        if (sp.issparse(S)):
            Sd = S.todense() # Ok if dimension of S is small compared to other matrices 
            Sinv = np.linalg.inv(Sd)
            self.K = self.P.dot(C.T).dot(sp.csr_matrix(Sinv))
        else:
            Sinv = np.linalg.inv(S)
            self.K = self.P.dot(C.T).dot(Sinv)
            
        err = y-C.dot(self.x_new)
        self.x_new = self.x_new + self.K.dot(err)  
        self.P = (sp.identity(len(self.x_new))-C.T.dot(self.K.T).T).dot(self.P)

        # Return the probability of the received measurement
        return lognormpdf(y, C.dot(self.x_new), S)
    


class KalmanSmoother(KalmanFilter):
    """ Forward/backward Kalman smoother
    
        Extends the KalmanFilter class and provides an additional method for smoothing
        backwards in time """
    def smooth(self, x_next, P_next, u=None, A=None, B=None, Q=None):
        """ Create smoothed estimate using knowledge about x_{k+1} and P_{k+1} and
            the relation x_{k+1} = A*x_k + B*u_k +v_k
            v_k ~ (0,Q)"""
        if (Q == None):
            Q = self.Q
        if (A == None):
            A = self.A
        if (B == None):
            B = self.B
        if (u == None):
            #k-diag is ones, but outside, so this is zero matrix
            u = (sp.eye(B.shape[1], 1, k=2)).tocsr()
        
        (x_np, P_np) = self.predict(A, B, Q, u)
        tmp = self.P.dot(A.T.dot(np.linalg.inv(P_np)))
        x_smooth = self.x_new + tmp.dot(x_next-x_np)
        P_smooth = self.P + tmp.dot(P_next - P_np).dot(tmp.T)
        
        self.x_new = x_smooth
        self.P = P_smooth