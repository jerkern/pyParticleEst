#! /usr/bin/python
""" A module for handling Kalman filtering.
    Uses scipy.sparse for handling sparse matrices, works with dense matrices aswell """
import numpy as np
import math
import scipy.sparse as sp
from numba import jit, double
#import scipy.sparse.linalg as spln

l2pi = math.log(2*math.pi)
def lognormpdf(x,mu,S):
    """ Calculate gaussian probability density of x, when x ~ N(mu,sigma) """
    err = x-mu
    return -0.5*(S.shape[0]*l2pi+np.linalg.slogdet(S)[1]+np.linalg.solve(S, err).T.dot(err))

def lognormpdf_vec(xl,mul,Sl):
    """ Calculate gaussian probability density of x, when x ~ N(mu,sigma) """
    N = len(xl)
    res = np.empty(N, dtype=float)

    for i in range(N):
        err = xl[i]-mul[i]
        S = Sl[i]
        res[i] = -0.5*(S.shape[0]*l2pi+np.linalg.slogdet(S)[1]+np.linalg.solve(S, err).T.dot(err))
    return res

lognormpdf_jit = jit(double[:,:](double[:,:], double[:,:,:]))(lognormpdf_vec)

#@autojit
#def lognormpdf_vec(xl,mul,Sl):
#    """ Calculate gaussian probability density of x, when x ~ N(mu,sigma) """
#    N = len(xl)
#    res = np.empty(N, dtype=float)
#    for i in xrange(N):
#        err = xl[i]-mul[i]
#        res[i] = -0.5*(Sl[i].shape[0]*l2pi+np.linalg.slogdet(Sl[i])[1]+np.linalg.solve(Sl[i], err).T.dot(err))
#    return res

#@jit(double[:](double[:,:],double[:,:],double[:,:,:]))


class KalmanFilter(object):
    """ A Kalman filter class, does filtering for systems of the type:
        z_{k+1} = A*z_{k}+f_k + v_k
        y_k = C*z_k +f_k e_k
        f_k - Additive (time-varying) constant
        h_k - Additive (time-varying) constant
        v_k ~ N(0,Q)
        e_k ~ N(0,R)
        """
        
    def __init__(self, lz, A=None, C=None, Q=None, R=None, f_k=None, h_k=None):
        """ z_{k+1} = A*z_{k}+f_k + v_k
        y_k = C*z_k + h_k + e_k 
        v_k ~ N(0,Q)
        e_k ~ N(0,R)
        """
        self.A = None
        self.C = None
        self.R = None      # Measurement noise covariance
        self.Q = None      # Process noise covariance
        if (f_k == None):
            self.f_k = np.zeros((lz,1))
        self.h_k = None
        self.lz = lz       
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
        (self.z, self.P) = self.predict_full(A=self.A, f_k=self.f_k, Q=self.Q)  
        
    def predict(self, z, P):
        return self.predict_full(z, P, A=self.A, f_k=self.f_k, Q=self.Q)
    
    def predict_full_inplace(self, z, P, A, f_k, Q):
        """ Calculate next state estimate without actually updating
            the internal variables """
        z[:] = f_k + A.dot(z)     # Calculate the next state
        P[:,:] = A.dot(P).dot(A.T) + Q  # Calculate the estimated variance  
        return (z, P)
    
    def predict_full(self, z, P, A, f_k, Q):
        """ Calculate next state estimate without actually updating
            the internal variables """
        z = f_k + A.dot(z)     # Calculate the next state
        P = A.dot(P).dot(A.T) + Q  # Calculate the estimated variance  
        return (z, P)
    
    def measurement_diff(self, y, z, C, h_k=None):
        yhat = np.zeros_like(y)
        if (C != None):
            yhat += C.dot(z)
        if (h_k != None):
            yhat += h_k
        return (y-yhat)
    
    def measure(self, y, z, P):
        """ Do a measurement update, i.e correct the current estimate 
            with information from a new measurement """

        return self.measure_full(y, z, P, C=self.C, h_k=self.h_k, R=self.R)

    def measure_full(self, y, z, P, C, h_k, R):
        if (C != None):
            S = C.dot(P).dot(C.T)+R
  

            if (sp.issparse(S)):
                Sd = S.todense() # Ok if dimension of S is small compared to other matrices 
                Sinv = np.linalg.inv(Sd)
                K = P.dot(C.T).dot(sp.csr_matrix(Sinv))
            else:
                Sinv = np.linalg.inv(S)
                K = P.dot(C.T).dot(Sinv)
             
            err = self.measurement_diff(y, z, C, h_k)
            z[:] = z + K.dot(err)  
            P[:,:] = P - K.dot(C).dot(P)
        else:
            if (h_k != None):
                err = y - h_k
            else:
                err = y
            S = R
        
        
        
        # Return the probability of the received measurement
        return lognormpdf(0, err, S)


class KalmanSmoother(KalmanFilter):
    """ Forward/backward Kalman smoother
    
        Extends the KalmanFilter class and provides an additional method for smoothing
        backwards in time """
    
#    def __init__(self, z0, P0, **kwargs):
#        super(KalmanSmoother, self).__init__(z0=z0, P0=P0, **kwargs)
#        self.M = None
        
    def smooth(self, z, P, z_next, P_next, A, f, Q):
        """ Create smoothed estimate using knowledge about x_{k+1} and P_{k+1} and
            the relation x_{k+1} = A*x_k + f_k +v_k
            v_k ~ (0,Q)"""
        
        (z_np, P_np) = self.predict_full(z, P, A, f, Q)
        J = P.dot(A.T.dot(np.linalg.inv(P_np)))
        z_smooth = z + J.dot(z_next-z_np)
        P_smooth = P + J.dot((P_next - P_np).dot(J.T))
        M_smooth = J.dot(P_next)
        return (z_smooth, P_smooth, M_smooth)
