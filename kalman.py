#! /usr/bin/python
""" A module for handling kalman filtering """
import numpy as np
import math
import scipy.sparse as sp
import scipy.sparse.linalg as spln

class KalmanFilter(object):
    """ A Kalman filter class """
    def __init__(self, A, B, C, x0, P0, Q, R):
        self.A = A
        self.B = B
        self.C = C
        self.P = P0
        #self.I = np.zeros(np.shape)
        self.R = R
        self.Q = Q
        self.x_new = x0
        self.K = None #np.zeros([np.shape(P0)[0],np.shape(C)[0]])
        
        
    def time_update(self, A = None, B = None, Q = None, u = None):
        """ A method for performing the Kalman filter time update """
        if (Q == None):
            Q = self.Q
        if (A == None):
            A = self.A
        if (B == None):
            B = self.B
        if (u == None):
            #k-diag is ones, but outside, so this is zero matrix
            u = (sp.eye(B.shape[1], 1, k=2)).tocsr()
        
        tmp = self.predict(A, B, Q, u)
        self.x_new = tmp[0]
        self.P = tmp[1]
        
    def predict(self, A, B, Q, u):
        tmp = B.dot(u)
        x = A.dot(self.x_new) + tmp
        P = A.dot(self.P).dot(A.T) + Q
        return (x, P)
    
    def meas_update(self, y, C = None, R = None):
        """ A method for handling the Kalman filter meas update """
        if (C == None):
            C = self.C
        if (R == None):
            R = self.R
        #S = np.dot(np.dot(C, self.P), C.T)+R
        S = C.dot(self.P).dot(C.T)+R
        #if (len(R) == 1):
        #    Sinv = 1./S
        #else:
        #    Sinv = ln.solve(S, np.eye(C.shape[0]))

        #self.K = np.dot(self.P , np.dot(C.T, Sinv) )
        #self.K = self.P.dot(spln.spsolve(S.T, C).T)
#        self.K = np.zeros([self.P.shape[0],C.shape[0]])
#        for i in range(self.K.shape[1]):
#            tmp = C[i,:]
#            self.K[:,i] = self.P.dot(spln.spsolve(S.T, tmp).T)
        if (sp.issparse(S)):
            Sd = S.todense() # Ok if dimension of S is small compared to other matrices 
            Sinv = np.linalg.inv(Sd)
            self.K = self.P.dot(C.T).dot(sp.csr_matrix(Sinv))
            norm_coeff = math.sqrt(np.linalg.det(Sd)*math.pow(2*math.pi,len(Sd)))
        else:
            Sinv = np.linalg.inv(S)
            self.K = self.P.dot(C.T).dot(Sinv)
            norm_coeff = math.sqrt(np.linalg.det(S)*math.pow(2*math.pi,len(S)))
            
#        tmp = np.zeros(C.shape)
#        for i in range(tmp.shape[1]):
#            tmp[:,i] = spln.spsolve(S.T, C[:,i])http://sites.google.com/site/wayneholder/gps-vs-barometric-altitude
        
        #err = y-np.dot(C, self.x_new)
        err = y-C.dot(self.x_new)
        self.x_new = self.x_new + self.K.dot(err)  
        #self.P = np.dot((np.eye(len(self.x_new))-np.dot(self.K, C)), self.P)
        #self.P = (sp.identity(len(self.x_new))-self.K.dot(C)).dot(self.P)
        self.P = (sp.identity(len(self.x_new))-C.T.dot(self.K.T).T).dot(self.P)
        #return np.dot(err.T, np.linalg.solve(np.dot( np.dot(C, self.P), C.T), err))

        if (sp.issparse(S)):
            numerator = spln.spsolve(S, err).T.dot(err)
        else:
            numerator = np.linalg.solve(S, err).T.dot(err)
        return math.exp(-0.5*numerator)/norm_coeff



class KalmanSmoother(KalmanFilter):
    """ Forward/backward Kalman smoother """
    def smooth(self, x_next, P_next, u=None, A=None, B=None, Q=None):
        """ Create smoothed estimate """
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