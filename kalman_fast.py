#! /usr/bin/python
""" A module for quick handling of kalman filtering in the special case where A=I and u=0 """
import numpy as np
import math

class KalmanFilterFast(object):
    """ A Kalman filter class for the case when the transition matrix is identity and no input
        drives the system, diagonal covariance """
    def __init__(self,x0, P0, Q, R):
        self.P = P0
        self.R = R
        self.Q = Q
        self.x_new = x0
        self.K = None
        
        
    def time_update(self, Q = None):
        """Predict one step forward """
        if (Q == None):
            Q = self.Q
       
        self.P = self.P + Q
                        
    def meas_update(self, y,ind, R = None):
        """ Correct estimate using a new measurement,
            ind - the indices in self.x_new that are observed with y """
        
        nbr = ind.astype(int)
        if (R == None):
            R = self.R
        S = self.P[nbr] + R
        Sinv = (1./S)
        self.K = self.P[nbr]*Sinv
        err = y-self.x_new[nbr]
        self.x_new[nbr] = self.x_new[nbr] + self.K*err
        self.P[nbr] = (np.ones((y.size,1))-self.K)*self.P[nbr]
    
        norm_coeff = np.prod(np.sqrt(S))*math.pow(2*math.pi,0.5*len(S))
        val = math.exp(-0.5*np.sum(err.ravel()*Sinv.ravel()*err.ravel()))
        if (math.log(val) - math.log(norm_coeff) < -200.0):
            return 0.0
        else:
            return val/norm_coeff

class KalmanSmootherFast(KalmanFilterFast):
    """ Forward/backward Kalman smoother """
    def smooth(self, x_next, P_next):
        """ Create smoothed estimate """
        
        x_np = self.x_new
        P_np = self.P + self.Q
            
        tmp = self.P/P_np
        x_smooth = self.x_new + tmp*(x_next-x_np)
        P_smooth = self.P + tmp*(P_next - P_np)*tmp
        
        self.x_new = x_smooth
        self.P = P_smooth       
