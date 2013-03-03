#! /usr/bin/python
""" A module for quick handling of kalman filtering for some special cases """
import numpy as np
import math

class KalmanFilterFast(object):
    """ A Kalman filter class for the case when the transition matrix is identity and no input
        drives the system, diagonal covariance """
    def __init__(self,x0, P0, Q, R):
        self.P = P0
        #print np.shape(self.P)
        #self.I = np.zeros(np.shape)
        self.R = R
        self.Q = Q
        self.x_new = x0
        self.K = None #np.zeros([np.shape(P0)[0],np.shape(C)[0]])
        
        
    def time_update(self, Q = None):
        """ A method for performing the Kalman filter time update """
        if (Q == None):
            Q = self.Q
       
        self.P = self.P + Q
                        
    def meas_update(self, y,ind, R = None):
        """ A method for handling the Kalman filter meas update """
        
        #indices = ind[:, 2] + ind[:, 1]+ind[:, 0]
        nbr = ind.astype(int)
        S = self.P[nbr] + R
        Sinv = (1./S)
        self.K = self.P[nbr]*Sinv
        err = y-self.x_new[nbr]
        self.x_new[nbr] = self.x_new[nbr] + self.K*err
        self.P[nbr] = (np.ones((y.size,1))-self.K)*self.P[nbr]
        tmp = err*Sinv
    
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
