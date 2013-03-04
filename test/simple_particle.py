import numpy
import math
import copy
import mixed_nl_gaussian
import kalman

class SimpleParticle(mixed_nl_gaussian.MixedNLGaussian):
    """ Implement a simple system by extending the MixedNLGaussian class """
    def __init__(self, x0, c):
        
        A = numpy.array([[1.0, 1.0], [0.0, 1.0]])
        B = numpy.array([[0.0, 0.0], [1.0, -1.0]])
        C = numpy.array([[c, 0.0]])
        
        P = numpy.diag([1.0, 1.0])
        
        R = numpy.array([[1.]])
        
        # These could be made time-varying or otherwise dependent on the nonlinear states
        self.Q = numpy.diag([ 0.01, 0.0, 0.0])
        self.Q_in = numpy.diag([ 0.12, 0.12])

        self.kf = kalman.KalmanSmoother(A,B,C, numpy.reshape(x0,(-1,1)), P0=P, Q=None, R=R)
        # Non-linear state, measurement matrix C depends on the value of c
        self.c = c 
        
    def sample_input_noise(self, u):
        """ Return a perturbed vector u by adding guassian noise to the u[2] component """ 
        return numpy.vstack((u[:2], numpy.random.normal(u[2],math.sqrt(self.Q[0,0])))) 
    
    def update(self, data):
        # Update linear states
        self.kf.time_update(u=self.linear_input(data),Q=self.get_lin_Q())
        # Update non-linear state
        self.c += data[2,0]
        
    def measure(self, y):
        # measurement matrix C depends on the value of c
        return self.kf.meas_update(y, C=numpy.array([[self.c, 0.0]]))
    
    def get_R(self):
        return self.kf.R
    
    def get_lin_A(self):
        return self.kf.A
    
    def get_lin_B(self):
        return self.kf.B

    def get_lin_C(self):
        return numpy.array([[self.c, 0.0]])

    def get_nonlin_B(self):
        return numpy.zeros((1,2))
    
    def get_full_B(self):
        return numpy.vstack((self.get_nonlin_B(),self.kf.B))

    def get_nonlin_A(self):
        return numpy.zeros((1,2))
    
    def get_full_A(self):
        return numpy.vstack((self.get_nonlin_A(),self.get_lin_A()))
    
    def get_lin_Q(self):
        return self.get_Q()[1:,1:]
    
    def get_Q(self):
        B = self.get_full_B()
        return self.Q+B.dot(self.Q_in.dot(B.T))
    
    def get_nonlin_state(self):
        return numpy.array([[self.c]])

    def set_nonlin_state(self,inp):
        self.c = inp[0]

    def get_lin_est(self):
        return (self.kf.x_new, self.kf.P)

    def set_lin_est(self, est):
        (inp,Pin) = est
        self.kf.x_new = copy.copy(inp)
        self.kf.P = copy.copy(Pin)
        
    def linear_input(self, u):
        return u[:2].reshape((-1,1))
