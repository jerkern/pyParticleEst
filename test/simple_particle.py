import numpy
import math
import copy
import part_utils
import kalman

class SimpleParticle(part_utils.MixedNLGaussian):
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
        self.c = c
        
    def sample_input_noise(self, u):
        #return numpy.vstack((numpy.random.multivariate_normal(u[:2].reshape(-1),self.linQ).reshape((-1,1)), 
        #                     numpy.random.normal(u[2],math.sqrt(self.nonlinQ))))
        return numpy.vstack((u[:2], numpy.random.normal(u[2],math.sqrt(self.Q[0,0])))) 
    
    def update(self, data):
        self.kf.time_update(u=data[:2,:],Q=self.get_lin_Q())
        self.c += data[2,0]
        
    def measure(self, y):
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
