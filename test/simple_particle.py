import numpy
import math
import mixed_nl_gaussian

class SimpleParticle(mixed_nl_gaussian.MixedNLGaussian):
    """ Implement a simple system by extending the MixedNLGaussian class """
    def __init__(self, x0, c):
        """ Define all model variables """
        A = numpy.array([[1.0, 1.0], [0.0, 1.0]])
        B = numpy.array([[0.0, 0.0], [1.0, -1.0]])
        C = numpy.array([[c, 0.0]])
        
        P = 100000*numpy.diag([1.0, 1.0])
        
        R = numpy.array([[1.]])
        
        Ae = numpy.zeros((1,2))
        Be = numpy.zeros((1,2))
        
        # These could be made time-varying or otherwise dependent on the nonlinear states
        self.Q_in = numpy.diag([ 0.12, 0.12])
        Qe= numpy.diag([ 0.01,])
        Qz = B.dot(self.Q_in.dot(B.T)) # Noise is acting on input
        Qez = numpy.zeros((1,2))
                   
        # Linear states handled by base-class
        super(SimpleParticle,self).__init__(z0=numpy.reshape(x0,(-1,1)),
                                            P0=P,
                                            e0 = c,
                                            Az=A, Bz=B, C=C,
                                            Ae=Ae, Be=Be,
                                            R=R, Qe=Qe, Qz=Qz, Qez=Qez)

        # Non-linear state, measurement matrix C depends on the value of c
        self.c = c 
        
    def sample_input_noise(self, u): 
        """ Return a perturbed vector u by adding Gaussian noise to the u[2] component """ 
        return numpy.vstack((u[:2], 
                             numpy.random.normal(u[2],math.sqrt(self.Qe)))) 
    
    def update(self, data):
        """ Perform a time update of all states """
        # Update linear states
        super(SimpleParticle, self).update(data=data[:2].reshape((-1,1)))
        # Update non-linear state
        self.c += data[2,0]
        
    def measure(self, y):
        """ Perform a measurement update """
        # measurement matrix C depends on the value of c
        self.set_dynamics(C=numpy.array([[self.c, 0.0]]))
        return super(SimpleParticle,self).measure(y)
    
    def next_pdf(self, next_cpart, u):
        return super(SimpleParticle,self).next_pdf(next_cpart, u[:2].reshape((-1,1)))
    
    def get_nonlin_state(self):
        return numpy.array([[self.c]])

    def set_nonlin_state(self,inp):
        self.c = inp[0]
        self.set_dynamics(C=numpy.array([[self.c, 0.0]]))
        
    def linear_input(self, u):
        return u[:2].reshape((-1,1))
