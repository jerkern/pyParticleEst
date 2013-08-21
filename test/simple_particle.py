import numpy
import mixed_nl_gaussian

class SimpleParticle(mixed_nl_gaussian.MixedNLGaussian):
    """ Implement a simple system by extending the MixedNLGaussian class """
    def __init__(self, x0, c):
        """ Define all model variables """
        A = numpy.array([[1.0, 1.0], [0.0, 1.0]])
        self.B = numpy.array([[0.0, 0.0], [1.0, -1.0]])
        
        
        # Non-linear state, measurement matrix C depends on the value of c
        eta = numpy.array([[c,]])
        C = numpy.array([[eta[0,0], 0.0]])
        
        P = 100000*numpy.diag([1.0, 1.0])
        
        R = numpy.array([[5.]])
        
        Ae = numpy.zeros((1,2))
        
        # These could be made time-varying or otherwise dependent on the nonlinear states
        self.Q_in = numpy.diag([ 0.12, 0.12])
        Qe= numpy.diag([ 0.01,])
        Qz = numpy.diag([ 0.0000001, 0.0000001])+self.B.dot(self.Q_in.dot(self.B.T)) # Noise is acting on input
        Qez = numpy.zeros((1,2))
                   
        # Linear states handled by base-class
        super(SimpleParticle,self).__init__(z0=numpy.reshape(x0,(-1,1)),
                                            P0=P,
                                            e0 = eta,
                                            Az=A, C=C, Ae=Ae,
                                            R=R, Qe=Qe, Qz=Qz, Qez=Qez)
        
    def prep_update(self, u):
        """ Perform a time update of all states """
        self.set_dynamics(fz=self.B.dot(u))
        # Update linear states
        
    def prep_measure(self, y):
        """ Perform a measurement update """
        # measurement matrix C depends on the value of c
        self.set_dynamics(C=numpy.array([[self.eta[0,0], 0.0]]))
        return y
    
    def next_pdf(self, next_cpart, u):
        self.set_dynamics(fz=self.B.dot(u))
        return super(SimpleParticle,self).next_pdf(next_cpart, None)

    def set_nonlin_state(self,inp):
        super(SimpleParticle, self).set_nonlin_state(inp)
        # Update linear dynamics
        self.set_dynamics(C=numpy.array([[self.eta[0,0], 0.0]]))

