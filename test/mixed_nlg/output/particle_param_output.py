import numpy
import mixed_nl_gaussian

class ParticleParamOutput(mixed_nl_gaussian.MixedNLGaussian):
    """ Implement a simple system by extending the MixedNLGaussian class """
    def __init__(self, x0, P0, params):
        """ Define all model variables """
        A = numpy.array([[1.0, 1.0], [0.0, 1.0]])
        self.B = numpy.array([[0.0, 0.0], [1.0, -1.0]])
        
        
        C = numpy.array([[params[0], 0.0]])
        
        R = numpy.array([[1.0]])
        
        eta = numpy.array([[1.0,]])
        Ae = numpy.zeros((1,2))
        
        # These could be made time-varying or otherwise dependent on the nonlinear states
        self.Q_in = numpy.diag([ 0.1, 0.1])
        Qe= numpy.diag([ 0.01,])
        Qz = numpy.diag([ 0.0000001, 0.0000001])+self.B.dot(self.Q_in.dot(self.B.T)) # Noise is acting on input
        
                   
        # Linear states handled by base-class
        super(ParticleParamOutput,self).__init__(z0=numpy.reshape(x0,(-1,1)), P0=P0,
                                                 e0 = eta,
                                                 Az=A, C=C, Ae=Ae,
                                                 R=R, Qe=Qe, Qz=Qz)
        
    def update(self, u, noise):
        """ Perform a time update of all states """
        self.set_dynamics(fz=self.B.dot(u))
        # Update linear states
        super(ParticleParamOutput, self).update(u, noise)
        
    def measure(self, y):
        """ Perform a measurement update """
        return super(ParticleParamOutput,self).measure(y)
    
    def next_pdf(self, next_cpart, u):
        self.set_dynamics(fz=self.B.dot(u))
        return super(ParticleParamOutput,self).next_pdf(next_cpart, None)

    def set_params(self, params):
        """ New set of parameters """
        # Update all needed matrices and derivates with respect
        # to the new parameter set
        C = numpy.array([[params[0,0], 0.0]])
        C_grad = (numpy.array([[1.0, 0.0]]),)
        self.set_dynamics(C=C)
        self.set_dynamics_gradient(grad_C=C_grad)
