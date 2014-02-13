import numpy
import pyparticleest.models.mixed_nl_gaussian as mixed_nl_gaussian

def generate_refernce(z0, P0, Qz, R, uvec, steps, c_true):
    A = numpy.asarray(((1.0, 1.0), (0.0, 1.0)))
    B = numpy.array([[0.0, 0.0], [1.0, -1.0]])
    C = numpy.array([[c_true, 0.0]])
    
    states = numpy.zeros((2, steps+1))
    ulist = []
    ylist = []
    x0 = numpy.random.multivariate_normal(z0.ravel(), P0)
    states[:,0] = numpy.copy(x0)
    for i in range(steps):
            
        # Calc linear states
        u = uvec[:,i].reshape(-1,1)
        x = states[:,i].reshape((-1,1))
        xn = A.dot(x) + B.dot(u) + numpy.random.multivariate_normal(numpy.zeros((2,)), Qz).reshape((-1,1))
        states[:,i+1]=xn.ravel()

        # use the correct particle to generate the true measurement
        y = C.dot(xn).ravel() + numpy.random.normal((0.0,),R).reshape((-1,1))
        
        ulist.append(u)
        ylist.append(y)
    return (ulist, ylist, states)


class ParticleParamOutput(mixed_nl_gaussian.MixedNLGaussian):
    """ Implement a simple system by extending the MixedNLGaussian class """
    def __init__(self, x0, P0, Qz, R, params):
        """ Define all model variables """
        A = numpy.array([[1.0, 1.0], [0.0, 1.0]])
        self.B = numpy.array([[0.0, 0.0], [1.0, -1.0]])
        
        C = numpy.array([[params[0], 0.0]])
        
        R = numpy.array([[1.0]])
        
        eta = numpy.array([[1.0,]])
        Ae = numpy.zeros((1,2))

        Qe= numpy.diag([ 0.00000001,])
        fe = numpy.copy(eta)
                   
        # Linear states handled by base-class
        super(ParticleParamOutput,self).__init__(z0=numpy.reshape(x0,(-1,1)), P0=P0,
                                                 e0 = eta, fe=fe,
                                                 Az=A, C=C, Ae=Ae,
                                                 R=R, Qe=Qe, Qz=Qz)
        
    def prep_update(self, u):
        """ Perform a time update of all states """
        fe = numpy.copy(self.eta)
        self.set_dynamics(fz=self.B.dot(u), fe=fe)
        
    def prep_measure(self, y):
        return y    
    
    def set_params(self, params):
        """ New set of parameters """
        # Update all needed matrices and derivates with respect
        # to the new parameter set
        C = numpy.array([[params[0], 0.0]])
        C_grad = (numpy.array([[1.0, 0.0]]),)
        self.set_dynamics(C=C)
        self.set_dynamics_gradient(grad_C=C_grad)
        return super(ParticleParamOutput, self).set_params(params)