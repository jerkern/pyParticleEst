import numpy
from pyparticleest.models.mixed_nl_gaussian import MixedNLGaussian

def generate_reference(P0, Qz, R, params, steps):
    A = numpy.asarray(((1.0),))
    C = numpy.array([[1.0,],])
    
    states = numpy.zeros((1, steps+1))
    ylist = []
    z0 = params[0]
    x0 = numpy.copy(z0) #numpy.random.multivariate_normal((z0,), P0)
    states[:,0] = numpy.copy(x0)
    for i in range(steps):
            
        x = states[:,i].reshape((-1,1))
        xn = A.dot(x)+numpy.random.multivariate_normal((0.0,), Qz).reshape((-1,1))
        states[:,i+1]=xn.ravel()

        # use the correct particle to generate the true measurement
        y = C.dot(xn).ravel() + numpy.random.multivariate_normal((0.0,), R).reshape((-1,1))
        
        ylist.append(y)
    return (ylist, states)


class ParticleParamInit(MixedNLGaussian):
    """ Implement a simple system by extending the MixedNLGaussian class """
    def __init__(self, P0, Qz, R, params):
        """ Define all model variables """
        A = numpy.array([[1.0,],])
        
        C = numpy.array([[1.0,],])
        
        # Fake non-linear state
        eta = numpy.array([[1.0,]])
        Ae = numpy.zeros((1,1))
        
        z0 = numpy.reshape(params[0],(-1,1))

        Qe= numpy.diag([ 0.00000001,])
        fe = numpy.copy(eta)
                   
        # Linear states handled by base-class
        super(ParticleParamInit,self).__init__(z0=z0, P0=P0,
                                                 e0 = eta, fe=fe,
                                                 Az=A, C=C, Ae=Ae,
                                                 R=R, Qe=Qe, Qz=Qz)
        
        self.z0 = numpy.copy(z0)
        self.P0 = numpy.copy(P0)
        self.grad_z0 = (numpy.array((1.0,),),)
        self.grad_P0 = (numpy.zeros(self.P0.shape),)
        
    def prep_update(self, u):
        """ Perform a time update of all states """
        fe = numpy.copy(self.eta)
        self.set_dynamics(fe=fe)
        
    def prep_measure(self, y):
        return y    

    def get_z0_initial(self):
        return (self.z0, self.P0)

    def get_grad_z0_initial(self):
        return (self.grad_z0, self.grad_P0)
    
    def set_params(self, params):
        """ New set of parameters """
        # Update all needed matrices and derivates with respect
        # to the new parameter set
        self.z0 = numpy.array([[params[0],]]).reshape((-1,1))
        return super(ParticleParamInit, self).set_params(params)