import numpy
from pyparticleest.models.ltv import LTV

R = numpy.array([[0.01]])
Q = numpy.diag([ 0.01, 0.1])

def generate_reference(z0, P0, theta_true, steps):
    A = numpy.asarray(((1.0, theta_true), (0.0, 1.0)))
    C = numpy.array([[1.0, 0.0]])
    
    states = numpy.zeros((2, steps+1))
    ylist = []
    x0 = numpy.random.multivariate_normal(z0.ravel(), P0)
    states[:,0] = numpy.copy(x0)
    for i in range(steps):
            
        # Calc linear states
        x = states[:,i].reshape((-1,1))
        xn = A.dot(x) + numpy.random.multivariate_normal(numpy.zeros(x0.shape), Q).reshape((-1,1))
        states[:,i+1]=xn.ravel()

        # use the correct particle to generate the true measurement
        y = C.dot(xn).ravel() + numpy.random.multivariate_normal((0.0,),R).reshape((-1,1))
        
        ylist.append(y)
    return (ylist, states)

class ParticleParamTrans(LTV):
    """ Implement a simple system by extending the MixedNLGaussian class """
    def __init__(self, z0, P0, params):
        """ Define all model variables """
        A= numpy.array([[1.0, params[0]], [0.0, 1.0]])

        C = numpy.array([[1.0, 0.0]])

        z0 =  numpy.copy(z0).reshape((-1,1))
        # Linear states handled by base-class
        super(ParticleParamTrans,self).__init__(z0=z0, P0=P0, A=A,
                                                C=C, R=R, Q=Q,)
        
    def set_params(self, params):
        """ New set of parameters """
        # Update all needed matrices and derivates with respect
        # to the new parameter set
        A= numpy.array([[1.0, params[0]], [0.0, 1.0]])
        A_grad= numpy.array([[0.0, 1.0], [0.0, 0.0]])
        self.kf.set_dynamics(A=A)

