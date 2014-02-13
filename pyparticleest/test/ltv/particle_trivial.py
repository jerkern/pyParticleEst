'''
Created on Nov 29, 2013

@author: ajn
'''
import numpy
from pyparticleest.models.ltv import LTV

R = numpy.array([[0.5]])
Q = numpy.diag([ 0.1,])

def generate_reference(z0, P0, theta_true, steps):
    A = numpy.asarray(((theta_true,),))
    C = numpy.array([[1.0,]])
    
    states = numpy.zeros((1, steps+1))
    ylist = []
    x0 = numpy.random.multivariate_normal(z0.ravel(), P0)
    states[:,0] = numpy.copy(x0)
    f = numpy.array([[0.0,]])
    for i in range(steps):
            
        # Calc linear states
        x = states[:,i].reshape((-1,1))
        xn = A.dot(x) + f + numpy.random.multivariate_normal(numpy.zeros(x0.shape), Q).reshape((-1,1))
        states[:,i+1]=xn.ravel()

        # use the correct particle to generate the true measurement
        y = C.dot(xn).ravel() + numpy.random.multivariate_normal((0.0,),R).reshape((-1,1))
        
        ylist.append(y)
    return (ylist, states)

class ParticleTrivial(LTV):
    """ Implement a simple system by extending the MixedNLGaussian class """
    def __init__(self, z0, P0, params):
        """ Define all model variables """
        A= numpy.array([[params[0]],])

        C = numpy.array([[1.0,]])
        f = numpy.array([[0.0,]])
        z0 =  numpy.copy(z0).reshape((-1,1))
        # Linear states handled by base-class
        super(ParticleTrivial,self).__init__(z0=z0, P0=P0, A=A,
                                                C=C, R=R, Q=Q, f=f)
        
    def prep_update(self, u):
        """ Perform a time update of all states """
        return
        
    def prep_measure(self, y):
        """ Perform a measurement update """
        return y
    
    def set_params(self, params):
        """ New set of parameters """
        # Update all needed matrices and derivates with respect
        # to the new parameter set
        A= numpy.array([[params[0]], ])
        A_grad= numpy.array([[1.0],])
        self.set_dynamics(A=A)
        self.set_dynamics_gradient(grad_A=A_grad)
        return super(ParticleTrivial, self).set_params(params)
