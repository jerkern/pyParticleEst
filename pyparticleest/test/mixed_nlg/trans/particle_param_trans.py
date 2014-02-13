import numpy
from pyparticleest.models.mixed_nl_gaussian import MixedNLGaussian

class ParticleParamTrans(MixedNLGaussian):
    """ Implement a simple system by extending the MixedNLGaussian class """
    def __init__(self, eta0, z0, P0, params):
        """ Define all model variables """
        Ae = numpy.array([[params[0],]])

        C = numpy.array([[0.0,]])
        
        R = numpy.array([[0.1]])

        Az = numpy.eye(1.0)
        
        Qe= numpy.diag([ 0.1,])
        Qz = numpy.diag([ 0.1,])
        z0 =  numpy.copy(z0).reshape((-1,1))
        fe = numpy.copy(eta0).reshape((-1,1))
        # Linear states handled by base-class
        super(ParticleParamTrans,self).__init__(z0=numpy.reshape(z0,(-1,1)), P0=P0,
                                                 e0 = eta0,
                                                 Az=Az, C=C, Ae=Ae,
                                                 R=R, Qe=Qe, Qz=Qz,
                                                 fe=fe, params=params)
        
    def prep_update(self, u):
        """ Perform a time update of all states """
        fe = numpy.copy(self.eta)
        self.set_dynamics(fe=fe)
        
    def prep_measure(self, y):
        """ Perform a measurement update """
        h = numpy.copy(self.eta)
        self.set_dynamics(h=h)
        return y
    
    def next_pdf(self, next_cpart, u):
        return super(ParticleParamTrans,self).next_pdf(next_cpart, None)

    def set_params(self, params):
        """ New set of parameters """
        # Update all needed matrices and derivates with respect
        # to the new parameter set
        Ae = numpy.array([[params[0],]])
        Ae_grad = [numpy.array([[1.0,]]),]
        self.set_dynamics(Ae=Ae)
        self.set_dynamics_gradient(grad_Ae=Ae_grad)
        return super(ParticleParamTrans, self).set_params(params)
