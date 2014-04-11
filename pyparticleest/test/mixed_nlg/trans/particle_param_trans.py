import numpy
from pyparticleest.models.mixed_nl_gaussian import MixedNLGaussianInitialGaussian
        
def generate_data(theta, Qxi, Qz, R, steps):
    # Create reference
    y = list()
    x = list()
    e = numpy.random.normal(0.0, 1.0)
    z = numpy.random.normal(1.0, 1.0)
    
    x.append((numpy.vstack((e,z))))
    y.append(None)
    
    for i in range(steps):
        
        #e = e + theta * z + numpy.random.multivariate_normal((0.0,), Qxi)
        #z = z + numpy.random.multivariate_normal((0.0,), Qz)
        e = e + z + numpy.random.multivariate_normal((0.0,), Qxi)
        z = theta * z + numpy.random.multivariate_normal((0.0,), Qz)
        x.append((numpy.vstack((e,z))))
        y.append(e + numpy.random.multivariate_normal((0.0,),R))

    return (x, y)


class ParticleParamTrans(MixedNLGaussianInitialGaussian):
    """ Implement a simple system by extending the MixedNLGaussian class """
    def __init__(self, params, R, Qxi, Qz):
        """ Define all model variables """
        Axi = numpy.eye(1.0)
        Az = numpy.array([[params[0],]])
        C = numpy.array([[0.0,]])
        
        
        z0 = numpy.ones((1,1))
        Pz0 = numpy.eye(1)
        xi0 = numpy.zeros((1,1))
        Pxi0 = numpy.eye(1)

        # Linear states handled by base-class
        super(ParticleParamTrans,self).__init__(Az=Az, C=C, Axi=Axi,
                                                R=R, Qxi=Qxi, Qz=Qz,
                                                z0=z0, xi0=xi0,
                                                Pz0=Pz0, Pxi0=Pxi0)

    def get_nonlin_pred_dynamics(self, particles, u):
        xil = numpy.vstack(particles)[:,0]
        fxil = xil[:,numpy.newaxis,numpy.newaxis]
        return (None, fxil, None)
        
    def get_meas_dynamics(self, y, particles):
        xil = numpy.vstack(particles)[:,0]
        h = xil[:,numpy.newaxis,numpy.newaxis]
        return (numpy.asarray(y).reshape((-1,1)), None, h, None)
    
    def set_params(self, params):
        """ New set of parameters """
        # Update all needed matrices and derivates with respect
        # to the new parameter set
        #Axi = numpy.array([[params[0],]])
        #self.set_dynamics(Axi=Axi)
        Az = numpy.array([[params[0],]])
        self.set_dynamics(Az=Az)
