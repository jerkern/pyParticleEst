'''
Created on Dec 13, 2013

@author: ajn
'''

import numpy
import math
from pyparticleest.models.hierarchial import Hierarchial

Rz = numpy.array([[0.01]])
Qz = numpy.diag([ 0.01, 0.1])
z0 = numpy.array([0.0, 1.0, ])
P0 = numpy.eye(2)

def generate_reference(steps):
    states = numpy.zeros((3, steps+1))
    e = numpy.random.uniform(-math.pi/32.0, math.pi/32.0)
    z = numpy.random.multivariate_normal(z0.ravel(), P0).reshape((-1,1))
    C = numpy.array([[1.0, 0.0]])
        
    ylist = []
    
    
    states[0,0] = numpy.copy(e)
    states[1:,0] = numpy.copy(z).ravel()
    
    for i in range(steps):
    
        A = numpy.array([[math.cos(e), -math.sin(e)],
                         [math.sin(e), math.cos(e)]])        
        e = e + numpy.random.uniform(-math.pi/16.0, math.pi/16.0)
        z = A.dot(z) + numpy.random.multivariate_normal(numpy.zeros(len(z)), Qz).reshape((-1,1))
        
        states[0, i+1]=e
        states[1:,i+1]=z.ravel()


        C = numpy.array([[1.0, 0.0]])
        # use the correct particle to generate the true measurement
        y_l = C.dot(z).ravel() + numpy.random.multivariate_normal((0.0,),Rz).reshape((-1,1))
        y_nl = numpy.asarray((e + numpy.random.uniform(0, math.pi/32.0))).reshape((-1,1))
        ylist.append(numpy.vstack((y_l, y_nl)))
    
    return (ylist, states)

class ParticleTest1(Hierarchial):
    """ Implement a simple system by extending the MixedNLGaussian class """
    def __init__(self, e0):
        """ Define all model variables """
        A = numpy.asarray([[math.cos(e0), -math.sin(e0)],
                         [math.sin(e0), math.cos(e0)]]) 

        C = numpy.array([[1.0, 0.0]])

        z =  numpy.copy(z0).reshape((-1,1))
        # Linear states handled by base-class
        super(ParticleTest1,self).__init__(z0=z, P0=P0, e0=e0, Az=A,
                                                C=C, R=Rz, Qz=Qz,)
        
    def prep_update(self, u):
        """ Perform a time update of all states """
        A = numpy.array([[math.cos(self.eta), -math.sin(self.eta)],
                         [math.sin(self.eta), math.cos(self.eta)]]) 
        self.set_dynamics(Az=A)
        return

    def calc_next_eta(self, u, noise):
        return self.eta + noise

    def sample_process_noise(self, u):
        """ Return process noise for input u """
        return numpy.random.uniform(-math.pi/16.0, math.pi/16.0)

    def prep_measure(self, y):
        # No need to update any dynamics, or pre-processing the measurement data
        return y
    
    def next_eta_pdf(self, next_part, u):
        if (next_part.eta > self.eta - math.pi/16.0 and
            next_part.eta < self.eta + math.pi/16.0):
            return -math.log(math.pi/4)
        else:
            return -numpy.Inf

    def split_measure(self, y):
        """ Split measurement into linear/non-liner part """
        y_nl = numpy.asarray(y[1]).reshape((-1,1))
        y_l = numpy.asarray(y[0]).reshape((-1,1))
        return (y_nl, y_l)
    
    def measure_nl(self, y_nl):
        if (y_nl[0,0] > self.eta and
            y_nl[0,0] < self.eta + math.pi/32.0):
            return -math.log(math.pi/32.0)
        else:
            return -numpy.Inf
    
    def fwd_peak_density_eta(self, u):
        return -math.log(math.pi/32.0)