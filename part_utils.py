""" Collection of functions and classes used for Particle Filtering/Smoothing """


class ParticleFilteringBase(object):
    """ Base class for particles to be used with particle filtering """
    def sample_input_noise(self, u):
        """ Return a noise perturbed input vector u """
        raise NotImplementedError( "Should have implemented this" )
    
    def update(self, data):
        raise NotImplementedError( "Should have implemented this" )
        
    def measure(self, y):
        raise NotImplementedError( "Should have implemented this" )



class ParticleSmoothingBase(ParticleFilteringBase):
    """ Base class for particles to be used with particle smoothing """
    def next_pdf(self, next_cpart, u):
        """ Return the probability density value for the possible future state 'next' given input u """
        raise NotImplementedError( "Should have implemented this" )
    
    def sample_smooth(self, filt_traj, ind, next_cpart):
        """ Return a collapsed particle with the rao-blackwellized states sampled """
        raise NotImplementedError( "Should have implemented this" )
    
    def collapse(self):
        """ Return a sample of the particle where the rao-blackwellized states
        are drawn from the MVN that results from CLGSS structure """
        
        raise NotImplementedError( "Should have implemented this" )
    
class ParticleSmoothingBaseRB(ParticleSmoothingBase):
    
    def clin_update(self, u):
        """ Kalman update of the linear states conditioned on the non-linear trajectory estimate """
        raise NotImplementedError( "Should have implemented this" )
    
    def clin_measure(self, y):
        """ Kalman measuement of the linear states conditioned on the non-linear trajectory estimate """
        raise NotImplementedError( "Should have implemented this" )

    def clin_smooth(self, z_next, u):
        """ Kalman smoothing of the linear states conditioned on the next particles linear states """ 
        raise NotImplementedError( "Should have implemented this" )

    def set_nonlin_state(self, eta):
        """ Set the non-linear state estimates """
        raise NotImplementedError( "Should have implemented this" )
    
    def get_nonlin_state(self):
        """ Return the non-linear state estimates """
        raise NotImplementedError( "Should have implemented this" )
    
    def set_lin_est(self, lest):
        """ Set the estimate of the rao-blackwellized states """
        raise NotImplementedError( "Should have implemented this" )
 
    def get_lin_est(self):
        """ Return the estimate of the rao-blackwellized states """
        raise NotImplementedError( "Should have implemented this" )
 
    def linear_input(self, u):
        """ Extract the part of u affect the conditionally rao-blackwellized states """
        raise NotImplementedError( "Should have implemented this" )    
    
    

