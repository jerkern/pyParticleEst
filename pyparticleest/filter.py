""" Particle filtering implementation """

import numpy
import math
import copy

def sample(w, n):
    """ Return n random indices, where the probability if index
        i is given by w[i].
        
        w - probability weights
        n - number of indices to return """

    wc = numpy.cumsum(w)
    wc /= wc[-1] # Normalize
    u = (range(n)+numpy.random.rand(1))/n
    return numpy.searchsorted(wc, u)


class ParticleFilter(object):
    """ Particle Filer class, creates filter estimates by calling appropriate
        methods in the supplied particle objects and handles resampling when
        a specified threshold is reach """
    
    def __init__(self, model, res = 0):
        """ Create particle filter.
        res - 0 or 1 if resampling on or off """
        
        self.res = res
        self.model = model
    
    def forward(self, pa, u, y, t):
        pa = ParticleApproximation(self.model.copy_ind(pa.part), pa.w)
        
        resampled = False
        if (self.res > 0 and pa.calc_Neff() < self.res*pa.num):
            ancestors = pa.resample(self.model, pa.num)
            resampled = True
        else:
            ancestors = numpy.arange(pa.num,dtype=int)
        
        pa = self.update(pa, u=u, t=t)
        if (y != None):
            pa = self.measure(pa, y=y, t=t+1)
        return (pa, resampled, ancestors)
    
    def update(self, pa, u, t, inplace=True):
        """ Update particle approximation using u as kinematic input.
            
            If inplace=True the particles are update then returned,
            otherwise a new ParticleApproximation is first created
            leaving the original one intact """
        
        u = numpy.reshape(u,(-1,1))
        
        # Prepare the particle for the update, eg. for 
        # mixed linear/non-linear calculate the variables that
        # depend on the current state
#        for k in range(pa.num):
#            pa.part[k].prep_update(u)
        
        if (not inplace):
            pa_out = copy.deepcopy(pa)
            pa = pa_out
            
        v = self.model.sample_process_noise(particles=pa.part, u=u, t=t)
        self.model.update(particles=pa.part, u=u, t=t, noise=v)
        
        return pa 
    
    
    def measure(self, pa, y, t, inplace=True):
        """ Evaluate and update particle approximation using new measurement r
            
            If inplace=True the particles are update then returned,
            otherwise a new ParticleApproximation is first created
            leaving the original one intact """

        if (not inplace):
            pa_out = copy.deepcopy(pa)
            pa = pa_out

        #y = pa.part[k].prep_measure(r)
        new_weights = self.model.measure(particles=pa.part, y=y, t=t)
        
        pa.w = pa.w + new_weights
        # Keep the weights from going to -Inf
        #pa.w -= numpy.max(pa.w)
        
        return pa
        

class AuxiliaryParticleFilter(ParticleFilter):
    """ Auxiliary Particle Filer class, creates filter estimates by calling appropriate
        methods in the supplied particle objects and handles resampling when
        a specified threshold is reach """
    
    def forward(self, pa, u, y, t):
        pa = copy.deepcopy(pa)
        resampled = False
        
        if (y != None):
            l1w =  self.model.eval_1st_stage_weights(pa.part, u, y, t)
            pa.w += l1w
            pa.w -= numpy.max(pa.w)
            
        if (self.res and pa.calc_Neff() < self.res*pa.num):
            ancestors = pa.resample(self.model, pa.num)
            resampled = True
            l1w = l1w[ancestors]
        else:
            ancestors = numpy.arange(pa.num,dtype=int)
        
        pa = self.update(pa, u=u, t=t)
        
        if (y != None):
            pa.w += self.model.measure(particles=pa.part, y=y, t=t+1)
            pa.w -= l1w
            pa.w -= numpy.max(pa.w)
        
        return (pa, resampled, ancestors)
        

class TrajectoryStep(object):
    """ Store particle approximation, input, output and timestamp for
        a single time index in a trajectory """
    def __init__(self, pa, u = None, y = None, t = None, ancestors = None):
        """ u[t] contains the input for takin x[t] to x[t+1]
            y[t] is the measurment of x[t] """
        self.pa = pa
        self.u = u
        self.y = y
        # t is time
        self.t = t
        self.ancestors = ancestors

class ParticleTrajectory(object):
    """ Store particle trajectories, each time instance is saved
        as a TrajectoryStep object """
        
    def __init__(self, model, N, resample=2.0/3.0, t0=0, filter='PF'):
        """ Initialize the trajectory with a ParticleApproximation """
        particles = model.create_initial_estimate(N)
        pa = ParticleApproximation(particles=particles)
        self.traj = [TrajectoryStep(pa, t=t0, ancestors=numpy.arange(N)),]
        
        if (filter == 'PF'):
            self.pf = ParticleFilter(model=model, res=resample)
        elif (filter == 'APF'):
            self.pf = AuxiliaryParticleFilter(model=model, res=resample)
        else:
            raise ValueError('Bad filter type')
        self.len = 1
        return
    
    def forward(self, u, y):
        self.traj[-1].u = u
        (pa_nxt, resampled, ancestors) = self.pf.forward(self.traj[-1].pa, u=u, y=y, t=self.traj[-1].t)
        self.traj.append(TrajectoryStep(pa_nxt, t=self.traj[-1].t+1, y=y, ancestors=ancestors))
        #self.traj[-1].y = y
        self.len = len(self.traj)
        return resampled
    
    def measure(self, y):
        self.traj[-1].y = y
        self.pf.measure(self.traj[-1].pa, y=y, t=self.traj[-1].t, inplace=True)
        
    def prep_rejection_sampling(self):
        """ Find the maximum over all inputs of the pdf for the next timestep,
            used for rejection sampling in the particle smoother """
        for k in range(self.len-1):
            self.traj[k].peak_fwd_density = self.traj[k].pa.calc_fwd_peak_density(self.traj[k].u)
    
    def __len__(self):
        return len(self.traj)
    
    def __getitem__(self, index):
        return self.traj[index]
    
    def spawn(self):
        """ Create new ParticleTrajectory starting at the end of
            the current one """
        return ParticleTrajectory(copy.deepcopy(self.traj[-1].pa), resample=self.pf.res, t0=self.traj[-1].t, lp_hack=self.pf.lp_hack)

    def extract_signals(self):
        """ Throw away the particle approxmation and return a list contaning just
            inputs, output and timestamps """
        signals = []
        for k in range(self.len):
            signals.append(TrajectoryStep(pa=None, u=self.traj[k].u, y=self.traj[k].y, t=self.traj[k].t))
            
        return signals
    
    def perform_smoothing(self, M, method="full", smoother_options=None):
        """ return an array of smoothed trajectories 
            M - number of smoothed trajectories """
        from smoother import SmoothTrajectory
        
        # Calculate coefficients needed for rejection sampling in the backward smoothing
#        if (rej_sampling):
#            self.prep_rejection_sampling()
        options={}
        if (method == 'rs' or method == 'rsas'):
            coeffs = numpy.empty(self.len, dtype=float)
            for k in range(self.len):
                coeffs[k] = self.pf.model.next_pdf_max(particles=self.traj[k].pa.part,
                                                       u=self.traj[k].u, t=self.traj[k].t)
            options['maxpdf'] = coeffs
        if (method == 'mcmc'):
                options['R'] = 30
        if (smoother_options != None):
            options.update(smoother_options)
            
        straj = SmoothTrajectory(self, M=M, method=method, options=options)
            
        return straj


class ParticleApproximation(object):
    """ Contains collection of particles approximating a pdf
        particles - collection of particles
        weights - weight for each particle
        seed - value to initialize all particles with
        num - number of particles
        
        Use either seed and num or particles (and optionally weights
        if not uniform) """
    def __init__(self, particles=None, logw=None, seed=None, num=None):
        if (particles != None):
            self.part = numpy.asarray(particles)
            num = len(particles)
        else:
            self.part = numpy.empty(num, type(seed))
            for k in range(num):
                self.part[k] = copy.deepcopy(seed)
        
        if (logw != None):
            self.w = numpy.copy(logw)
        else:
            self.w = -math.log(num)*numpy.ones(num)
        
        self.num = num

    def __len__(self):
        return len(self.part)
    
    def calc_Neff(self):
        tmp = numpy.exp(self.w - numpy.max(self.w))
        tmp /= numpy.sum(tmp)
        return 1.0 / numpy.sum(numpy.square(tmp))
    
    def resample(self, model, N=None):
        """ Resample approximation so all particles have the same weight,
            new number of particles is N. If left out the number of particles
            remains the same """
        
        if (N  == None):
            N = self.num
        
        tmp = self.w - numpy.max(self.w)
        new_ind = sample(numpy.exp(tmp), N)
        new_part = model.copy_ind(self.part, new_ind)
        
        self.w = numpy.log(numpy.ones(N, dtype=numpy.float) / N)
        self.part = new_part
        self.num = N
        return new_ind
        
    def sample(self):
        """ Draw one particle at random with probability corresponding to its weight """
        return self.part[sample(numpy.exp(self.w),1)[0]]
    
    def find_best_particles(self, n=1):
        """ Find n-best particles """
        indices = numpy.argsort(self.w)
        return indices[range(n)]
