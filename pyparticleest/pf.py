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
    new_ind = numpy.zeros(n, dtype=numpy.int)
    for k in range(n):
        # First index in wc that is greater than u[k]
        new_ind[k] = numpy.argmax(wc > u[k])
        
    return new_ind  


class ParticleFilter(object):
    """ Particle Filer class, creates filter estimates by calling appropriate
        methods in the supplied particle objects and handles resampling when
        a specified threshold is reach """
    
    def __init__(self, model, res = 0):
        """ Create particle filter.
        res - 0 or 1 if resampling on or off
        lp_hack - switches to a (mathematically incorrect) mode of lowpass
                filtering weights instead of multiplying them """
        
        self.res = res
        self.model = model
    
    def forward(self, u, y, pa):
        pa = copy.deepcopy(pa)
        resampled = False
        if (self.res and pa.N_eff < self.res*pa.num):
            pa.resample()
            resampled = True
        
        pa = self.update(u, pa)
        pa = self.measure(y, pa)
        return (pa, resampled)
    
    def update(self, u, pa, inplace=True):
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
            
        v = self.model.sample_process_noise(u=u, particles=pa.part)
        self.model.update(u=u, noise=v, particles=pa.part)
        
        return pa 
    
    
    def measure(self, r, pa, inplace=True):
        """ Evaluate and update particle approximation using new measurement r
            
            If inplace=True the particles are update then returned,
            otherwise a new ParticleApproximation is first created
            leaving the original one intact """

        if (not inplace):
            pa_out = copy.deepcopy(pa)
            pa = pa_out

        #y = pa.part[k].prep_measure(r)
        new_weights = self.model.measure(y=r, particles=pa.part)
        
        pa.w = pa.w + new_weights
        # Keep the weights from going to -Inf
        pa.w -= numpy.max(pa.w)
        
        # Calc N_eff
        w = numpy.exp(pa.w)
        w /= sum(w)
        pa.N_eff = 1 / sum(w ** 2)
        
        return pa
        

class AuxiliaryParticleFilter(object):
    """ Auxiliary Particle Filer class, creates filter estimates by calling appropriate
        methods in the supplied particle objects and handles resampling when
        a specified threshold is reach """
    
    def __init__(self, res):
        """ Create particle filter.
        res - 0 or 1 if resampling on or off """
        self.res = res
    
    def forward(self, u, y, pa):
        """ Update particle approximation using u as kinematic input.
            
            If inplace=True the particles are update then returned,
            otherwise a new ParticleApproximation is first created
            leaving the original one intact """
        
        u = numpy.reshape(u,(-1,1))
        pa_old = pa
        pa = copy.deepcopy(pa)
        resampled = False
        
        # Prepare the particle for the update, eg. for 
        # mixed linear/non-linear calculate the variables that
        # depend on the current state
        for k in range(pa.num):
            pa_old.part[k].prep_update(u)
            pa.part[k].prep_update(u)
        
        l1w = numpy.zeros((pa.num,))
        
        for k in range(pa.num):
            tmp = copy.deepcopy(pa.part[k])
            l1w[k] =  tmp.eval_1st_stage_weight(u,y)
            
            
        # Keep the weights from going to -Inf
        pa.w = pa.w + l1w
        pa.w -= numpy.max(pa.w)
        w = numpy.exp(pa.w)
        w /= sum(w)
        
        pa.N_eff = 1 / sum(w ** 2)
        
        if (self.res and pa.N_eff < self.res*pa.num):
            new_ind = pa.resample()
            l1w = l1w[new_ind]
            resampled = True
      
        for k in range(pa.num):
            v = pa.part[k].sample_process_noise(u)
            pa.part[k].update(u, v)
        
        l2w = numpy.zeros((pa.num,))
        for k in range(pa.num):
            yp = pa.part[k].prep_measure(y)
            l2w[k] = pa.part[k].measure(yp)
            
        pa.w = pa.w + l2w - l1w
        pa.w -= numpy.max(pa.w)
        return (pa, resampled)
    
        

class TrajectoryStep(object):
    """ Store particle approximation, input, output and timestamp for
        a single time index in a trajectory """
    def __init__(self, pa, u = None, y = None, t = None):
        """ u[t] contains the input for takin x[t] to x[t+1]
            y[t] is the measurment of x[t] """
        self.pa = pa
        self.u = u
        self.y = y
        # t is time
        self.t = t

class ParticleTrajectory(object):
    """ Store particle trajectories, each time instance is saved
        as a TrajectoryStep object """
        
    def __init__(self, model, N, resample=2.0/3.0, t0=0, filter='PF'):
        """ Initialize the trajectory with a ParticleApproximation """
        particles = model.create_initial_estimate(N)
        pa = ParticleApproximation(particles=particles)
        self.traj = [TrajectoryStep(pa, t=t0),]
        
        self.pf = ParticleFilter(model=model, res=resample)
#        if (filter == 'PF'):
#            self.pf = ParticleFilter(model=model, res=resample)
#        elif (filter == 'APF'):
#            self.pf = AuxiliaryParticleFilter(res=resample)
#        else:
#            raise ValueError('Bad filter type')
        self.len = 1
        return
    
    def forward(self, u, y):
        self.traj[-1].u = u
        (pa_nxt, resampled) = self.pf.forward(u, y, self.traj[-1].pa)
        self.traj.append(TrajectoryStep(pa_nxt, t=self.traj[-1].t+1))
        self.traj[-1].y = y
        self.len = len(self.traj)
        return resampled
        
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

    def efficiency(self):
        """ Calculate the ratio of effective particles / total particles """
        return self.traj[-1].pa.N_eff/self.traj[-1].pa.num
    
    def extract_signals(self):
        """ Throw away the particle approxmation and return a list contaning just
            inputs, output and timestamps """
        signals = []
        for k in range(self.len):
            signals.append(TrajectoryStep(pa=None, u=self.traj[k].u, y=self.traj[k].y, t=self.traj[k].t))
            
        return signals
    
    def perform_smoothing(self, M, method="normal"):
        """ return an array of smoothed trajectories 
            M - number of smoothed trajectories """
        from pyparticleest.ps import SmoothTrajectory
        
        # Calculate coefficients needed for rejection sampling in the backward smoothing
#        if (rej_sampling):
#            self.prep_rejection_sampling()
        options={}
        if (method == 'rs'):
            coeffs = numpy.empty(self.len, dtype=float)
            for k in range(self.len):
                coeffs[k] = math.exp(self.pf.model.next_pdf_max(particles=self.traj[k].pa.part,
                                                                u=self.traj[k].u)) 
            options['maxpdf'] = coeffs
        if (method == 'mh'):
            options['R'] = 30
            
        straj = numpy.empty(M, SmoothTrajectory)
        #print "smoothing"
        for i in range(M):
            #print "%d/%d" % (i,M)
            straj[i] = SmoothTrajectory(self, method=method, options=options)
            
        return straj

class ParticleApproximation(object):
    """ Contains collection of particles approximating a pdf
        particles - collection of particles
        weights - weight for each particle
        seed - value to initialize all particles with
        num - number of particles
        
        Use either seed and num or particles (and optionally weights
        if not uniform) """
    def __init__(self, particles=None, weights=None, seed=None, num=None):
        if (particles != None):
            self.part = numpy.asarray(particles)
            num = len(particles)
        else:
            self.part = numpy.empty(num, type(seed))
            for k in range(num):
                self.part[k] = copy.deepcopy(seed)
        
        if (weights != None):
            weights = numpy.asarray(weights)
            weights /= sum(weights)
        else:
            weights = numpy.ones(num, numpy.float) / num
        
        self.w = numpy.log(weights)
        self.num = num
        n_eff = 1 / sum(weights ** 2)
        self.N_eff = n_eff
        return
    
    def __len__(self):
        return len(self.part)
    
    def resample(self, N=None):
        """ Resample approximation so all particles have the same weight,
            new number of particles is N. If left out the number of particles
            remains the same """
        
        #print "Resample! %f/%d (%0.2f%%)" % (self.N_eff, self.num, 100.0*self.N_eff/self.num)
        # To few effective particles, trigger resampling
        if (N  == None):
            N = self.num
        
        new_ind = sample(numpy.exp(self.w), N)
        new_part = numpy.empty(N, type(self.part[0]))
        for k in range(numpy.shape(new_ind)[0]):
            new_part[k] = copy.copy(self.part[new_ind[k]])
        
        self.w = numpy.log(numpy.ones(N, dtype=numpy.float) / N)
        self.part = new_part
        self.num = N
        self.N_eff = 1 / sum(numpy.exp(self.w) ** 2)
        return new_ind
        
    def sample(self):
        """ Draw one particle at random with probability corresponding to its weight """
        return self.part[sample(numpy.exp(self.w),1)[0]]
    
    def find_best_particles(self, n=1):
        """ Find n-best particles """
        indices = numpy.argsort(self.w)
        return indices[range(n)]
