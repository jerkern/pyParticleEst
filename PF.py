""" Particle filtering implementation """

import numpy
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
    """ Particle Filer """
    
    def __init__(self, res = 0, lp_hack = None):
        """ Create particle filter.
        res - 0 or 1 if resampling on or off"""
        
        self.res = res
        self.lp_hack = lp_hack
        
    def update(self, u, pa, inplace=True):
        """ Update particle approximation using u as kinematic input """
        
        u = numpy.reshape(u,(-1,1))
        
        if (not inplace):
            pa_out = copy.deepcopy(pa)
            pa = pa_out
        
        for k in range(pa.num):
            
            un = pa.part[k].sample_input_noise(u)
            pa.part[k].update(un)
            
        return pa 
    
    
    def measure(self, r, pa, inplace=True):
        """ Evaluate and update particle approximation using new measurement r """

        if (not inplace):
            pa_out = copy.deepcopy(pa)
            pa = pa_out

        
        new_weights = numpy.empty(pa.num, numpy.float)
        for k in range(pa.num):
            new_weights[k] = pa.part[k].measure(r)

        if (self.lp_hack == None):
            # Check so that the approximation hasn't degenerated
            s = sum(new_weights)
    #        assert s != 0.0
            
            if (s != 0.0):
                new_weights /= s
                tmp = pa.w * new_weights
                if (sum(tmp) != 0.0):
                    pa.w = tmp
                else:
                    print "Filter has degenerated completely!"
            else:
                print "All particles bad! (should trigger assert, but disabled)"
    


        else:
            # lowpass filter hack work-around, not mathematically correct
            s = sum(new_weights)
            if (s != 0.0):
                new_weights = new_weights/s
                pa.w = (1-self.lp_hack)*pa.w + self.lp_hack*new_weights

        pa.w /= sum(pa.w)
        pa.N_eff = 1 / sum(pa.w ** 2)
        
        resampled = False
        if (self.res and pa.N_eff < self.res*pa.num):
            pa.resample()
            resampled = True
        
        return (pa, resampled)
        

class TrajectoryStep(object):
    def __init__(self, pa, u = None, y = None, t = None):
        self.pa = pa
        # u[t] contains the input for takin x[t] to x[t+1]
        self.u = u
        # y[t] is the measurment of x[t]
        self.y = y
        # t is time
        self.t = t

class ParticleTrajectory(object):
    """ Store particle trajectories """
        
    def __init__(self, pa, resample=2.0/3.0, t0=0, lp_hack = None):
        """ Initialize the trajectory with a ParticleApproximation """
        self.traj = [TrajectoryStep(pa, t=t0),]
        
        self.pf = ParticleFilter(res=resample,lp_hack=lp_hack)
        self.len = 1
        return
    
    def update(self, u):
        pa_nxt = self.pf.update(u, self.traj[-1].pa, False)
        if self.traj[-1].u != None:
            print "Warning, multiple inputs for the same timeindex is not supported!"
        self.traj[-1].u = u
        self.traj.append(TrajectoryStep(pa_nxt, t=self.traj[-1].t+1))
        self.len = len(self.traj)
                
    def measure(self, y):
        if self.traj[-1].y != None:
            print "Warning, multiple measurements for the same timeindex is not supported!"
        
        (pa, resampled) = self.pf.measure(y, self.traj[-1].pa, True)
        self.traj[-1].y = y
        return resampled
        
    def prep_rejection_sampling(self):
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
        return self.traj[-1].pa.N_eff/self.traj[-1].pa.num
    
    def extract_signals(self):
        signals = []
        for k in range(self.len):
            signals.append(TrajectoryStep(pa=None, u=self.traj[k].u, y=self.traj[k].y, t=self.traj[k].t))
            
        return signals

class ParticleApproximation(object):
    """ Contains collection of particles approximating a pdf 
        num - number of particles
        seed - value to initialize all particles with
        kin_var - variance of noise added in the update step """
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
        
        self.w = weights
        self.num = num
        n_eff = 1 / sum(weights ** 2)
        self.N_eff = n_eff
        return
    
    def __len__(self):
        return len(self.part)
    
    def resample(self, N=None):
        """ Resample approximation so all particles have the same weight """
        
        print "Resample! %f/%d (%0.2f%%)" % (self.N_eff, self.num, 100.0*self.N_eff/self.num)
        # To few effective particles, trigger resampling
        if (N  == None):
            N = self.num
        
        new_ind = sample(self.w, N)
        new_part = numpy.empty(N, type(self.part[0]))
        for k in range(numpy.shape(new_ind)[0]):
            new_part[k] = copy.deepcopy(self.part[new_ind[k]])

        self.w = numpy.ones(N, dtype=numpy.float) / N
        self.part = new_part
        self.num = N
        self.N_eff = 1 / sum(self.w ** 2)
        
    def sample(self):
        """ Draw one particle at random with probability corresponding to its weight """
        return self.part[sample(self.w,1)[0]]
    
    def find_best_particles(self, n=1):
        """ Find n-best particles """
        indices = numpy.argsort(self.w)
        return indices[range(n)]

    def calc_fwd_peak_density(self, u):
        coeffs = numpy.empty(self.num, float)
        for k in range(self.num):
            coeffs[k] = self.part[k].fwd_peak_density(u)
            
        return numpy.max(coeffs)

