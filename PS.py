""" Rao-Blackwellized Particle smoother implementation """

import numpy
import math
import copy
import PF

class SmoothTrajectory(object):
    """ Store smoothed trajectory """
    def __init__(self, pt, rej_sampling=True):
        """ Create smoothed trajectory from filtered trajectory
            pt - particle trajectory to be smoothed
            rej_sampling - Use rejection sampling instead of evaluating all 
                           weights (reduces time complexity to M*N from M*N^2
            """
        # Initialise from end time estimates
        part = pt[-1].pa.sample()
        # Collapse particle which is conditionally linear gaussian
        # to a single point by sampling the MVN 
        self.traj = numpy.zeros(len(pt), dtype=type(part)) 
        self.traj[-1] = copy.copy(part)
        self.traj[-1].sample_smooth(None)
        self.u = [pt[-1].u, ]
        self.y = [pt[-1].y, ]
        self.t = [pt[-1].t, ]

        for cur_ind in reversed(range(len(pt)-1)):

            step = pt[cur_ind]
            pa = step.pa
            ind = None
            p_next = -numpy.Inf*numpy.ones(numpy.shape(pa.w))
            
            if (rej_sampling and step.peak_fwd_density > 0.0):
                cnt = 0
                w = numpy.exp(pa.w)
                while (cnt < len(p_next)/2.0): # Time-out limit
                    tmp_ind = PF.sample(w,1)[0]
                    p = pa.part[tmp_ind].next_pdf(self.traj[cur_ind+1],
                                                  step.u)
                    accept = numpy.random.uniform()
                    p_acc = p - step.peak_fwd_density
                        
                    if (math.log(accept) <= p_acc):
                        ind = tmp_ind
                        break
                    else:
                        p_next[tmp_ind] = p
                    cnt += 1
            
            
            if (ind == None): # No index found, do exhaustive search
                w = numpy.copy(pa.w)
                for i in range(len(pa)):
                    if (p_next[i] == -numpy.Inf):
                        p_next[i] = pa.part[i].next_pdf(self.traj[cur_ind+1],
                                                        step.u)
                    
                w += p_next
                # Normalize
                try:
                    w_norm = numpy.exp(w)
                    w_norm /= numpy.sum(w_norm)
                except FloatingPointError:
                    # All weights zero, for robustness just draw a random sample
                    w_norm  = (0.0*w + 1.0)/len(w)
                     
                ind = PF.sample(w_norm,1)[0]

            # Select 'previous' particle
            prev_part = pa.part[ind]
            p_index = pt.traj.index(step)           
            # Sample smoothed linear estimate
            self.traj[cur_ind]=copy.deepcopy(prev_part)
            self.traj[cur_ind].sample_smooth(self.traj[cur_ind+1])
            self.u.append(step.u)
            self.y.append(step.y)
            self.t.append(step.t)
            
        # Reorder so the list are ordered with increasing time
        self.u.reverse()
        self.y.reverse()
        self.t.reverse()
        
    def __len__(self):
        return len(self.traj)
    
    def constrained_smoothing(self, z0, P0):
        
        self.traj[0].set_lin_est((numpy.copy(z0), numpy.copy(P0)))

        for i in range(len(self.traj)-1):
            
            if (self.y[i] != None):
                # Estimate z_t given information about eta_{t+1}
                y = self.traj[i].prep_measure(self.y[i])
                self.traj[i].clin_measure(y=y, next_part=self.traj[i+1])
            
            # Predict z_{t+1} given information about eta_{t+1}
            self.traj[i].prep_update(self.u[i])
            tmp = self.traj[i].clin_predict(self.traj[i+1])
            self.traj[i+1].set_lin_est(tmp)
        
        if (self.y[-1] != None):
            y = self.traj[-1].prep_measure(self.y[-1])
            self.traj[-1].clin_measure(y, next_part=None)

        # Backward smoothing
#        for i in reversed(range(len(self.traj)-1)):
#            self.traj[i].clin_smooth(self.traj[i+1])



def do_smoothing(pt, M, rej_sampling=True):
    """ return an array of smoothed trajectories 
        M - number of smoothed trajectories """
    
    # Calculate coefficients needed for rejection sampling in the backward smoothing
    if (rej_sampling):
        pt.prep_rejection_sampling()
    
    straj = numpy.empty(M, SmoothTrajectory)
    print "smoothing"
    for i in range(M):
        print "%d/%d" % (i,M)
        straj[i] = SmoothTrajectory(pt, rej_sampling=rej_sampling)
        
    return straj


def extract_smooth_approx(straj, ind):
    """ Create particle approximation from collection of trajectories """
    
    part = numpy.empty(len(straj), type(straj[0].traj[ind]))
    
    print "extract smooth: num_traj=%d, ind=%d" % (len(straj), ind)
    
    for i in range(len(straj)):
        part[i] = copy.deepcopy(straj[i].traj[ind])
        
    pa = PF.ParticleApproximation(particles=part)
    
    return pa
        
    
def replay(pt, signals, ind, callback=None):
    """ Run a particle filter with signals extracted from another filter,
        useful to e.g. have one filter which (partially) overlapps with another
        """
    print "len(pt)=%d. len(signals)=%d, ind=%d" % (len(pt), len(signals), ind)
    if (ind == len(signals)-1):
        # Nothing to do, all measurements already used
        return
    
    # The starting index has already incorporated the measurement from the smoothing.
    pt.update(signals[ind].u)
    for i in range(ind+1, len(signals)):
        print i
        if (signals[i].y):
            pt.measure(signals[i].y)
            if (callback != None):
                callback(pt)
        if (signals[i].u):
            pt.update(signals[i].u)
            if (callback != None):
                callback(pt)
