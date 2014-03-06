""" Rao-Blackwellized Particle smoother implementation """

import numpy
import math
import copy
import pyparticleest.pf as pf

def bsi_full(pa, model, next, u, opt):
    p_next = model.next_pdf(next, u, pa.part)
            
    w = pa.w + p_next
    w = w - numpy.max(w)
    w_norm = numpy.exp(w)
    w_norm /= numpy.sum(w_norm)
    return pf.sample(w_norm, 1)


def rs_sampler(pa, model, next, u, opt):
    maxpdf = opt['maxpdf']
    max_iter = opt['max_iter']
    ind = pf.sample(pa.w, max_iter)
    test = numpy.log(numpy.random.uniform(size=max_iter))
    N = len(pa.part)
    p_next = numpy.empty(N, dtype=float)
    not_tested = numpy.ones(N, dtype=bool)
    for i in xrange(len(ind)):
        if (not_tested[ind[i]]):
            not_tested[ind[i]] = False
            p_next[ind[i]] = model.next_pdf(next, u, pa.part[ind[i]:ind[i]+1])
        if test[i] < p_next[ind[i]]/maxpdf:
            # Accept sample
            return ind[i]
    if (not_tested.any()):
        p_next[not_tested] = model.next_pdf(next, u, pa.part[not_tested])
   
    w = pa.w + p_next
    w = w - numpy.max(w)
    w_norm = numpy.exp(w)
    w_norm /= numpy.sum(w_norm)
    return pf.sample(w_norm, 1)
        
        

class SmoothTrajectory(object):
    """ Store smoothed trajectory """
    def __init__(self, pt, method='normal', options=None):
        """ Create smoothed trajectory from filtered trajectory
            pt - particle trajectory to be smoothed
            rej_sampling - Use rejection sampling instead of evaluating all 
                           weights (reduces time complexity to M*N from M*N^2
            """
        # Initialise from end time estimates
        part = pt[-1].pa.sample()
        # Collapse particle which is conditionally linear gaussian
        # to a single point by sampling the MVN 
        self.traj = numpy.zeros(len(pt), dtype=list) 
        self.traj[-1] = pt.pf.model.sample_smooth(None, None, numpy.asarray([part,]))
        
        self.u = [pt[-1].u, ]
        self.y = [pt[-1].y, ]
        self.t = [pt[-1].t, ]
        opt = dict()
        if (method=='normal'):
            sampler = bsi_full
        elif (method=='mh'):
            R = options['R']
            pass
        elif (method=='rs'):
            N = len(pt[-1].pa.part)
            opt['max_iter'] = int(0.67*N)
            sampler = rs_sampler
        else:
            raise ValueError('Unknown sampler: %s' % method)
        
        for cur_ind in reversed(range(len(pt)-1)):

            step = pt[cur_ind]
            pa = step.pa
            if (method=='rs'):
                opt['maxpdf'] = options['maxpdf'][cur_ind]
            ind = sampler(pa, pt.pf.model, self.traj[cur_ind+1], step.u, opt=opt)
            # Select 'previous' particle
            self.traj[cur_ind] = pt.pf.model.sample_smooth(self.traj[cur_ind+1], step.u, pa.part[ind:(ind+1)])
            self.u.append(step.u)
            self.y.append(step.y)
            self.t.append(step.t)
            
        # Reorder so the list are ordered with increasing time
        self.u.reverse()
        self.y.reverse()
        self.t.reverse()
        
    def __len__(self):
        return len(self.traj)
    
#    def constrained_smoothing(self, z0, P0):
#        """ Kalman smoothing of the linear states conditione on the non-linear
#            trajetory """
#        
#        self.traj[0].set_lin_est((numpy.copy(z0), numpy.copy(P0)))
#
#        for i in range(len(self.traj)-1):
#            
#            if (self.y[i] != None):
#                # Estimate z_t
#                y = self.traj[i].prep_measure(self.y[i])
#                self.traj[i].clin_measure(y=numpy.asarray(y).reshape((-1,1)), next_part=self.traj[i+1])
#            
#            # Update z_t and dynamics given information about eta_{t+1}
#            etan = self.traj[i+1].get_nonlin_state()
#            self.traj[i].meas_eta_next(etan)
## Cond dynamics are already calculated during the sample_smooth step
##            # Predict z_{t+1} given information about eta_{t+1}, save
##            # conditional dynamics for later use in the smoothing step
##            #self.traj[i].cond_dynamics(etan)
#            lin_est = self.traj[i].kf.predict()
#            self.traj[i+1].set_lin_est(lin_est)
#
#        
#        if (self.y[-1] != None):
#            y = self.traj[-1].prep_measure(self.y[-1])
#            self.traj[-1].clin_measure(numpy.asarray(y).reshape((-1,1)), next_part=None)
#
#        # Backward smoothing
#        for i in reversed(range(len(self.traj)-1)):
#            self.traj[i].clin_smooth(self.traj[i+1])
#
#
#
#def extract_smooth_approx(straj, ind):
#    """ Create particle approximation from collection of trajectories """
#    
#    part = numpy.empty(len(straj), type(straj[0].traj[ind]))
#    
#    print "extract smooth: num_traj=%d, ind=%d" % (len(straj), ind)
#    
#    for i in range(len(straj)):
#        part[i] = copy.deepcopy(straj[i].traj[ind])
#        
#    pa = pf.ParticleApproximation(particles=part)
#    
#    return pa
#        
#    
#def replay(pt, signals, ind, callback=None):
#    """ Run a particle filter with signals extracted from another filter,
#        useful to e.g. have one filter which (partially) overlapps with another
#        """
#    print "len(pt)=%d. len(signals)=%d, ind=%d" % (len(pt), len(signals), ind)
#    if (ind == len(signals)-1):
#        # Nothing to do, all measurements already used
#        return
#    
#    # The starting index has already incorporated the measurement from the smoothing.
#    pt.update(signals[ind].u)
#    for i in range(ind+1, len(signals)):
#        print i
#        if (signals[i].y):
#            pt.measure(signals[i].y)
#            if (callback != None):
#                callback(pt)
#        if (signals[i].u):
#            pt.update(signals[i].u)
#            if (callback != None):
#                callback(pt)
