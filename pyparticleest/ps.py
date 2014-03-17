""" Rao-Blackwellized Particle smoother implementation """

import numpy
import math
import copy
import pyparticleest.pf as pf

def bsi_full(pa, model, next, u, opt):
    p_next = model.next_pdf(pa.part, next, u)
            
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
            p_next[ind[i]] = model.next_pdf(pa.part[ind[i]:ind[i]+1], next, u)
        if test[i] < p_next[ind[i]]/maxpdf:
            # Accept sample
            return ind[i]
    if (not_tested.any()):
        p_next[not_tested] = model.next_pdf(pa.part[not_tested], next, u)
   
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
        self.model =  pt.pf.model
        self.traj[-1] = self.model.sample_smooth(numpy.asarray([part,]), None, None)
        
        
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
            ind = sampler(pa, self.model, self.traj[cur_ind+1], step.u, opt=opt)
            # Select 'previous' particle
            self.traj[cur_ind] = self.model.sample_smooth(pa.part[ind:(ind+1)], self.traj[cur_ind+1], step.u)
            self.u.append(step.u)
            self.y.append(step.y)
            self.t.append(step.t)
            
        # Reorder so the list are ordered with increasing time
        self.u.reverse()
        self.y.reverse()
        self.t.reverse()
        
    def __len__(self):
        return len(self.traj)
    
    def constrained_smoothing(self):
        """ Kalman smoothing of the linear states conditione on the non-linear
            trajetory """
        (z0, P0) = self.model.get_rb_initial([self.traj[0][0],])
        particles = self.model.create_initial_estimate(1)
        self.model.set_states(particles, self.traj[0][0], z0, P0)
        self.straj = list()
        T = len(self.traj)
        for i in xrange(T-1):
            if (self.y[i] != None):
                self.model.measure(particles, self.y[i])
            self.model.meas_xi_next(particles, self.traj[i+1][0], self.u[i])
            (_xil, zl, Pl) = self.model.get_states(particles)
            self.model.set_states(particles, self.traj[i][0], zl, Pl)
            self.straj.append(particles)

            particles = copy.deepcopy(particles)
            self.model.cond_predict(particles, self.traj[i+1][0], self.u[i])
            (_xil, zl, Pl) = self.model.get_states(particles)
            self.model.set_states(particles, self.traj[i+1][0], zl, Pl)
            
        if (self.y[-1] != None):
            self.model.measure(particles, self.y[-1])
        (_xil, zl, Pl) = self.model.get_states(particles)
        self.model.set_states(particles, self.traj[-1][0], zl, Pl)
        self.straj.append(particles)
        
        # Backward smoothing
        for i in reversed(xrange(len(self.traj)-1)):
            (xin, zn, Pn) = self.model.get_states(self.straj[i+1])
            (xi, z, P) = self.model.get_states(self.straj[i])
            (Al, fl, Ql) = self.model.calc_cond_dynamics(self.straj[i], self.traj[i+1], self.u[i])
            (zs, Ps, Ms) = self.model.kf.smooth(z[0], P[0], zn[0], Pn[0], Al[0], fl[0], Ql[0])
            self.model.set_states(self.straj[i], xi, zs, Ps)

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
