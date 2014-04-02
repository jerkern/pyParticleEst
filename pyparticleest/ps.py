""" Rao-Blackwellized Particle smoother implementation """

import numpy
import math
import copy
import pyparticleest.pf as pf

def bsi_full(pa, model, next, u):
    M = len(next)
    res = numpy.empty(M, dtype=int)
    for j in xrange(M):
        p_next = model.next_pdf(pa.part, next[j:j+1], u)
            
        w = pa.w + p_next
        w = w - numpy.max(w)
        w_norm = numpy.exp(w)
        w_norm /= numpy.sum(w_norm)
        res[j] = pf.sample(w_norm, 1)
    return res


def bsi_rs(pa, model, next, u, maxpdf, max_iter):
    M = len(next)
    todo = numpy.asarray(range(M))
    res = numpy.empty(M, dtype=int)
    weights = numpy.copy(pa.w)
    weights -= numpy.max(weights)
    weights = numpy.exp(weights)
    weights /= numpy.sum(weights)
    for _i in xrange(max_iter):

        ind = numpy.random.permutation(pf.sample(weights, len(todo)))
        pn = model.next_pdf(pa.part[ind], next[todo], u)
        test = numpy.log(numpy.random.uniform(size=len(todo)))
        accept = test < pn - maxpdf
        res[todo[accept]] = ind[accept]
        todo = todo[~accept]
        if (len(todo) == 0):
            return res
    
    res[todo] = bsi_full(pa, model, next[todo], u)        
    return res

def bsi_mcmc(pa, model, next, u, R, ancestors):
    M = len(next)
    ind = ancestors
    weights = numpy.copy(pa.w)
    weights -= numpy.max(weights)
    weights = numpy.exp(weights)
    weights /= numpy.sum(weights)
    pind = model.next_pdf(pa.part[ind], next, u)                     
    for _j in xrange(R):
        propind = numpy.random.permutation(pf.sample(weights, M))
        pprop = model.next_pdf(pa.part[propind], next, u)
        diff = pprop - pind
        diff[diff > 0.0] = 0.0
        test = numpy.log(numpy.random.uniform(size=M))
        accept = test < diff
        ind[accept] = propind[accept]
        pind[accept] = pprop[accept] 
    
    return ind
        
def bsi_rsas(pa, model, next, u, maxpdf, x1, P1, sv, sw, ratio):
    M = len(next)
    todo = numpy.asarray(range(M))
    res = numpy.empty(M, dtype=int)
    weights = numpy.copy(pa.w)
    weights -= numpy.max(weights)
    weights = numpy.exp(weights)
    weights /= numpy.sum(weights)
    pk = x1
    Pk = P1
    stop_criteria = ratio / len(pa)
    while (True):

        ind = numpy.random.permutation(pf.sample(weights, len(todo)))
        pn = model.next_pdf(pa.part[ind], next[todo], u)
        test = numpy.log(numpy.random.uniform(size=len(todo)))
        accept = test < pn - maxpdf
        ak = numpy.sum(accept)
        mk = len(todo)
        res[todo[accept]] = ind[accept]
        todo = todo[~accept]
        if (len(todo) == 0):
            return res
        # meas update for adaptive stop
        mk2 = mk*mk
        sw2 = sw*sw
        pk = pk + (mk*Pk)/(mk2*Pk+sw2)*(ak-mk*pk)
        Pk = (1-(mk2*Pk)/(mk2*Pk+sw2))*Pk
        # predict
        pk = (1 - ak/mk)*pk
        Pk = (1 - ak/mk)**2*Pk+sv*sv
        if (pk < stop_criteria):
            break
    
    res[todo] = bsi_full(pa, model, next[todo], u)        
    return res

class SmoothTrajectory(object):
    """ Store smoothed trajectory """
    def __init__(self, pt, M=1, method='normal', options=None):
        """ Create smoothed trajectory from filtered trajectory
            pt - particle trajectory to be smoothed
            rej_sampling - Use rejection sampling instead of evaluating all 
                           weights (reduces time complexity to M*N from M*N^2
            """
        
        self.traj = numpy.zeros((len(pt),), dtype=numpy.ndarray)
        self.Mz = None
        self.model =  pt.pf.model
        # Initialise from end time estimates

        tmp = numpy.copy(pt[-1].pa.w)
        tmp -= numpy.max(tmp)
        tmp = numpy.exp(tmp)
        tmp = tmp / numpy.sum(tmp)
        ind = pf.sample(tmp, M)
        self.traj[-1] = self.model.sample_smooth(pt[-1].pa.part[ind], None, None)[numpy.newaxis,]
        
        
        self.u = [pt[-1].u, ]
        self.y = [pt[-1].y, ]
        self.t = [pt[-1].t, ]
        opt = dict()
        if (method=='full'):
            pass
        elif (method=='mcmc'):
            ancestors = pt[-1].ancestors[ind]
        elif (method=='rs'):
            N = len(pt[-1].pa.part)
            max_iter = int(0.1*N)
        elif (method=='rsas'):
            x1 = 1.0
            P1 = 1.0
            sv = 1.0
            sw = 1.0
            ratio = 1.0
        else:
            raise ValueError('Unknown sampler: %s' % method)
        
        
        for cur_ind in reversed(xrange(len(pt)-1)):
            step = pt[cur_ind]
            self.model.t = step.t
            pa = step.pa
            if (method=='rs'):
                ind = bsi_rs(pa, self.model, self.traj[cur_ind+1][0], step.u,
                             options['maxpdf'][cur_ind],max_iter)
            elif (method=='rsas'):
                ind = bsi_rsas(pa, self.model, self.traj[cur_ind+1][0], step.u,
                               options['maxpdf'][cur_ind],x1,P1,sv,sw,ratio)                
            elif (method=='mcmc'):
                ind = bsi_mcmc(pa, self.model, self.traj[cur_ind+1][0], step.u, 
                               options['R'], ancestors)
                ancestors = step.ancestors[ind]
            elif (method=='normal'):
                ind = bsi_full(pa, self.model, self.traj[cur_ind+1][0], step.u)
            # Select 'previous' particle
            self.traj[cur_ind] = numpy.copy(self.model.sample_smooth(pa.part[ind],
                                                                     self.traj[cur_ind+1][0],
                                                                     step.u))[numpy.newaxis,]
            self.u.append(step.u)
            self.y.append(step.y)
            self.t.append(step.t)
        
        self.traj = numpy.vstack(self.traj)
        # Reorder so the list are ordered with increasing time
        self.u.reverse()
        self.y.reverse()
        self.t.reverse()
        
    def __len__(self):
        return len(self.traj)
    
    def constrained_smoothing(self):
        """ Kalman smoothing of the linear states conditioned on the non-linear
            trajetory """
        
        T = self.traj.shape[0]
        M = self.traj.shape[1]
        self.Mz = numpy.empty((T-1,M), dtype=numpy.ndarray)
        particles = self.model.create_initial_estimate(M)
        for j in xrange(M):
            (z0, P0) = self.model.get_rb_initial([self.traj[0][j][0],])
            self.model.set_states(particles[j:j+1], (self.traj[0][j][0],), (z0,), (P0,))
        
        T = len(self.traj)
        self.straj = numpy.empty((T, M), dtype=object)
        
        for i in xrange(T-1):
            self.model.t = self.t[i]
            if (self.y[i] != None):
                self.model.measure(particles, self.y[i])
            for j in xrange(M):
                self.model.meas_xi_next(particles[j:j+1], self.traj[i+1][j][0], self.u[i])
            for j in xrange(M):
                (_xil, zl, Pl) = self.model.get_states(particles[j:j+1])
                self.model.set_states(particles[j:j+1], self.traj[i][j][0], zl, Pl)
            self.straj[i] = particles

            particles = copy.deepcopy(particles)
            for j in xrange(M):
                self.model.cond_predict(particles[j:j+1], self.traj[i+1][j][0], self.u[i])
                (_xil, zl, Pl) = self.model.get_states(particles[j:j+1])
                self.model.set_states(particles[j:j+1], self.traj[i+1][j][0], zl, Pl)
            
        if (self.y[-1] != None):
            self.model.measure(particles, self.y[-1])
        
        
        for j in xrange(M):
            (_xil, zl, Pl) = self.model.get_states(particles[j:j+1])
            self.model.set_states(particles[j:j+1], self.traj[-1][j][0], zl, Pl)
        self.straj[-1] = particles
        
        # Backward smoothing
        for i in reversed(xrange(T-1)):
            self.model.t = self.t[i]
            (xin, zn, Pn) = self.model.get_states(self.straj[i+1])
            (xi, z, P) = self.model.get_states(self.straj[i])
            for j in xrange(M):
                (Al, fl, Ql) = self.model.calc_cond_dynamics(self.straj[i,j:j+1], self.traj[i+1][j], self.u[i])
                (zs, Ps, Ms) = self.model.kf.smooth(z[j], P[j], zn[j], Pn[j], Al[0], fl[0], Ql[0])
                self.model.set_states(self.straj[i][j:j+1], xi[j], (zs,), (Ps,))
                self.Mz[i,j] = Ms

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
