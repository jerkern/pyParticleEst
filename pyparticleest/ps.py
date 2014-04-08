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
        
        if hasattr(self.model, 'post_smoothing'):
            # Do e.g. constrained smoothing for RBPS models
            self.straj = self.model.post_smoothing(self)
        else:
            self.straj = self.traj
            
        
    def __len__(self):
        return len(self.traj)
    
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
