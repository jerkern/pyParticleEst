""" Rao-Blackwellized Particle smoother implementation """

import numpy
import math
import copy
import pyparticleest.pf as pf

def bsi_full(pa, model, next, u, t):
    M = len(next)
    res = numpy.empty(M, dtype=int)
    for j in xrange(M):
        p_next = model.next_pdf(pa.part, next[j:j+1], u, t)
            
        w = pa.w + p_next
        w = w - numpy.max(w)
        w_norm = numpy.exp(w)
        w_norm /= numpy.sum(w_norm)
        res[j] = pf.sample(w_norm, 1)
    return res


def bsi_rs(pa, model, next, u, t, maxpdf, max_iter):
    M = len(next)
    todo = numpy.asarray(range(M))
    res = numpy.empty(M, dtype=int)
    weights = numpy.copy(pa.w)
    weights -= numpy.max(weights)
    weights = numpy.exp(weights)
    weights /= numpy.sum(weights)
    for _i in xrange(max_iter):

        ind = numpy.random.permutation(pf.sample(weights, len(todo)))
        pn = model.next_pdf(pa.part[ind], next[todo], u, t)
        test = numpy.log(numpy.random.uniform(size=len(todo)))
        accept = test < pn - maxpdf
        res[todo[accept]] = ind[accept]
        todo = todo[~accept]
        if (len(todo) == 0):
            return res
    
    res[todo] = bsi_full(pa, model, next[todo], u)        
    return res

def bsi_mcmc(pa, model, next, u, t, R, ancestors):
    M = len(next)
    ind = ancestors
    weights = numpy.copy(pa.w)
    weights -= numpy.max(weights)
    weights = numpy.exp(weights)
    weights /= numpy.sum(weights)
    pind = model.next_pdf(pa.part[ind], next, u, t)                     
    for _j in xrange(R):
        propind = numpy.random.permutation(pf.sample(weights, M))
        pprop = model.next_pdf(pa.part[propind], next, u, t)
        diff = pprop - pind
        diff[diff > 0.0] = 0.0
        test = numpy.log(numpy.random.uniform(size=M))
        accept = test < diff
        ind[accept] = propind[accept]
        pind[accept] = pprop[accept] 
    
    return ind
        
def bsi_rsas(pa, model, next, u, t, maxpdf, x1, P1, sv, sw, ratio):
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
        pn = model.next_pdf(pa.part[ind], next[todo], u, t)
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
    
    res[todo] = bsi_full(pa, model, next[todo], u, t)        
    return res

class SmoothTrajectory(object):
    """ Store smoothed trajectory """
    def __init__(self, pt, M=1, method='normal', options=None):
        """ Create smoothed trajectory from filtered trajectory
            pt - particle trajectory to be smoothed  """
        
        self.traj = None
        self.straj = None
        self.u = None
        self.y = None
        self.t = None
        
        self.model =  pt.pf.model
        if (method=='full' or method=='mcmc' or method=='rs' or method=='rsas'):
            self.perform_bsi(pt=pt, M=M, method=method, options=options)
        elif (method=='ancestor'):
            self.perform_ancestors(pt=pt, M=M)
        elif (method=='mhips'):
            pass
        elif (method=='bp'):
            pass
        else:
            raise ValueError('Unknown smoother: %s' % method)
        
    def __len__(self):
        return len(self.traj)
    
    def perform_ancestors(self, pt, M):
        
        self.u = [pt[-1].u, ]
        self.y = [pt[-1].y, ]
        self.t = [pt[-1].t, ]
        self.traj = numpy.zeros((len(pt),), dtype=numpy.ndarray)
        
        tmp = numpy.copy(pt[-1].pa.w)
        tmp -= numpy.max(tmp)
        tmp = numpy.exp(tmp)
        tmp = tmp / numpy.sum(tmp)
        ind = pf.sample(tmp, M)
        self.traj[-1] = self.model.sample_smooth(pt[-1].pa.part[ind],
                                                 next_part=None,
                                                 u=pt[-1].u,
                                                 t=pt[-1].t)[numpy.newaxis,]
        ancestors = pt[-1].ancestors[ind]
        
        for cur_ind in reversed(xrange(len(pt)-1)):
            step = pt[cur_ind]
            self.model.t = step.t
            pa = step.pa
            
            ind = ancestors
            ancestors = step.ancestors[ind]
            # Select 'previous' particle
            self.traj[cur_ind] = numpy.copy(self.model.sample_smooth(pa.part[ind],
                                                                     self.traj[cur_ind+1][0],
                                                                     step.u,
                                                                     step.t))[numpy.newaxis,]
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
        
    
    def perform_bsi(self, pt, M, method, options):
        self.traj = numpy.zeros((len(pt),), dtype=numpy.ndarray)
        # Initialise from end time estimates

        tmp = numpy.copy(pt[-1].pa.w)
        tmp -= numpy.max(tmp)
        tmp = numpy.exp(tmp)
        tmp = tmp / numpy.sum(tmp)
        ind = pf.sample(tmp, M)
        self.traj[-1] = self.model.sample_smooth(pt[-1].pa.part[ind],
                                                 next_part=None,
                                                 u=pt[-1].u,
                                                 t=pt[-1].t)[numpy.newaxis,]
        
        
        self.u = [pt[-1].u, ]
        self.y = [pt[-1].y, ]
        self.t = [pt[-1].t, ]
        opt = dict()
        if (method=='full'):
            pass
        elif (method=='mcmc' or method=='ancestor' or method=='mhips'):
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
                ind = bsi_rs(pa, self.model, self.traj[cur_ind+1][0],
                             step.u, step.t,
                             options['maxpdf'][cur_ind],max_iter)
            elif (method=='rsas'):
                ind = bsi_rsas(pa, self.model, self.traj[cur_ind+1][0],
                               step.u, step.t,
                               options['maxpdf'][cur_ind],x1,P1,sv,sw,ratio)                
            elif (method=='mcmc'):
                ind = bsi_mcmc(pa, self.model, self.traj[cur_ind+1][0],
                               step.u, step.t,
                               options['R'], ancestors)
                ancestors = step.ancestors[ind]
            elif (method=='full'):
                ind = bsi_full(pa, self.model, self.traj[cur_ind+1][0], step.u, step.t)
            elif (method=='ancestor' or method=='bp'):
                ind = ancestors
                ancestors = step.ancestors[ind]
            # Select 'previous' particle
            self.traj[cur_ind] = numpy.copy(self.model.sample_smooth(pa.part[ind],
                                                                     self.traj[cur_ind+1][0],
                                                                     step.u,
                                                                     step.t))[numpy.newaxis,]
            self.u.append(step.u)
            self.y.append(step.y)
            self.t.append(step.t)
        
        self.traj = numpy.vstack(self.traj)
        # Reorder so the list are ordered with increasing time
        self.u.reverse()
        self.y.reverse()
        self.t.reverse()
        
        if (method=='mhips' and len(self.traj) > 1):
            # The trajectories are initialised using the forward filters,
            # now run the backard proposer using these as initial input
            iter = options['mhips_iter']
            for i in xrange(iter):
                self.perform_mhips()
        
        if hasattr(self.model, 'post_smoothing'):
            # Do e.g. constrained smoothing for RBPS models
            self.straj = self.model.post_smoothing(self)
        else:
            self.straj = self.traj
    
    def perform_bp(self):
        T = len(self.traj)
        self.traj[T-1] = self.model.propose_smooth(self.traj[T-2],
                                                   self.u[T-2],
                                                   self.u[T-1],
                                                   self.y[T-1],
                                                   None)
        for i in reversed(xrange((1, T-1))):
            self.traj[i] = self.model.propose_smooth(self.traj[i-1],
                                                     self.u[i-1],
                                                     self.u[i],
                                                     self.y[i],
                                                     self.traj[i+1])
        self.traj[0] = self.model.propose_smooth(None,
                                                 None,
                                                 self.u[0],
                                                 self.y[0],
                                                 self.traj[1])

