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
        T = len(pt)
        self.u = numpy.empty(T, dtype=object)
        self.y = numpy.empty(T, dtype=object)
        self.t = numpy.empty(T, dtype=object)
        for i in xrange(T):
            self.u[i] = pt[i].u
            self.y[i] = pt[i].y
            self.t[i] = pt[i].t
        
        self.model =  pt.pf.model
        if (method=='full' or method=='mcmc' or method=='rs' or method=='rsas'):
            self.perform_bsi(pt=pt, M=M, method=method, options=options)
        elif (method=='ancestor'):
            self.perform_ancestors(pt=pt, M=M)
        elif (method=='mhips'):
            self.perform_ancestors(pt=pt, M=M)
            if 'R' in options:
                R = options['R']
            else:
                R = 10
            for _i in xrange(R):
                self.perform_mhips_pass(pt=pt, M=M, options=options)
        elif (method=='bp'):
            if 'R' in options:
                R = options['R']
            else:
                R = 10
            self.perform_bp(pt, M, R)
        else:
            raise ValueError('Unknown smoother: %s' % method)
        
    def __len__(self):
        return len(self.traj)
    
    def perform_ancestors(self, pt, M):
        
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
        
        self.traj = numpy.vstack(self.traj)

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
       
        self.traj = numpy.vstack(self.traj)
        
        if hasattr(self.model, 'post_smoothing'):
            # Do e.g. constrained smoothing for RBPS models
            self.straj = self.model.post_smoothing(self)
        else:
            self.straj = self.traj
    
    def perform_bp(self, pt, M, R):
        
        T = len(pt)
        self.traj = numpy.zeros((T,), dtype=numpy.ndarray)
        
        # Initialise from end time estimates
        tmp = numpy.copy(pt[-1].pa.w)
        tmp -= numpy.max(tmp)
        tmp = numpy.exp(tmp)
        tmp = tmp / numpy.sum(tmp)
        anc = pf.sample(tmp, M)
 
        for t in reversed(xrange(T)):

            # Initialise from filtered estimate
            if (t < T-1):
                self.traj[t] = numpy.copy(self.model.sample_smooth(pt[t].pa.part[anc],
                                                                   self.traj[t+1][0],
                                                                   self.u[t],
                                                                   self.t[t]))[numpy.newaxis,]
            else:
                self.traj[t] = numpy.copy(self.model.sample_smooth(pt[t].pa.part[anc],
                                                                   None,
                                                                   self.u[t],
                                                                   self.t[t]))[numpy.newaxis,]

            if (t > 0):
                anc = pt[t].ancestors[anc]
                tmp = numpy.copy(pt[t-1].pa.w)
                tmp -= numpy.max(tmp)
                tmp = numpy.exp(tmp)
                tmp = tmp / numpy.sum(tmp)

            for _ in xrange(R):

                if (t > 0):
                    # Propose new ancestors
                    panc = pf.sample(tmp, M)
                    partp_prop=pt[t-1].pa.part[panc]
                    partp_curr=pt[t-1].pa.part[anc]
                    up = self.u[t-1]
                    tp=self.t[t-1]
                else:
                    partp_prop = None
                    partp_curr = None
                    up = None
                    tp = None

                if (t < T-1):
                    partn = self.traj[t+1][0]
                else:
                    partn = None

                (pprop, acc) = mc_step(model=self.model,
                                       partp_prop=partp_prop,
                                       partp_curr=partp_curr,
                                       up=up,
                                       tp=tp,
                                       curpart=self.traj[t][0],
                                       y=self.y[t],
                                       u=self.u[t],
                                       t=self.t[t],
                                       partn=partn)
             
                # Update with accepted proposals
                self.traj[t][:1,acc] = pprop[acc][numpy.newaxis,]
                anc[acc] = panc[acc]
                                                                                                
        self.traj = numpy.vstack(self.traj)
         
        if hasattr(self.model, 'post_smoothing'):
            # Do e.g. constrained smoothing for RBPS models
            self.straj = self.model.post_smoothing(self)
        else:
            self.straj = self.traj
    
    def perform_mhips_pass(self, pt, M, options):
        T = len(self.traj)
        for i in reversed(xrange((T))):
            if (i == T-1):
                partp=self.traj[i-1]
                up=self.u[i-1]
                tp=self.t[i-1]
                y=self.y[i]
                u=self.u[i]
                t=self.t[i]
                partn=None
            elif (i == 0):
                partp=None
                up=None
                tp=None
                y=self.y[i]
                u=self.u[i]
                t=self.t[i]
                partn=self.traj[i+1]
            else:
                partp=self.traj[i-1]
                up=self.u[i-1]
                tp=self.t[i-1]
                y=self.y[i]
                u=self.u[i]
                t=self.t[i]
                partn=self.traj[i+1]
                
            (prop, acc) = mc_step(model=self.model, partp_prop=partp, partp_curr=partp, up=up, tp=tp,
                                  curpart=self.traj[i], y=y, u=u, t=t,
                                  partn=partn)
            self.traj[i][acc] = prop[acc]
            
    def perform_mhips_pass_reduced(self, pt, M, options):
        """ Runs MHIPS with the proposal density q and p(x_{t+1}|x_t) """
        T = len(self.traj)
        for i in reversed(xrange((T))):
                
            if (i > 0):
                xprop = numpy.copy(self.traj[i-1])
                noise = self.model.sample_process_noise(xprop, self.u[i-1], self.t[i-1])
                xprop= self.model.update(xprop, self.u[i-1], self.t[i-1], noise)
            else:
                xprop = self.model.create_initial_estimate(M)
            
            if (self.y[i] != None):
                logp_y_prop = self.model.measure(particles=numpy.copy(xprop),
                                                 y=self.y[i],
                                                 t=self.t[i])
                                              
                logp_y_curr = self.model.measure(particles=numpy.copy(self.traj[i]),
                                                 y=self.y[i],
                                                 t=self.t[i])
            else:
                logp_y_prop = numpy.zeros(M)
                logp_y_curr = numpy.zeros(M)
                
            if (i < T-1):
                logp_next_prop = self.model.next_pdf(particles=xprop,
                                                     next_part=self.traj[i+1],
                                                     u=self.u[i],
                                                     t=self.t[i])
                logp_next_curr = self.model.next_pdf(particles=self.traj[i],
                                                     next_part=self.traj[i+1],
                                                     u=self.u[i],
                                                     t=self.t[i])
            else:
                logp_next_prop = numpy.zeros(M)
                logp_next_curr = numpy.zeros(M)                                  
        
 
            # Calc ratio
            ratio = ((logp_y_prop - logp_y_curr) +
                     (logp_next_prop - logp_next_curr))
            
            test = numpy.log(numpy.random.uniform(size=M))
            ind = test < ratio
            self.traj[i][ind] = xprop[ind]


def mc_step(model, partp_prop, partp_curr, up, tp, curpart, y, u, t, partn):
    M = len(curpart)
    xprop = model.propose_smooth(partp=partp_prop, 
                                 up=up,
                                 tp=tp,
                                 y=y,
                                 u=u,
                                 t=t,
                                 partn=partn)
    
    # Accept/reject new sample
    logp_q_prop = model.logp_smooth(xprop,
                                    partp=partp_prop, 
                                    up=up,
                                    tp=tp,
                                    y=y,
                                    u=u,
                                    t=t,
                                    partn=partn)
    logp_q_curr = model.logp_smooth(curpart,
                                    partp=partp_curr, 
                                    up=up,
                                    tp=tp,
                                    y=y,
                                    u=u,
                                    t=t,
                                    partn=partn)

    
    if (partp_prop != None and partp_curr != None):
        logp_prev_prop = model.next_pdf(particles=partp_prop,
                                        next_part=xprop,
                                        u=up,
                                        t=tp)
        logp_prev_curr = model.next_pdf(particles=partp_curr,
                                        next_part=curpart,
                                        u=up,
                                        t=tp)
    else:
        logp_prev_prop = numpy.zeros(M)
        logp_prev_curr = numpy.zeros(M)
                                 
    if (y != None):
        logp_y_prop = model.measure(particles=numpy.copy(xprop),
                                    y=y, t=y)
                                      
        logp_y_curr = model.measure(particles=numpy.copy(curpart),
                                    y=y, t=t)
    else:
        logp_y_prop = numpy.zeros(M)
        logp_y_curr = numpy.zeros(M)
        
    if (partn != None):
        logp_next_prop = model.next_pdf(particles=xprop,
                                        next_part=partn,
                                        u=u,
                                        t=t)
        logp_next_curr = model.next_pdf(particles=curpart,
                                        next_part=partn,
                                        u=u,
                                        t=t)
    else:
        logp_next_prop = numpy.zeros(M)
        logp_next_curr = numpy.zeros(M)                                  
    
    
    # Calc ratio
    ratio = ((logp_prev_prop - logp_prev_curr) + 
             (logp_y_prop - logp_y_curr) +
             (logp_next_prop - logp_next_curr) +
             (logp_q_curr - logp_q_prop))
    
    test = numpy.log(numpy.random.uniform(size=M))
    acc = test < ratio
    return (xprop, acc)   