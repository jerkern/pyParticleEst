""" Rao-Blackwellized Particle smoother implementation """

import numpy
import copy
import PF

class SmoothTrajectory(object):
    """ Store smoothed trajectory """
    def __init__(self, pt, rej_sampling=True, callback=None):
        """ Create smoothed trajectory from filtered trajectory """ 
        # Initialise from end time estimates
        part = pt[-1].pa.sample()
        # Collapse particle which is conditionally linear gaussian
        # to a single point by sampling the MVN 
        spart = part.collapse()
        self.traj = [spart, ]
        self.u = [pt[-1].u, ]
        self.y = [pt[-1].y, ]
        self.t = [pt[-1].t, ]

        cur_ind = len(pt)

        for step in reversed(pt[:-1]):
            cur_ind -= 1
#            print "%d/%d" %(cur_ind, len(pt)-1)
            pa = step.pa
            ind = None
            p_next = -1.0*numpy.ones(numpy.shape(pa.w))
            
            if (rej_sampling and step.peak_fwd_density > 0.0):
                cnt = 0
                while (cnt < len(p_next)/2.0): # Time-out limit
                    tmp_ind = PF.sample(pa.w,1)[0]
                    p = pa.part[tmp_ind].next_pdf(spart, step.u)
                    accept = numpy.random.uniform()
                    p_acc = p / step.peak_fwd_density
                        
                    if (accept <= p_acc):
                        ind = tmp_ind
                        #print "Rej sampling accept!"
                        break
                    else:
                        p_next[tmp_ind] = p
                    cnt += 1
            
            
            if (ind == None): # No index found, do exhaustive search
                w = numpy.copy(pa.w)
                for i in range(len(pa)):
                    if (p_next[i] < 0.0):
                        p_next[i] = pa.part[i].next_pdf(spart, step.u)
                    
                w *= p_next
                # Normalize
                try:
                    w_norm = w / numpy.sum(w)
                except FloatingPointError:
                    # All weights zero, for robustness just draw a random sample
                    w_norm  = (0.0*w + 1.0)/len(w)
                     
                ind = PF.sample(w_norm,1)[0]

            # Select 'previous' particle
            prev_part = pa.part[ind]
            p_index = pt.traj.index(step)           
            # Sample smoothed linear estimate
            spart = prev_part.sample_smooth(pt, p_index, spart)
            
            self.traj.append(spart)
            self.u.append(step.u)
            self.y.append(step.y)
            self.t.append(step.t)
            
        # Reorder so the list are ordered with increasing time
        self.traj.reverse()
        self.u.reverse()
        self.y.reverse()
        self.t.reverse()
        self.part0 = copy.deepcopy(prev_part)
        
        if (callback):
            callback(self.traj)
        
    def __len__(self):
        return len(self.traj)


class SmoothTrajectoryRB(object):
    def __init__(self, st):
        """ Do constrained filtering of linear states
            st - smoothed trajectory """
        
        self.traj = numpy.empty(len(st), type(st.part0))
        self.u = copy.deepcopy(st.u)
        self.y = copy.deepcopy(st.y)
        self.t = copy.deepcopy(st.t)
        
        # Initialise trajectory to contain objects of the correct type
        for i in range(len(self.traj)):
            self.traj[i] = copy.deepcopy(st.part0)
                   
        # Forward filtering
        self.traj[0].set_nonlin_state(st.traj[0].eta)

        for i in range(len(self.traj)-1):
            
#            print "RBFwd: %d" % i
            
            if (self.y[i] != None):
                tmp = self.traj[i].clin_measure(self.y[i])
                self.traj[i].set_lin_est(tmp)
            
            tmp = self.traj[i].clin_update(self.traj[i].linear_input(self.u[i]))

            self.traj[i+1].set_nonlin_state(st.traj[i+1].eta)
            self.traj[i+1].set_lin_est(tmp)
        
        # Backard smoothing
        for i in reversed(range(len(self.traj)-1)):
            
#            print "RBBack: %d" % i
            
            tmp = self.traj[i].clin_smooth(self.traj[i+1].get_lin_est(),
                                           self.traj[i].linear_input(self.u[i]))

            self.traj[i].set_lin_est(tmp)
    
            
    def __len__(self):
        return len(self.traj)


def do_smoothing(pt, M, rej_sampling=True, callback=None):
    """ return an array of smoothed trajectories 
        M - number of smoothed trajectories """
    
    # Calculate coefficients needed for rejection sampling in the backward smoothing
    if (rej_sampling):
        pt.prep_rejection_sampling()
    
    straj = numpy.empty(M, SmoothTrajectory)
    print "smoothing"
    for i in range(M):
        print "%d/%d" % (i,M)
        straj[i] = SmoothTrajectory(pt, rej_sampling=rej_sampling, callback=callback)
        
    return straj


def do_rb_smoothing(straj):
    """ Calculate rao-blackwellized smoothing of regular smoothed trajectories  """
    
    print "rb smoothing"
    M = len(straj)
    for i in range(M):
        print "%d/%d" % (i,M)
        straj[i] = SmoothTrajectoryRB(straj[i])
    
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
