'''
Created on Jun 25, 2014

@author: Jerker Nordh
'''

class Instrumenter(object):
    """
    Count number of operations performed

    Wraps all calls and counts the number of calls * number of particles in
    each call.

    Results can be access through the member variables:

        - self.cnt_sample
        - self.cnt_update
        - self.cnt_measure
        - self.cnt_pdfxn
        - self.cnt_pdfxn_full
        - self.cnt_pdfxnmax
        - self.cnt_propsmooth
        - self.cnt_pdfsmooth
        - self.cnt_eval1st

    Args:
     - model: Object of encapsulated model class
    """


    def __init__(self, model):
        self.model = model
        self.cnt_sample = 0
        self.cnt_update = 0
        self.cnt_measure = 0
        self.cnt_pdfxn = 0
        self.cnt_pdfxn_full = 0
        self.cnt_pdfxnmax = 0
        self.cnt_propsmooth = 0
        self.cnt_pdfsmooth = 0
        self.cnt_eval1st = 0

    def print_statistics(self):
        print "Modelclass : %s" % type(self.model)
        print "cnt_sample: %d" % self.cnt_sample
        print "cnt_update: %d" % self.cnt_update
        print "cnt_measure: %d" % self.cnt_measure
        print "cnt_pdfxn: %d" % self.cnt_pdfxn + self.cnt_pdfxn_full
        print "cnt_pdfxnmax: %d" % self.cnt_pdfxnmax
        print "cnt_propsmooth: %d" % self.cnt_propsmooth
        print "cnt_pdfsmooth: %d" % self.cnt_pdfsmooth


    def print_total_ops(self):
        print "total ops: %d" % (self.cnt_sample + self.cnt_update +
                                 self.cnt_measure + self.cnt_pdfxn +
                                 self.cnt_pdfxnmax + self.cnt_propsmooth +
                                 self.cnt_pdfsmooth)

    def create_initial_estimate(self, N):
        """ Sample N particle from initial distribution """
        return self.model.create_initial_estimate(N)

    def sample_process_noise(self, particles, u, t):
        """ Return process noise for input u """
        self.cnt_sample += len(particles)
        return self.model.sample_process_noise(particles, u, t)

    def update(self, particles, u, t, noise):
        """ Update estimate using 'data' as input """
        self.cnt_update += len(particles)
        return self.model.update(particles, u, t, noise)

    def measure(self, particles, y, t):
        """ Return the log-pdf value of the measurement """
        self.cnt_measure += len(particles)
        return self.model.measure(particles, y, t)

    def copy_ind(self, particles, new_ind=None):
        return self.model.copy_ind(particles, new_ind)

    def logp_xnext(self, particles, next_part, u, t):
        """ Return the log-pdf value for the possible future state 'next' given input u """
        self.cnt_pdfxn += max(len(particles), len(next_part))
        return self.model.logp_xnext(particles, next_part, u, t)

    def logp_xnext_max(self, particles, u, t):
        """ Return the log-pdf value for the possible future state 'next' given input u """
        self.cnt_pdfxnmax += len(particles)
        return self.model.logp_xnext_max(particles, u, t)

    def sample_smooth(self, particles, future_trajs, ut, yt, tt):
        """ Update ev. Rao-Blackwellized states conditioned on "next_part" """
        return self.model.sample_smooth(particles, future_trajs, ut, yt, tt)

    def propose_smooth(self, partp, up, tp, ut, yt, tt, future_trajs):
        """ Sample from a distrubtion q(x_t | x_{t-1}, x_{t+1}, y_t) """
        if (partp != None):
            N = len(partp)
        else:
            N = future_trajs.shape[1]
        self.cnt_propsmooth += N
        return self.model.propose_smooth(partp, up, tp, ut, yt, tt, future_trajs)

    def logp_proposal(self, prop_part, partp, up, tp, ut, yt, tt, future_trajs):
        """ Eval log q(x_t | x_{t-1}, x_{t+1}, y_t) """
        self.cnt_pdfsmooth += len(prop_part)
        return self.model.logp_proposal(prop_part, partp, up, tp,
                                        ut, yt, tt, future_trajs)

    def logp_xnext_full(self, particles, future_trajs, ut, yt, tt):
        self.cnt_pdfxn += max(len(particles), future_trajs.shape[1])
        return self.model.logp_xnext_full(particles, future_trajs, ut, yt, tt)

    def eval_1st_stage_weights(self, particles, u, y, t):
        self.cnt_eval1st += len(particles)
        return self.model.eval_1st_stage_weights(particles, u, y, t)
