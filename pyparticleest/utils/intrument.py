'''
Created on Jun 25, 2014

@author: Jerker Nordh
'''

class OpCount(object):
    def __init__(self, cnt_sample=0, cnt_update=0, cnt_measure=0, cnt_pdfxn=0,
                 cnt_pdfxn_full=0, cnt_pdfxnmax=0, cnt_propsmooth=0,
                 cnt_pdfsmooth=0, cnt_eval1st=0, cnt_eval_logp_x0=0):
        self.cnt_sample = cnt_sample
        self.cnt_update = cnt_update
        self.cnt_measure = cnt_measure
        self.cnt_pdfxn = cnt_pdfxn
        self.cnt_pdfxn_full = cnt_pdfxn_full
        self.cnt_pdfxnmax = cnt_pdfxnmax
        self.cnt_propsmooth = cnt_propsmooth
        self.cnt_pdfsmooth = cnt_pdfsmooth
        self.cnt_eval1st = cnt_eval1st
        self.cnt_eval_logp_x0 = cnt_eval_logp_x0

    def __add__(self, other):
        x = OpCount(cnt_sample=self.cnt_sample + other.cnt_sample,
                    cnt_update=self.cnt_update + other.cnt_update,
                    cnt_measure=self.cnt_measure + other.cnt_measure,
                    cnt_pdfxn=self.cnt_pdfxn + other.cnt_pdfxn,
                    cnt_pdfxn_full=self.cnt_pdfxn_full + other.cnt_pdfxn_full,
                    cnt_pdfxnmax=self.cnt_pdfxnmax + other.cnt_pdfxnmax,
                    cnt_propsmooth=self.cnt_propsmooth + other.cnt_propsmooth,
                    cnt_pdfsmooth=self.cnt_pdfsmooth + other.cnt_pdfsmooth,
                    cnt_eval1st=self.cnt_eval1st + other.cnt_eval1st,
                    cnt_eval_logp_x0=self.cnt_eval_logp_x0 + other.cnt_eval_logp_x0)
        return x


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
        self.oc = OpCount()

    def print_statistics(self):
        print("Modelclass : %s" % type(self.model))
        print("cnt_sample: %d" % self.cnt_sample)
        print("cnt_update: %d" % self.cnt_update)
        print("cnt_measure: %d" % self.cnt_measure)
        print("cnt_pdfxn: %d" % self.cnt_pdfxn + self.cnt_pdfxn_full)
        print("cnt_pdfxnmax: %d" % self.cnt_pdfxnmax)
        print("cnt_propsmooth: %d" % self.cnt_propsmooth)
        print("cnt_pdfsmooth: %d" % self.cnt_pdfsmooth)


    def print_total_ops(self):
        print("total ops: %d" % (self.cnt_sample + self.cnt_update +
                                 self.cnt_measure + self.cnt_pdfxn +
                                 self.cnt_pdfxnmax + self.cnt_propsmooth +
                                 self.cnt_pdfsmooth))

    def create_initial_estimate(self, N):
        """ Sample N particle from initial distribution """
        return self.model.create_initial_estimate(N)

    def sample_process_noise_full(self, ptraj, ancestors, ut, tt):
        """ Return process noise for input u """
        self.oc.cnt_sample += len(ancestors)
        return self.model.sample_process_noise_full(ptraj, ancestors, ut, tt)

    def update_full(self, particles, traj, uvec, yvec, tvec, ancestors, noise):
        """ Update estimate using 'data' as input """
        self.oc.cnt_update += len(particles)
        return self.model.update_full(particles, traj, uvec, yvec, tvec, ancestors, noise)

    def measure_full(self, particles, traj, uvec, yvec, tvec, ancestors):
        """ Return the log-pdf value of the measurement """
        self.oc.cnt_measure += len(particles)
        return self.model.measure_full(particles, traj, uvec, yvec, tvec, ancestors)

    def copy_ind(self, particles, new_ind=None):
        return self.model.copy_ind(particles, new_ind)

    def logp_xnext(self, particles, next_part, u, t):
        """ Return the log-pdf value for the possible future state 'next' given input u """
        self.oc.cnt_pdfxn += max(len(particles), len(next_part))
        return self.model.logp_xnext(particles, next_part, u, t)

    def logp_xnext_max_full(self, part, past_trajs, pind, uvec, yvec, tvec, cur_ind):
        """ Return the log-pdf value for the possible future state 'next' given input u """
        self.oc.cnt_pdfxnmax += len(part)
        return self.model.logp_xnext_max_full(part, past_trajs, pind, uvec, yvec, tvec, cur_ind)

    def sample_smooth(self, part, ptraj, anc, future_trajs, find, ut, yt, tt, cur_ind):
        """ Update ev. Rao-Blackwellized states conditioned on "next_part" """
        return self.model.sample_smooth(part, ptraj, anc, future_trajs, find, ut, yt, tt, cur_ind)

    def propose_smooth(self, ptraj, anc, future_trajs, find, yt, ut, tt, cur_ind):
        """ Sample from a distrubtion q(x_t | x_{t-1}, x_{t+1}, y_t) """
        if (ptraj is not None):
            N = len(anc)
        else:
            N = len(find)
        self.oc.cnt_propsmooth += N
        return self.model.propose_smooth(ptraj, anc, future_trajs, find, yt, ut, tt, cur_ind)

    def logp_proposal(self, prop_part, ptraj, anc, future_trajs, find, yt, ut, tt, cur_ind):
        """ Eval log q(x_t | x_{t-1}, x_{t+1}, y_t) """
        self.oc.cnt_pdfsmooth += len(prop_part)
        return self.model.logp_proposal(prop_part, ptraj, anc,
                                        future_trajs, find,
                                        yt, ut, tt, cur_ind)

    def logp_xnext_full(self, part, past_trajs, pind, future_trajs, find,
                        ut, yt, tt, cur_ind):
        self.oc.cnt_pdfxn += max(len(part), len(find))
        return self.model.logp_xnext_full(part, past_trajs, pind, future_trajs,
                                          find, ut, yt, tt, cur_ind)

    def logp_xnext_singlestep(self, part, past_trajs, pind,
                              future_parts, find, ut, yt, tt, cur_ind):
        self.oc.cnt_pdfxn += max(len(part), len(find))
        return self.model.logp_xnext_singlestep(part, past_trajs, pind,
                                                future_parts, find,
                                                ut, yt, tt, cur_ind)

    def eval_1st_stage_weights(self, particles, u, y, t):
        self.oc.cnt_eval1st += len(particles)
        return self.model.eval_1st_stage_weights(particles, u, y, t)

    def pre_mhips_pass(self, st):
        return self.model.pre_mhips_pass(st)

    def post_smoothing(self, st):
        return self.model.post_smoothing(st)

    def eval_logp_x0(self, particles, t):
        self.oc.cnt_eval_logp_x0 += len(particles)
        return self.model.eval_logp_x0(particles, t)

    def cond_predict_single_step(self, part, past_trajs, pind, future_parts, find, ut, yt, tt, cur_ind):
        return self.model.cond_predict_single_step(part, past_trajs, pind, future_parts, find, ut, yt, tt, cur_ind)

    def cond_sampled_initial(self, part, t):
        return self.model.cond_sampled_initial(part, t)
