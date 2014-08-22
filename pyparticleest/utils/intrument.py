'''
Created on Jun 25, 2014

@author: ajn
'''

class Instrumenter(object):
    '''
    Count number of operations performed
    '''


    def __init__(self, model):
        self.model = model
        self.cnt_sample = 0
        self.cnt_update = 0
        self.cnt_measure = 0
        self.cnt_pdfxn = 0
        self.cnt_pdfxnmax = 0
        self.cnt_propsmooth = 0
        self.cnt_pdfsmooth = 0
    
    def print_statistics(self):
        print "Modelclass : %s" % type(self.model)
        print "cnt_sample: %d" % self.cnt_sample
        print "cnt_update: %d" % self.cnt_update
        print "cnt_measure: %d" % self.cnt_measure
        print "cnt_pdfxn: %d" % self.cnt_pdfxn
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
        self.cnt_update  += len(particles)
        return self.model.update(particles, u, t, noise)
    
    def measure(self, particles, y, t):
        """ Return the log-pdf value of the measurement """
        self.cnt_measure  += len(particles)
        return self.model.measure(particles, y, t)
    
    def copy_ind(self, particles, new_ind=None):
        return self.model.copy_ind(particles, new_ind)
    
    def next_pdf(self, particles, next_part, u, t):
        """ Return the log-pdf value for the possible future state 'next' given input u """
        self.cnt_pdfxn  += max(len(particles), len(next_part))
        return self.model.next_pdf(particles, next_part, u, t)
    
    def next_pdf_max(self, particles, u, t):
        """ Return the log-pdf value for the possible future state 'next' given input u """
        self.cnt_pdfxnmax += len(particles)
        return self.model.next_pdf_max(particles, u, t)
    
    def sample_smooth(self, particles, next_part, u, y, t):
        """ Update ev. Rao-Blackwellized states conditioned on "next_part" """
        return self.model.sample_smooth(particles, next_part, u, y, t)
    
    def propose_smooth(self, partp, up, tp, u, y, t, partn):
        """ Sample from a distrubtion q(x_t | x_{t-1}, x_{t+1}, y_t) """
        if (partp != None):
            N = len(partp)
        else:
            N = len(partn)
        self.cnt_propsmooth  += N
        return self.model.propose_smooth(partp, up, tp, u, y, t, partn)

    def logp_smooth(self, prop_part, partp, up, tp, u, y, t, partn):
        """ Eval log q(x_t | x_{t-1}, x_{t+1}, y_t) """
        self.cnt_pdfsmooth += len(prop_part)
        return self.model.logp_smooth(prop_part, partp, up, tp, u, y, t, partn)