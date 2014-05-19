'''
Created on May 19, 2014

@author: ajn
'''

import numpy
from filter import ParticleTrajectory

class Simulator():
    def __init__(self, model, u, y):
        
        if (u != None):
            self.u = u
        else:
            self.u = [None] * len(y)
        self.y = y
        self.pt = None
        self.straj = None
        self.params = None
        self.model = model
    
    def set_params(self, params):
        self.params = numpy.copy(params)
        self.model.set_params(self.params)
    
    def simulate(self, num_part, num_traj, filter='PF', smoother='full', smoother_options=None, res=0.67, meas_first=False):
        resamplings=0
    
        # Initialise a particle filter with our particle approximation of the initial state,
        # set the resampling threshold to 0.67 (effective particles / total particles )
        self.pt = ParticleTrajectory(self.model, num_part, res,filter=filter)
        
        offset = 0
        # Run particle filter
        if (meas_first):
            self.pt.measure(self.y[0])
            offset = 1
        for i in range(offset,len(self.y)):
            # Run PF using noise corrupted input signal
            if (self.pt.forward(self.u[i-offset], self.y[i])):
                resamplings = resamplings + 1
            
        # Use the filtered estimates above to created smoothed estimates
        if (smoother != None):
            self.straj = self.pt.perform_smoothing(num_traj, method=smoother, smoother_options=smoother_options)
        return resamplings