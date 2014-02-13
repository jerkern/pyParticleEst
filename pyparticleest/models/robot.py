""" Module implementing robot motion models """

from __future__ import division
import math
import numpy
import scipy.stats
import part_utils


class ExactDifferentialRobot(object):
    """ Differential drive, without noise """
    
    def __init__(self, l, d, ticks=1000, state=(0.0, 0.0, math.pi/2),
                 wp=(0.0, 0.0)):
        """ Create robot object which can be used to simulate movement 
        
        l - wheel base distance
        d - wheel diameter
        ticks - ticks per revolution for the wheel encoders
        state - initial state (x,y,theta)
        wp - initial positions for the wheel positions (l,r)
        
        """
            
        assert(l > 0)
        assert(d > 0)
        
        self.l = l
        self.d = d
        self.state = numpy.asarray(state)

        self.old_S = sum(wp)
        self.old_D = numpy.diff(wp)[0]
        self.old2_S = self.old_S
        self.old2_D = self.old_D
        self.ticks = ticks

        # This contant is needed in the update phase        
        self.C = math.pi * self.d / self.ticks

    def calc_next_theta(self, wp):
        # Tustion approximation for the state
        # Velocities estimated using backward difference           
        #D = numpy.diff(wp)[0]
        D = wp[1] - wp[0]
        
        # First estimate new angle, since it doesn't depend on other states
        # and is needed for the other calculations        
        theta_new = self.state[2] + self.C/(2.0*self.l)*(D-self.old2_D)
        
        return theta_new
        
    def calc_next_xy(self, wp, theta_new):
        
        S = wp[0] + wp[1]

        cn = math.cos(theta_new)
        sn = math.sin(theta_new)
        
        co = math.cos(self.state[2])
        so = math.sin(self.state[2])
        
        return numpy.array((self.state[0]+self.C/4*((S-self.old_S)*cn+(self.old_S-self.old2_S)*co),
                            self.state[1]+self.C/4*((S-self.old_S)*sn+(self.old_S-self.old2_S)*so)))

    def kinematic(self, wp):
        """ Update state using the new wheel positions

            wp - wheel positions """

     
        #S = numpy.sum(wp)
        S = wp[0]+wp[1]
        #D = numpy.diff(wp)[0]
        D = wp[1]-wp[0]
        
        theta_new = self.calc_next_theta(wp)
        
        tmp = self.calc_next_xy(wp, theta_new)
        self.state[0] = tmp[0]
        self.state[1] = tmp[1]
        self.state[2] = theta_new
        
        self.old2_S = self.old_S
        self.old_S = S
        
        self.old2_D = self.old_D
        self.old_D = D
        
        # Return estimate of how robot has moved (forward, turn)
        return (self.C/2.0*(self.old_S - self.old2_S), 
                self.C/(2.0*self.l)*(self.old_D - self.old2_D))

    def set_state(self, state):
        self.state = state[0:3]
        self.old_S = state[3]
        self.old2_S = state[4]
        self.old_D = state[5]
        self.old2_D = state[6]
        
    def get_state(self):
        return numpy.asarray((self.state[0], self.state[1], self.state[2],
                              self.old_S, self.old2_S, self.old_D, self.old2_D))
        
    def add_state_noise(self, os):
        # Noise specified in local coordinate system, transform to global first
                # Add offset of sensor, offset is specified in local coordinate system,
        # rotate to match global coordinates
        theta = self.state[2]
        rot_mat = numpy.array(((math.cos(theta), -math.sin(theta)),
                    (math.sin(theta), math.cos(theta))))
        
        self.state[:2] = self.state[:2] + rot_mat.dot(os[:2])
        
        self.state[2] = self.state[2] + os[2]
        

class DifferentialRobot(part_utils.ParticleFilteringBase):
    """ Differential drive robot """

    def __init__(self, l, d, enc_noise, theta_noise, enc_noise_lin=0.0,
                 theta_noise_lin=0.0, ticks=1000, state=(0.0, 0.0, math.pi/2),
                 wp=(0.0, 0.0)):
        """ Create robot object which can be used to simulate movement 
        
        l - wheel base distance
        d - wheel diameter
        ticks - ticks per revolution for the wheel encoders
        state - initial state (x,y,theta)
        wp - initial positions for the wheel positions (l,r)
        
        """
        
        self.robot = ExactDifferentialRobot(l=l, d=d, ticks=ticks, state=state, wp=wp)

        self.enc_noise = enc_noise
        self.enc_noise_lin = enc_noise_lin
        self.motion = (0.0, 0.0)

        self.theta_noise = theta_noise
        self.theta_noise_lin = theta_noise_lin
        
        self.cur_enc_noise = enc_noise
        self.cur_theta_noise = theta_noise


    def update(self, u):
        """ Move robot according to kinematics """
        
        motion = self.robot.kinematic(u[:2])
        theta_noise = numpy.random.normal(0,self.cur_theta_noise)
        
        self.cur_enc_noise = self.enc_noise + numpy.abs(motion[0])*self.enc_noise_lin
        self.cur_theta_noise = self.theta_noise + numpy.abs(motion[1])*self.theta_noise_lin
            
        # Process internal noise
        self.robot.add_state_noise((0, 0, theta_noise))
        
        
    def sample_input_noise(self, w):
        
        w = numpy.asarray(w)
        
        # Noise corrupted input
        wn = numpy.random.uniform(w[:2,:]-self.cur_enc_noise/2.0, 
                               w[:2,:]+self.cur_enc_noise/2.0, (2,1))
        
        return wn.ravel()
        
    def measure(self, y):
        return NotImplemented
   
    def get_state(self):
        return self.robot.get_state()
    
    def set_state(self, state):
        self.robot.set_state(state)
        
    def set_pos(self, x, y, theta):
        state = self.get_state()
        state[0] = x
        state[1] = y
        state[2] = theta
        self.set_state(state)
        
    def get_pos(self):
        state = self.get_state()
        return state[:3] # x, y, theta
    

    def next_pdf(self, next_state, u):
        theta_n = next_state.eta[2]
        x_n = next_state.eta[0]
        y_n = next_state.eta[1]
        
        theta = self.robot.state[2]
        x = self.robot.state[0]
        y = self.robot.state[1]

        dx = x_n - x
        dy = y_n - y

        tmp1 = (4.0/self.robot.C*dy-(self.robot.old_S-self.robot.old2_S)*math.sin(theta))
        tmp2 = (4.0/self.robot.C*dx-(self.robot.old_S-self.robot.old2_S)*math.cos(theta))

        # Calculate sum of inputs for reaching new x and y coords, there are two solutions!
        K = math.sqrt((tmp1)**2 + (tmp2)**2)
       
        S1 = K + self.robot.old_S
        S2 = -K + self.robot.old_S
         
        # Calculate new angle for reaching provide x and y coords, thera are two solutions
        theta_hat1 = math.atan2(tmp1, tmp2)
        #theta_hat2 = math.atan2(-tmp1, -tmp2)
        theta_hat2 = theta_hat1 + math.pi
        # Calculate diff of inputs needed for reaching new angle, always assume the robot hasn't turned
        # more than halt a revolution
        theta_diff1 = numpy.mod(theta_hat1-theta+math.pi, 2.0*math.pi) - math.pi
        theta_diff2 = numpy.mod(theta_hat2-theta+math.pi, 2.0*math.pi) - math.pi
        D1 = 2.0/self.robot.C*self.robot.l*theta_diff1+self.robot.old2_D
        D2 = 2.0/self.robot.C*self.robot.l*theta_diff2+self.robot.old2_D
        
        # Calcuate needed inputs
        Pr1 = 0.5*(S1+D1)
        Pl1 = 0.5*(S1-D1)
        
        Pr2 = 0.5*(S2+D2)
        Pl2 = 0.5*(S2-D2)
       
        # Check probability of inputs
        prob_Pr1 = 1.0/self.cur_enc_noise if (Pr1 >= u[1]-self.cur_enc_noise/2.0 and Pr1 <= u[1]+self.cur_enc_noise/2.0) else 0.0
        prob_Pl1 = 1.0/self.cur_enc_noise if (Pl1 >= u[0]-self.cur_enc_noise/2.0 and Pl1 <= u[0]+self.cur_enc_noise/2.0) else 0.0
        prob_Pr2 = 1.0/self.cur_enc_noise if (Pr2 >= u[1]-self.cur_enc_noise/2.0 and Pr2 <= u[1]+self.cur_enc_noise/2.0) else 0.0
        prob_Pl2 = 1.0/self.cur_enc_noise if (Pl2 >= u[0]-self.cur_enc_noise/2.0 and Pl2 <= u[0]+self.cur_enc_noise/2.0) else 0.0
        
        try:
            (prob_theta1, prob_theta2) = scipy.stats.norm.pdf((theta_hat1, theta_hat2), loc=theta_n, shape=self.cur_theta_noise)
        except FloatingPointError:
            prob_theta1 = 0.0
            prob_theta2 = 0.0
            
        try:
            return math.log(max(prob_Pr1*prob_Pl1*prob_theta1,prob_Pr2*prob_Pl2*prob_theta2))
        except ValueError:
            return -numpy.Inf
    
    def fwd_peak_density(self, u):
#        return ((1.0/self.cur_enc_noise)**2 *
#                scipy.stats.norm.pdf(0, loc=0, shape=self.cur_theta_noise))
        return -2.0*math.log(self.cur_enc_noise)+scipy.stats.norm.logpdf(0, loc=0, shape=self.cur_theta_noise)
