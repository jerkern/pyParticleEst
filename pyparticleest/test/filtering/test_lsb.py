'''
Created on Nov 11, 2013

@author: ajn
'''

import numpy
import math
import matplotlib.pyplot as plt
from pyparticleest.models.mlnlg import MixedNLGaussianInitialGaussian
import pyparticleest.param_est as param_est
import scipy.io
import sys

C_theta = numpy.array([[ 0.0, 0.04, 0.044, 0.08],])
def calc_Ae_fe(eta, t):
    Ae = eta/(1+eta**2)*C_theta
    fe = 0.5*eta+25*eta/(1+eta**2)+8*math.cos(1.2*t)
    return (Ae, fe)

def calc_h(eta):
    return 0.05*eta**2


def generate_dataset(length):
    Az = numpy.array([[3.0, -1.691, 0.849, -0.3201],
                      [2.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.5, 0.0]])
        
    C = numpy.array([[0.0, 0.0, 0.0, 0.0]])
    
    Qe= numpy.diag([ 0.005])
    Qz = numpy.diag([ 0.01, 0.01, 0.01, 0.01])
    R = numpy.diag([0.1,])
    
    e_vec = numpy.zeros((1, length+1))
    z_vec = numpy.zeros((4, length+1))
    
    e = numpy.array([[0.0,]])
    z = numpy.zeros((4,1))
    
    e_vec[:,0] = e.ravel()
    z_vec[:,0] = z.ravel()
    
    y = numpy.zeros((1, length))
    t = 0
    h = calc_h(e)
    #y[:,0] = (h + C.dot(z)).ravel()
    
    for i in range(1,length+1):
        (Ae, fe) = calc_Ae_fe(e, t)
        
        e = fe + Ae.dot(z) + numpy.random.multivariate_normal(numpy.zeros((1,)),Qe)
        
        wz = numpy.random.multivariate_normal(numpy.zeros((4,)), Qz).ravel().reshape((-1,1))
        
        z = Az.dot(z) + wz
        t = t + 1
        h = calc_h(e)
        y[:,i-1] = (h + C.dot(z) + numpy.random.multivariate_normal(numpy.zeros((1,)), R)).ravel()
        e_vec[:,i] = e.ravel()
        z_vec[:,i] = z.ravel()
    
    return (y.T.tolist(), e_vec, z_vec)    

class ParticleLSB(MixedNLGaussianInitialGaussian):
    """ Model 60 & 61 from Lindsten & Schon (2011) """
    def __init__(self):
        """ Define all model variables """
        
        # No uncertainty in initial state
        xi0 = numpy.array([[0.0],])
        z0 =  numpy.array([[0.0],
                                [0.0],
                                [0.0],
                                [0.0]])
        P0 = 0.0*numpy.eye(4)
        
        Az = numpy.array([[3.0, -1.691, 0.849, -0.3201],
                          [2.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 0.5, 0.0]])
        
        Qxi= numpy.diag([ 0.005])
        Qz = numpy.diag([ 0.01, 0.01, 0.01, 0.01])
        R = numpy.diag([0.1,])

        super(ParticleLSB,self).__init__(xi0=xi0, z0=z0, Pz0=P0,
                                         Az=Az, R=R,
                                         Qxi=Qxi, Qz=Qz,)
   
    def get_nonlin_pred_dynamics(self, particles, u, t):
        N = len(particles)
        C_theta = numpy.array([[ 0.0, 0.04, 0.044, 0.08],])
        tmp = numpy.vstack(particles)[:,numpy.newaxis,:]
        xi = tmp[:,:,0]
        Axi = (xi/(1+xi**2)).dot(C_theta)
        Axi = Axi[:,numpy.newaxis,:]
        fxi = 0.5*xi+25*xi/(1+xi**2)+8*math.cos(1.2*t)
        fxi = fxi[:,numpy.newaxis,:]
        return (Axi, fxi, None)
        
    def get_meas_dynamics(self, particles, y, t):
        N = len(particles)
        tmp = numpy.vstack(particles)
        h = 0.05*tmp[:,0]**2
        h = h[:,numpy.newaxis,numpy.newaxis]
        
        return (numpy.asarray(y).reshape((-1,1)), None, h, None)

#class ParticleLSB_JN(ParticleLSB):
#    """ Model 60 & 61 from Lindsten & Schon (2011) """
#    def __init__(self):
#        """ Define all model variables """
#        
#        # No uncertainty in initial state
#        eta = numpy.array([[0.0],])
#        z0 =  numpy.array([[0.0],
#                           [0.0],
#                           [0.0],
#                           [0.0]])
#        P0 = 0.0*numpy.eye(4)
#        
#        Az = numpy.array([[3.0, -1.691, 0.849, -0.3201],
#                          [2.0, 0.0, 0.0, 0.0],
#                          [0.0, 1.0, 0.0, 0.0],
#                          [0.0, 0.0, 0.5, 0.0]])
#        
#        (Ae, fe) = calc_Ae_fe(eta, 0)
#        h = calc_h(eta)
#        C = numpy.array([[0.0, 0.0, 0.0, 0.0]])
#        
#        Qe= numpy.diag([ 0.005])
#        Qz = numpy.diag([ 0.01, 0.01, 0.01, 0.01])
#        R = numpy.diag([0.1,])
#
#        super(ParticleLSB,self).__init__(z0=numpy.reshape(z0,(-1,1)),
#                                         P0=P0, e0 = eta,
#                                         Az=Az, C=C, Ae=Ae,
#                                         R=R, Qe=Qe, Qz=Qz,
#                                         fe=fe, h=h)
#        
#    def eval_1st_stage_weight(self, u,y):
##        eta_old = copy.deepcopy(self.get_nonlin_state())
##        lin_old = copy.deepcopy(self.get_lin_est())
##        t_old = self.t
#        self.prep_update(u)
#        noise = numpy.zeros_like(self.eta)
#        self.update(u, noise)
#        
#        dh = numpy.asarray(((0.05*2*self.eta,),))
#        
#        yn = self.prep_measure(y)
#        self.kf.R = self.kf.R + dh*self.Qe*dh
#        logpy = self.measure(yn)
#        
#        # Restore state
##        self.set_lin_est(lin_old)
##        self.set_nonlin_state(eta_old)
##        self.t = t_old
#        
#        return logpy

def wmean(logw, val):
    w = numpy.exp(logw)
    w = w / sum(w)
    return numpy.sum(w*val.ravel())

if __name__ == '__main__':
    
    num = 300
    nums = 50
        
    # How many steps forward in time should our simulation run
    steps = 100
    
    sims = 1000
    
    if (len(sys.argv) > 1):
        if (sys.argv[1] == 'nogui'):
        
            sqr_err_eta = numpy.zeros((sims, steps+1))
            sqr_err_theta = numpy.zeros((sims, steps+1))

            C_theta = numpy.array([[ 0.0, 0.04, 0.044, 0.08],])
            for k in range(sims):
                # Create reference
                numpy.random.seed(k)
                (y, e, z) = generate_dataset(steps)
        
                model = ParticleLSB()
                # Create an array for our particles 
                ParamEstimator = param_est.ParamEstimation(model=model, u=None, y=y)
                ParamEstimator.simulate(num, nums, res=0.67, filter='PF', smoother='rsas')
        
                svals = numpy.zeros((2, nums, steps+1))
                
                for i in range(steps+1):
                    (xil, zl, Pl) = model.get_states(ParamEstimator.straj.traj[i])
                    svals[0,:,i] = numpy.vstack(xil).ravel()
                    svals[1,:,i] = 25.0+C_theta.dot(numpy.hstack(zl)).ravel()
                
                # Use average of trajectories
                svals_mean = numpy.mean(svals,1)
        
                theta = 25.0+C_theta.dot(z.reshape((4,-1)))
                sqr_err_eta[k,:] = (svals_mean[0,:] - e[0,:])**2
                sqr_err_theta[k,:] = (svals_mean[1,:] - theta)**2
        
                rmse_eta = numpy.sqrt(numpy.mean(sqr_err_eta[k,:]))
                rmse_theta = numpy.sqrt(numpy.mean(sqr_err_theta[k,:]))
                print "%d %f %f" % (k, numpy.mean(rmse_eta), numpy.mean(rmse_theta))
        elif (sys.argv[1] == 'apf_compare'):
            
            sqr_err_eta_pf = numpy.zeros((sims, steps+1))
            sqr_err_theta_pf = numpy.zeros((sims, steps+1))
            
            sqr_err_eta_apf = numpy.zeros((sims, steps+1))
            sqr_err_theta_apf = numpy.zeros((sims, steps+1))

            C_theta = numpy.array([[ 0.0, 0.04, 0.044, 0.08],])
            for k in range(sims):
                # Create reference
                numpy.random.seed(k)
                (y, e, z) = generate_dataset(steps)
        
                model = ParticleLSB()
                # Create an array for our particles 
                pepf = param_est.ParamEstimation(model=model, u=None, y=y)
                pepf.simulate(num, num_traj=1, res=0.67, filter='PF', smoother='ancestor')
                
                peapf = param_est.ParamEstimation(model=model, u=None, y=y)
                peapf.simulate(num, num_traj=1, res=0.67, filter='APF', smoother='ancestor')
        
                avg_pf = numpy.zeros((2, steps+1))
                avg_apf = numpy.zeros((2, steps+1))
                for i in range(steps+1):
                    avg_pf[0,i] = wmean(pepf.pt.traj[i].pa.w, numpy.vstack(pepf.pt.traj[i].pa.part)[:,0])
                    avg_pf[1,i] = wmean(pepf.pt.traj[i].pa.w, numpy.vstack(pepf.pt.traj[i].pa.part)[:,1])
                    
                    avg_apf[0,i] = wmean(peapf.pt.traj[i].pa.w, numpy.vstack(peapf.pt.traj[i].pa.part)[:,0])
                    avg_apf[1,i] = wmean(peapf.pt.traj[i].pa.w, numpy.vstack(peapf.pt.traj[i].pa.part)[:,1])
                        
                theta = 25.0+C_theta.dot(z.reshape((4,-1)))
                sqr_err_eta_pf[k,:] = (avg_pf[0,:] - e[0,:])**2
                sqr_err_theta_pf[k,:] = (avg_pf[1,:] - theta)**2
                        
                sqr_err_eta_apf[k,:] = (avg_apf[0,:] - e[0,:])**2
                sqr_err_theta_apf[k,:] = (avg_apf[1,:] - theta)**2
        
                rmse_eta_pf = numpy.sqrt(numpy.mean(sqr_err_eta_pf[k,:]))
                rmse_theta_pf = numpy.sqrt(numpy.mean(sqr_err_theta_pf[k,:]))
                
                rmse_eta_apf = numpy.sqrt(numpy.mean(sqr_err_eta_apf[k,:]))
                rmse_theta_apf = numpy.sqrt(numpy.mean(sqr_err_theta_apf[k,:]))
                
                print "%d: %f %f: %f %f" % (k, rmse_eta_pf, rmse_theta_pf, 
                                            rmse_eta_apf, rmse_theta_apf)
            pass
            
    else:
    

    
        # Create arrays for storing some values for later plotting    
        vals = numpy.zeros((2, num+1, steps+1))
    
        plt.ion()

        # Create reference
        #numpy.random.seed(1)
        #numpy.random.seed(86)
        (y, e, z) = generate_dataset(steps)
        # Store values for last time-step aswell    
    
        
        x = numpy.asarray(range(steps+1))
        model = ParticleLSB()
        # Create an array for our particles 
        ParamEstimator = param_est.ParamEstimation(model=model, u=None, y=y)
        ParamEstimator.simulate(num, nums, res=0.67, filter='PF', smoother='rsas')

        
        svals = numpy.zeros((2, nums, steps+1))
        vals = numpy.zeros((2, num, steps+1))
 
        for i in range(steps+1):
            (xil, zl, Pl) = model.get_states(ParamEstimator.straj.traj[i])
            svals[0,:,i] = numpy.vstack(xil).ravel()
            svals[1,:,i] = 25.0+C_theta.dot(numpy.hstack(zl)).ravel()
            (xil, zl, Pl) = model.get_states(ParamEstimator.pt.traj[i].pa.part)
            vals[0,:,i]=numpy.vstack(xil).ravel()
            vals[1,:,i]=25.0+C_theta.dot(numpy.hstack(zl)).ravel()
                
        svals_mean = numpy.mean(svals,1)
        plt.figure()

        for j in range(num):   
            plt.plot(range(steps+1),vals[0,j,:],'.', markersize=3.0, color='#BBBBBB')
        plt.plot(x, e.T,'k-',markersize=1.0)
        for j in range(nums):
            plt.plot(range(steps+1),svals[0,j,:],'--', markersize=2.0, color='#999999', dashes=(7,50))
        #plt.plot(range(steps+1),svals_mean[0,:],'--', markersize=1.0, color='1.0')

        #plt.savefig('rbps_fail_xi.eps', bbox_inches='tight')
        plt.show()
        
        plt.figure()
        for j in range(num):   
            plt.plot(range(steps+1),vals[1,j,:],'.', markersize=3.0, color='#BBBBBB')
        plt.plot(x, (25.0+C_theta.dot(z)).ravel(),'k-',markersize=1.0)
        for j in range(nums):
            plt.plot(range(steps+1),svals[1,j,:],'--', markersize=2.0, color='#999999', dashes=(10,25))
        #plt.plot(range(steps+1),svals_mean[1,:],'--', markersize=1.0, color='1.0')
        #plt.savefig('rbps_fail_theta.eps', bbox_inches='tight')
        plt.show()

        plt.draw()
        # Export data for plotting in matlab
        scipy.io.savemat('test_lsb.mat', {'svals_mean': svals_mean, 'vals': vals, 
                                          'svals': svals, 'y': y, 'z': z, 'e':e})
        
        plt.ioff()
        plt.show()
        plt.draw()
    print "exit"