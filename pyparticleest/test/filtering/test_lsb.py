'''
Created on Nov 11, 2013

@author: ajn
'''

import numpy
import math
import matplotlib.pyplot as plt
import pyparticleest.models.mlnlg as mlnlg
import pyparticleest.param_est as param_est
import scipy.io
import scipy.linalg
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

class ParticleLSB(mlnlg.MixedNLGaussianInitialGaussianProperBSi):
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
        if (y == None):
            return (y, None, None, None)
        else:
            tmp = numpy.vstack(particles)
            h = 0.05*tmp[:,0]**2
            h = h[:,numpy.newaxis,numpy.newaxis]
        
        return (numpy.asarray(y).reshape((-1,1)), None, h, None)

class ParticleLSB_EKF(ParticleLSB):

    def calc_Sigma_xi(self, particles, u, t): 
        """ Return sampled process noise for the non-linear states """
        (Axi, fxi, Qxi, _, _, _) = self.get_nonlin_pred_dynamics_int(particles=particles, u=u, t=t)
        (_xil, zl, Pl) = self.get_states(particles)
        N = len(particles)
                    
        Sigma = numpy.zeros((N, self.lxi, self.lxi))
        for i in xrange(N):
            Sigma[i] = Qxi[i] + Axi[i].dot(Pl[i]).dot(Axi[i].T)

        return Sigma

    def eval_1st_stage_weights(self, particles, u, y, t):
        N = len(particles)
        part = numpy.copy(particles)
        xin = self.pred_xi(part, u, t)
        Sigma = self.calc_Sigma_xi(particles, u, t)
        self.cond_predict(part, xin, u, t)
        
        tmp = numpy.vstack(part)
        h = 0.05*tmp[:,0]**2
        h_grad = 0.1*tmp[:,0]
        
        tmp = (h_grad**2)
        Rext = self.kf.R + Sigma*tmp[:,numpy.newaxis, numpy.newaxis]
        logRext = numpy.log(Rext)
        diff = y - h
        
        lyz = numpy.empty(N)
        l2pi = math.log(2*math.pi)
        for i in xrange(N):
            lyz[i] = -0.5*(l2pi + logRext[i,0,0] + (diff[i].ravel()**2)/Rext[i,0,0])
        return lyz

class ParticleLSB_UKF(ParticleLSB):

    def eval_1st_stage_weights(self, particles, u, y, t):
        N = len(particles)
        part = numpy.copy(particles)
        #xin = self.pred_xi(part, u, t)

        (Axi, fxi, _, _, _, _) = self.get_nonlin_pred_dynamics_int(particles=particles, u=u, t=t)
        (Az, fz, _, _, _, _) = self.get_lin_pred_dynamics_int(particles=particles, u=u, t=t)
        (_xil, zl, Pl) = self.get_states(particles)

        Rext = numpy.empty(N)
        diff = numpy.empty(N)

        for i in xrange(N):
            m = numpy.vstack((zl[i], numpy.zeros((6,1))))
            K = scipy.linalg.block_diag(Pl[i], self.Qxi, self.kf.Q, self.kf.R)
            Na = len(K)
            (U,s,V) = numpy.linalg.svd(Na*K)
            Kroot = U.dot(numpy.diag(numpy.sqrt(s)))

            ypred = numpy.empty(2*Na)
            # Some ugly hard-coding here of the function f and g
            # g = 0.05*xi**2
            for j in xrange(Na):
                val = m + Kroot[:,j:j+1]
                xin = fxi[i]+Axi[i].dot(val[:4])+val[4]
                ypred[j] = 0.05*(xin)**2+val[9]

                val = m - Kroot[:,j:j+1]
                xin = fxi[i]+Axi[i].dot(val[:4])+val[4]
                ypred[Na+j] = 0.05*(xin)**2+val[9]

            # Construct estimate of covariance for predicted measurement
            Rext[i] = numpy.cov(ypred)
            diff[i] = y - numpy.mean(ypred)

        logRext = numpy.log(Rext)

        lyz = numpy.empty(N)
        l2pi = math.log(2*math.pi)
        for i in xrange(N):
            lyz[i] = -0.5*(l2pi + logRext[i] + (diff[i].ravel()**2)/Rext[i])
        return lyz


#    def eval_1st_stage_weights(self, particles, u, y, t):
#        N = len(particles)
#        part = numpy.copy(particles)
#        xin = self.pred_xi(part, u, t)
#        Sigma = self.calc_Sigma_xi(particles, u, t)
#        self.cond_predict(part, xin, u, t)
#        return self.measure_1st(part, Sigma, y, t)
#    
#    def measure_1st(self, particles, Sigma, y, t):
#        """ Return the log-pdf value of the measurement """
#        
#        (xil, zl, Pl) = self.get_states(particles)
#        N = len(particles)
#        (y, Cz, hz, Rz, _, _, _) = self.get_meas_dynamics_int(particles=particles, y=y, t=t)
#        h_grad = self.get_h_grad(particles, y, t)
#        lyz = numpy.empty(N)
#        for i in xrange(len(zl)):
#            Rext = Rz[i]+h_grad[i].dot(Sigma[i]).dot(h_grad[i].T)
#            lyz[i] = self.kf.measure_full(y=y, z=zl[i], P=Pl[i], C=Cz[i], h_k=hz[i], R=Rext)
#        return lyz
#    
#    def get_h_grad(self, particles, y, t):
#        tmp = numpy.vstack(particles)
#        h_grad = 0.1*tmp[:,0]
#        return h_grad[:,numpy.newaxis,numpy.newaxis]

def wmean(logw, val):
    w = numpy.exp(logw)
    w = w / sum(w)
    return numpy.sum(w*val.ravel())

if __name__ == '__main__':
    
    num = 300
    nums = 50
        
    # How many steps forward in time should our simulation run
    steps = 100
    
    if (len(sys.argv) > 1):
        if (sys.argv[1] == 'nogui'):
            sims = 1000
            sqr_err_eta = numpy.zeros((sims, steps+1))
            sqr_err_theta = numpy.zeros((sims, steps+1))

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
            
            mode = sys.argv[2]
            
            print "Running tests for %s" % mode
            
            sims = 50000
            part_count = (5, 10, 15, 20, 25, 30, 50, 75, 100, 150, 200, 300, 500)
            rmse_eta = numpy.zeros((sims, len(part_count)))
            rmse_theta = numpy.zeros((sims, len(part_count)))
            filter = 'PF'
            model=ParticleLSB()
            if (mode.lower() == 'epf'):
                model = ParticleLSB_EKF()
                filter = 'APF'
            elif (mode.lower() == 'upf'):
                model = ParticleLSB_UKF()
                filter = 'APF'
            elif (mode.lower() == 'apf'):
                filter = 'APF'
            else:
                pass
            
            for k in range(sims):
                
                # Create reference
                numpy.random.seed(k)
                (y, e, z) = generate_dataset(steps)
                pe = param_est.ParamEstimation(model=model, u=None, y=y)
                    
                for ind, pc in enumerate(part_count):
                
                    pe.simulate(pc, num_traj=1, res=0.67, filter=filter, smoother='ancestor')
                    avg = numpy.zeros((2, steps+1))

                    for i in range(steps+1):
                        avg[0,i] = wmean(pe.pt.traj[i].pa.w, numpy.vstack(pe.pt.traj[i].pa.part)[:,0])
                        zest = numpy.vstack(pe.pt.traj[i].pa.part)[:,1:5].T
                        thetaest = 25.0+C_theta.dot(zest)
                        avg[1,i] = wmean(pe.pt.traj[i].pa.w, thetaest)
                        
                    theta = 25.0+C_theta.dot(z.reshape((4,-1)))
                    sqr_err_eta = (avg[0,:] - e[0,:])**2
                    sqr_err_theta = (avg[1,:] - theta)**2
                            
                    rmse_eta[k, ind] = numpy.sqrt(numpy.mean(sqr_err_eta))
                    rmse_theta[k, ind] = numpy.sqrt(numpy.mean(sqr_err_theta))
                    
            for ind, pc in enumerate(part_count):
                print "%d: (%f, %f)" % (pc, numpy.mean(rmse_eta[:,ind]), numpy.mean(rmse_theta[:,ind]))
                
            
    else:
    

    
        # Create arrays for storing some values for later plotting    
        vals = numpy.zeros((2, num+1, steps+1))
    
        plt.ion()

        # Create reference
        numpy.random.seed(1)
        #numpy.random.seed(86)
        (y, e, z) = generate_dataset(steps)
        # Store values for last time-step aswell    
    
        
        x = numpy.asarray(range(steps+1))
        model = ParticleLSB()
        # Create an array for our particles 
        ParamEstimator = param_est.ParamEstimation(model=model, u=None, y=y)
        ParamEstimator.simulate(num, nums, res=0.67, filter='PF', smoother='mcmc')

        
        svals = numpy.zeros((2, nums, steps+1))
        vals = numpy.zeros((2, num, steps+1))
 
        for i in range(steps+1):
            (xil, zl, Pl) = model.get_states(ParamEstimator.straj.straj[i])
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