import numpy
import pyparticleest.models.ltv
import pyparticleest.paramest.param_est as param_est
import matplotlib.pyplot as plt

ka1_ind = 0
ka2_ind = 1
kd_ind = 2
vi_ind = 3
Icl_ind = 4

def calc_A(T, mb, m1, ka1, ka2, kd, Icl, vi):
    m2 = 0.6*Icl/(0.6*vi*mb)
    m3 = m1*0.6/(1-0.6)
    m4 = 0.4*Icl/(vi*mb)
    A = numpy.asarray(((1.0 - T*(kd+ka1), 0.0, 0.0, 0.0),
                       (T*kd, 1.0-T*ka2, 0.0, 0.0),
                       (T*ka1, T*ka2, 1.0-T*(m2+m4), T*m1),
                       (0.0, 0.0, T*m2, 1.0-T*(m1+m3))))
    return A

def calc_f(T, Isc1b, Isc2b, Ipb, Ilb, u):
    f = numpy.asarray(((T*(Isc1b+u)), T*Isc2b, T*Ipb, T*Ilb)).reshape((4,1))
    return f

def calc_A_f(params, u):
    T=1
    mb = 70
    Iirb = 1.0
    m1 = 0.1766
    
    
    ka1 = params[ka1_ind]
    ka2 = params[ka2_ind]
    kd = params[kd_ind]
    Icl = params[Icl_ind]
    vi = params[vi_ind]
    Isc1b = Iirb/(kd+ka1)
    Isc2b = kd/ka2*Isc1b
    
    m2 = 0.6*Icl/(0.6*vi*mb)
    m3 = m1*0.6/(1-0.6)
    m4 = 0.4*Icl/(vi*mb)
    
    Ipb = Iirb / (m2+m2-m1*m2/(m1+m3))
    Ilb = Ipb*m2/(m1+m3)
    
    A = calc_A(T=T, mb=mb, m1=m1,
               ka1 = ka1, ka2 = ka2,
               kd = kd, vi = vi,
               Icl = Icl)
    f = calc_f(T=T, Isc1b=Isc1b, Isc2b=Isc2b, Ilb=Ilb, Ipb=Ipb, u=u)
    return (A, f)
 
def calc_C(params):
    C = numpy.asarray((0.0, 0.0, 1.0/params[vi_ind], 0.0)).reshape((1,4))
    return C

def generate_dataset(steps, z0, P0, Q, R, params):
    params = numpy.exp(params)
    x = numpy.zeros((steps+1,4,1))
    y = numpy.zeros((steps+1,1,1))
    u = numpy.zeros((steps+1,1,1))
    C = calc_C(params)
    x[0] = numpy.random.multivariate_normal(z0.ravel(), P0).reshape((-1,1))
    y[0] = C.dot(x[0]) + numpy.random.multivariate_normal((0.0,), R).reshape((-1,1))
    
    for k in range(0, steps):
        u[k] = numpy.random.multivariate_normal((0.0, ), ((1.0,),)).reshape((-1,1))
    
    for k in range(0,steps):
        (A, f) = calc_A_f(params, u[k])
                          
        x[k+1] = A.dot(x[k]) + f + numpy.random.multivariate_normal(numpy.zeros(4), Q).reshape((-1,1))
        y[k+1] = C.dot(x[k+1]) + numpy.random.multivariate_normal((0.0,), R).reshape((-1,1))
        
    return (x,y, u)
 
class Model(pyparticleest.models.ltv.LTV):
    
    def __init__(self, params, z0, P0, Q, R):
        params = numpy.exp(params)
        C = calc_C(params)
        (A, f) = calc_A_f(params, 0)
        self.params = numpy.copy(params).ravel()
        self.C = C
        self.A = A
        self.f = f
        super(Model, self).__init__(A=A, C=C, 
                                    z0=z0, P0=P0,
                                    Q=Q, R=R)
    
    
    def set_params(self, params):
        params = numpy.exp(params)
        C = calc_C(params)
        (A, f) = calc_A_f(params, 0)
        self.params = numpy.copy(params).ravel()
        self.kf.set_dynamics(A=A, C=C, f_k=f)
        
    def get_pred_dynamics(self, t, u):
        (A, f) = calc_A_f(self.params, u)
        return (None, f, None)
    

def callback(params, Q):
    print "params = %s" % numpy.exp(params)
    

if __name__ == '__main__':
    steps = 200
    num = 1
    M = 1
    tests = 5
    max_iter = 20
    tol=0.01
    z0 = numpy.asarray((0.0, 0.0, 0.0, 0.0))
    P0 = 1000.0*numpy.eye(4)
    Q = 0.1*numpy.eye(4)
    R = 0.1*numpy.eye(1)
    params = numpy.log(numpy.asarray((0.004, 0.0182, 0.0164, 0.05, 1.1069)))
    (x, y, u) = generate_dataset(steps, z0, P0, Q, R, params)
    model = Model(params, z0, P0, Q, R)
    estimator = param_est.ParamEstimation(model, u=u, y=y)
    
    Q_vec = numpy.empty(tests+1)
    param_vec = numpy.empty((tests+1, len(params)))
    
    plt.ion()
    
    def callback_sim(estimator):
        plt.figure(1)
        plt.clf()
        C = calc_C(numpy.exp(estimator.params))
        plt.plot(range(steps+1),x[:,0,0],'r-')
        plt.plot(range(steps+1),x[:,1,0],'g-')
        plt.plot(range(steps+1),x[:,2,0],'b-')
        plt.plot(range(steps+1),x[:,3,0],'k-')
        plt.plot(range(steps+1),y[:,0,0],'bx')
        plt.plot(range(steps+1),estimator.straj.traj[:,0,0],'r--')
        plt.plot(range(steps+1),estimator.straj.traj[:,0,1],'g--')
        plt.plot(range(steps+1),estimator.straj.traj[:,0,2],'b--')
        plt.plot(range(steps+1),estimator.straj.traj[:,0,3],'k--')
        plt.plot(range(steps+1),C.dot(estimator.straj.traj[:,0,:4,numpy.newaxis])[0,:,0],'ro')
        plt.plot(range(steps+1),y[:,0,0],'bx')
        plt.draw()
        plt.show()
        
    #estimator.simulate(1, 1, meas_first=True)
    
#    estimator.maximize(params, 1, 1, meas_first=True, max_iter=10,
#                       callback_sim=callback_sim, callback=callback)
    numpy.set_printoptions(precision=5)
    callback(params, None)
    (param_vec[0], Q_vec[0]) = estimator.maximize(params, 1, 1, meas_first=True, max_iter=max_iter, tol=tol)
    callback(param_vec[0], Q_vec[0])
    for k in xrange(1,tests+1):
    
        params0 = numpy.random.uniform(params-1.0*numpy.abs(params), params+1.0*numpy.abs(params))
        print "%s -> " % numpy.exp(params0),
        (param_vec[k], Q_vec[k]) = estimator.maximize(params0, 1, 1, meas_first=True, max_iter=max_iter, tol=tol)
#    (param_vec[k], Q_vec[k]) = estimator.maximize(params0, 1, 1, meas_first=True, max_iter=500,
#                                                  callback=callback, tol=0.0001, callback_sim=callback_sim)
        print "%s : %.02f" % (numpy.exp(param_vec[k]), Q_vec[k])
        callback_sim(estimator)
        
    plt.ioff()
    sorted = numpy.argsort(Q_vec)
    print "result:"
    for ind in sorted:
        if (ind == 0):
            print "*",
        print "Q=%.02f params=%s" % (Q_vec[ind] ,numpy.exp(param_vec[ind]))
    
    estimator.set_params(param_vec[sorted[-1]])
    estimator.simulate(1, 1, meas_first=True)
    callback_sim(estimator)

    
    
    
    