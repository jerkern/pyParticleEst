import numpy as np
import scipy.linalg as lalg
cimport numpy as np
cimport cython

def compute_logprod_derivative(Alup, dA, B, dB):
    """ I = logdet(A)+Tr(inv(A)*B)
        dI/dx = Tr(inv(A)*(dA - dA*inv(A)*B + dB) """
        
    tmp = lalg.cho_solve(Alup, B, check_finite=False)
    tmp2 = dA + dB - dA.dot(tmp)
    return np.trace(lalg.cho_solve(Alup,tmp2, check_finite=False))

#def compute_l2_grad_f_slow(N, lenp, dim, out, perr, f_grad, tmp):
#    diff_l2 = np.zeros((N, lenp, dim, dim))
#    if (f_grad != None):
#        for i in xrange(N):
#            for j in xrange(lenp):
#                tmp = f_grad[i][j].dot(perr[i].T)
#                diff_l2[i,j,:,:] -= tmp + tmp.T
#    out += diff_l2

@cython.boundscheck(False) 
def compute_l2_grad_f(unsigned int N, unsigned int lenp, unsigned int dim,
                      np.ndarray[np.double_t, ndim=4] out,
                      np.ndarray[np.double_t, ndim=3] perr,
                      np.ndarray[np.double_t, ndim=4] f_grad,
                      np.ndarray[np.double_t, ndim=2] tmp):
    cdef unsigned int i, j, k, l
    for i in range(N):
        for j in range(lenp):
            #f_grad[i][j].dot(perr[i].T, tmp)
            #out[i,j,:,:] += -tmp - tmp.T
            for k in range(dim):
                for l in range(dim):
                    tmp[k,l] = f_grad[i,j,k,0] * perr[i,l,0]
            for k in range(dim):
                for l in range(dim):
                    out[i,j,k,l] += -tmp[k,l] -tmp[l,k]

#def compute_l2_grad_A_slow(N, lenp, dim, out, perr, lxi, zl, Pl, M, A, A_grad, tmp1, tmp2):
#    diff_l2 = np.zeros((N, lenp, dim, dim))
#    tmp3 = np.zeros((dim,1))
#    if (A_grad != None):
#        for i in xrange(N):
#            for j in xrange(lenp):
#                A_grad[i][j].dot(zl[i],tmp3)
#                tmp3.dot(perr[i].T,tmp1)
#                diff_l2[i,j,:,:] += -tmp1 - tmp1.T
#                A_grad[i][j].dot(Pl[i], tmp2)
#                tmp2.dot(A[i].T,tmp1)
#                diff_l2[i,j,:,:] += tmp1 + tmp1.T
#                A_grad[i][j][:lxi,:].dot(M[i],tmp2[:lxi,:])
#                diff_l2[i,j,:lxi,lxi:] +=  -tmp2[:lxi,:]
#                diff_l2[i,j,lxi:,:lxi] += -tmp2[:lxi,:].T
#                A_grad[i][j,lxi:].dot(M[i], tmp2[lxi:,:])
#                diff_l2[i,j,lxi:,lxi:] += -tmp2[lxi:,:]
#    out += diff_l2

@cython.boundscheck(False) 
def compute_l2_grad_A(unsigned int N, unsigned int lenp, unsigned int dim,
                      np.ndarray[np.double_t, ndim=4] out,
                      np.ndarray[np.double_t, ndim=3] perr,
                      unsigned int lxi,
                      np.ndarray[np.double_t, ndim=3] Pn,
                      np.ndarray[np.double_t, ndim=3] zl,
                      np.ndarray[np.double_t, ndim=3] Pl,
                      np.ndarray[np.double_t, ndim=3] M,
                      np.ndarray[np.double_t, ndim=3] A,
                      np.ndarray[np.double_t, ndim=4] A_grad,
                      np.ndarray[np.double_t, ndim=2] tmp1,
                      np.ndarray[np.double_t, ndim=2] tmp2):
    # tmp1 ~ (dim, dim)
    # tmp2 ~(dim, dim-lxi)
    cdef unsigned int i, j, k, l, m
    for i in xrange(N):
        for j in xrange(lenp):
            
            #A_grad[i,j].dot(zl[i],tmp2[:,0]) (dim,1)
            for k in range(dim):
                tmp2[k, 0] = 0.0
                for m in range(dim-lxi):
                    tmp2[k, 0] += A_grad[i,j,k,m] * zl[i,m,0]
            
            #tmp2[:,0].dot(perr[i].T, tmp1) (dim, dim)
            for k in range(dim):
                for l in range(dim):
                    tmp1[k, l] = tmp2[k,0] * perr[i,l,0]
            
            #out[i,j] += -tmp1 - tmp1.T       
            for k in range(dim):
                for l in range(dim):            
                    out[i,j,k,l] += -tmp1[k,l] - tmp1[l,k]
            
            #A_grad[i][j].dot(Pl[i], tmp2) (dim, dim-lxi)            
            for k in range(dim):
                for l in range(dim-lxi):
                    tmp2[k,l] = 0.0
                    for m in range(dim-lxi):
                        tmp2[k,l] += A_grad[i,j,k,m]*Pl[i,m,l]
            
            #tmp2.dot(A[i].T,tmp1) (dim, dim)
            for k in range(dim):
                for l in range(dim):
                    tmp1[k,l] = 0.0
                    for m in range(dim-lxi):
                        tmp1[k,l] += tmp2[k,m]*A[i,l,m]
            
                  
            #diff_l2[i,j,:,:] += tmp1 + tmp1.T
            for k in range(dim):
                for l in range(dim):            
                    out[i,j,k,l] += tmp1[k,l] + tmp1[l,k]
           
            #A_grad[i][j].dot(M[i],tmp2[:,:]) (dim, dim-lxi)
            for k in range(dim):
                for l in range(dim-lxi):
                    tmp2[k,l] = 0.0
                    for m in range(dim-lxi):
                        tmp2[k,l] += A_grad[i,j,k,m]*M[i,m,l]
                        
            #diff_l2[i,j,:,lxi:] +=  -tmp2
            #diff_l2[i,j,lxi:,:] += -tmp2.T                      
            for k in range(dim):
                for l in range(dim-lxi):
                    out[i,j,k,<unsigned int>(lxi+l)] += -tmp2[k,l]
                    out[i,j,<unsigned int>(lxi+l),k] += -tmp2[k,l]
            
            #diff_l2[i,j,lxi:,lxi:] += -tmp2[lxi:,:]
#            for k in range(dim-lxi):
#                for l in range(dim-lxi):
#                    out[i,j,<unsigned int>(lxi+k),<unsigned int>(lxi+l)] += Pn[i,k,l]
            
def compute_pred_err(N, dim, xn, f, A, zl, out):
    for i in xrange(N):
        out[i] = xn[i] - f[i] - A[i].dot(zl[i])

def compute_l2(N, lxi, dim, perr, Pn, A, Pl, M, out):
    for i in xrange(N):

        out[i] = perr[i].dot(perr[i].T) + A[i].dot(Pl[i]).dot(A[i].T)
        
        #Axi = A[i][:lxi]
        #Az = A[i][lxi:]
        
        #tmp = -Axi.dot(M[i])
        #out[i,lxi:,:lxi] += tmp.T
        #out[i,:lxi,lxi:] += tmp
        tmp = -A[i].dot(M[i])
        out[i,:,lxi:] += tmp
        out[i,lxi:,:] += tmp.T
        
        
        #tmp2 = Pn[i] - M[i].T.dot(Pl[i]) - Az.dot(M[i])        
        #out[i, lxi:,lxi:] += tmp2
        out[i, lxi:,lxi:] += Pn[i]

#def compute_l2_grad(perr, lenp, lxi, zl, Pl, M, A, f_grad, A_grad):
#    N = perr.shape[0]
#    diff_l2 = np.zeros((N, lenp, perr.shape[1], perr.shape[1]))
#    if (f_grad != None):
#        for i in xrange(N):
#            for j in xrange(lenp):
#                tmp = f_grad[i][j].dot(perr[i].T)
#                diff_l2[i,j,:,:] -= tmp + tmp.T
#    if (A_grad != None):
#        for i in xrange(N):
#            for j in xrange(lenp):
#                tmp = A_grad[i][j].dot(zl[i]).dot(perr[i].T)
#                diff_l2[i,j,:,:] -= tmp + tmp.T
#                tmp = A_grad[i][j].dot(Pl[i]).dot(A[i].T)
#                diff_l2[i,j,:,:] += tmp + tmp.T
#                tmp = -A_grad[i][j,:lxi].dot(M[i])
#                diff_l2[i,j,:lxi,lxi:] +=  tmp
#                diff_l2[i,j,lxi:,:lxi] += tmp.T
#                tmp = -A_grad[i][j,lxi:].dot(M[i])
#                diff_l2[i,j,lxi:,lxi:] += tmp
#    return diff_l2