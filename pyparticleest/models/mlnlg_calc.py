import numpy

def compute_l2_grad_f_slow(N, lenp, dim, out, perr, f_grad, tmp):
    diff_l2 = numpy.zeros((N, lenp, dim, dim))
    if (f_grad != None):
        for i in xrange(N):
            for j in xrange(lenp):
                tmp = f_grad[i][j].dot(perr[i].T)
                diff_l2[i,j,:,:] -= tmp + tmp.T
    out += diff_l2

def compute_l2_grad_f(N, lenp, dim, out, perr, f_grad, tmp):
    for i in range(N):
        for j in range(lenp):
            #f_grad[i][j].dot(perr[i].T, tmp)
            #out[i,j,:,:] += -tmp - tmp.T
            for k in range(dim):
                for l in range(dim):
                    tmp[k,l] = f_grad[i,j,k]*perr[i,l,0]
            for k in range(dim):
                for l in range(dim):
                    out[i,j,k,l] += -tmp[k,l] -tmp[l,k]

def compute_l2_grad_A_slow(N, lenp, dim, out, perr, lxi, zl, Pl, M, A, A_grad, tmp1, tmp2):
    diff_l2 = numpy.zeros((N, lenp, dim, dim))
    tmp3 = numpy.zeros((dim,1))
    if (A_grad != None):
        for i in xrange(N):
            for j in xrange(lenp):
                A_grad[i][j].dot(zl[i],tmp3)
                tmp3.dot(perr[i].T,tmp1)
                diff_l2[i,j,:,:] += -tmp1 - tmp1.T
                A_grad[i][j].dot(Pl[i], tmp2)
                tmp2.dot(A[i].T,tmp1)
                diff_l2[i,j,:,:] += tmp1 + tmp1.T
                A_grad[i][j][:lxi,:].dot(M[i],tmp2[:lxi,:])
                diff_l2[i,j,:lxi,lxi:] +=  -tmp2[:lxi,:]
                diff_l2[i,j,lxi:,:lxi] += -tmp2[:lxi,:].T
                A_grad[i][j,lxi:].dot(M[i], tmp2[lxi:,:])
                diff_l2[i,j,lxi:,lxi:] += -tmp2[lxi:,:]
    out += diff_l2
 
def compute_l2_grad_A(N, lenp, dim, out, perr, lxi, zl, Pl, M, A, A_grad, tmp1, tmp2):
    # tmp1 ~ (dim, dim)
    # tmp2 ~(dim, dim-lxi)
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
           
            #A_grad[i][j][:lxi,:].dot(M[i],tmp2[:lxi,:]) (lxi, dim-lxi)
            for k in range(lxi):
                for l in range(dim-lxi):
                    tmp2[k,l] = 0.0
                    for m in range(dim-lxi):
                        tmp2[k,l] += A_grad[i,j,k,m]*M[i,m,l]
            
            #diff_l2[i,j,:lxi,lxi:] +=  -tmp2[:lxi,:]
            #diff_l2[i,j,lxi:,:lxi] += -tmp2[:lxi,:].T                      
            for k in range(lxi):
                for l in range(dim-lxi):
                    out[i,j,k,lxi+l] += -tmp2[k,l]
                    out[i,j,lxi+l,k] += -tmp2[k,l]
            
            #A_grad[i][j,lxi:].dot(M[i], tmp2[lxi:,:]) (dim-lxi, dim-lxi)
            for k in range(dim-lxi):
                for l in range(dim-lxi):
                    tmp2[lxi+k,l] = 0.0
                    for m in range(dim-lxi):
                        tmp2[lxi+k,l] += A_grad[i,j,lxi+k,m]*M[i,m,l]
            
            #diff_l2[i,j,lxi:,lxi:] += -tmp2[lxi:,:]
            for k in range(dim-lxi):
                for l in range(dim-lxi):
                    out[i,j,lxi+k,lxi+l] += -tmp2[lxi+k,l]
            
#compute_l2_grad_f_jit = numba.jit('void(int64, int64, int64, double[:,:,:,:], double[:,:,:], double[:,:,:], double[:,:])',
#                                  nopython=True)(compute_l2_grad_f)
#compute_l2_grad_A_jit = numba.jit('void(int64, int64, int64, double[:,:,:,:], double[:,:,:], int64, double[:,:,:], double[:,:,:], double[:,:,:], double[:,:,:], double[:,:,:,:], double[:,:], double[:,:])',
#                                  nopython=True)(compute_l2_grad_A)