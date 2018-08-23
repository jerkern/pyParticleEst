"""
Helper functions for computing some of the heavy parts when using MLNLG and LTV
models
"""

import scipy.linalg as lalg
import numpy as np

from builtins import range


def compute_logprod_derivative(Alup, dA, B, dB):
    """ I = logdet(A)+Tr(inv(A)*B)
        dI/dx = Tr(inv(A)*(dA - dA*inv(A)*B + dB) """

    tmp = lalg.cho_solve(Alup, B, check_finite=False)
    tmp2 = dA + dB - dA.dot(tmp)
    return np.trace(lalg.cho_solve(Alup, tmp2, check_finite=False))

# def compute_l2_grad_f_slow(N, lenp, dim, out, perr, f_grad, tmp):
#    diff_l2 = np.zeros((N, lenp, dim, dim))
#    if (f_grad is not None):
#        for i in xrange(N):
#            for j in xrange(lenp):
#                tmp = f_grad[i][j].dot(perr[i].T)
#                diff_l2[i,j,:,:] -= tmp + tmp.T
#    out += diff_l2


def compute_l2_grad_f(N, lenp, dim, out, perr, f_grad, tmp):

    for i in range(N):
        for j in range(lenp):
            #f_grad[i][j].dot(perr[i].T, tmp)
            #out[i,j,:,:] += -tmp - tmp.T
            for k in range(dim):
                for l in range(dim):
                    tmp[k, l] = f_grad[i, j, k, 0] * perr[i, l, 0]
            for k in range(dim):
                for l in range(dim):
                    out[i, j, k, l] += -tmp[k, l] - tmp[l, k]


def compute_l2_grad_A(N, lenp, dim,
                      out, perr, lxi, Pn, zl, Pl, M, A, A_grad, tmp1, tmp2):
    # tmp1 ~ (dim, dim)
    # tmp2 ~(dim, dim-lxi)

    for i in range(N):
        for j in range(lenp):

            #A_grad[i,j].dot(zl[i],tmp2[:,0]) (dim,1)
            for k in range(dim):
                tmp2[k, 0] = 0.0
                for m in range(dim - lxi):
                    tmp2[k, 0] += A_grad[i, j, k, m] * zl[i, m, 0]

            #tmp2[:,0].dot(perr[i].T, tmp1) (dim, dim)
            for k in range(dim):
                for l in range(dim):
                    tmp1[k, l] = tmp2[k, 0] * perr[i, l, 0]

            #out[i,j] += -tmp1 - tmp1.T
            for k in range(dim):
                for l in range(dim):
                    out[i, j, k, l] += -tmp1[k, l] - tmp1[l, k]

            #A_grad[i][j].dot(Pl[i], tmp2) (dim, dim-lxi)
            for k in range(dim):
                for l in range(dim - lxi):
                    tmp2[k, l] = 0.0
                    for m in range(dim - lxi):
                        tmp2[k, l] += A_grad[i, j, k, m] * Pl[i, m, l]

            #tmp2.dot(A[i].T,tmp1) (dim, dim)
            for k in range(dim):
                for l in range(dim):
                    tmp1[k, l] = 0.0
                    for m in range(dim - lxi):
                        tmp1[k, l] += tmp2[k, m] * A[i, l, m]

            #diff_l2[i,j,:,:] += tmp1 + tmp1.T
            for k in range(dim):
                for l in range(dim):
                    out[i, j, k, l] += tmp1[k, l] + tmp1[l, k]

            #A_grad[i][j].dot(M[i],tmp2[:,:]) (dim, dim-lxi)
            for k in range(dim):
                for l in range(dim - lxi):
                    tmp2[k, l] = 0.0
                    for m in range(dim - lxi):
                        tmp2[k, l] += A_grad[i, j, k, m] * M[i, m, l]

            #diff_l2[i,j,:,lxi:] +=  -tmp2
            #diff_l2[i,j,lxi:,:] += -tmp2.T
            for k in range(dim):
                for l in range(dim - lxi):
                    out[i, j, k, (lxi + l)] += -tmp2[k, l]
                    out[i, j, (lxi + l), k] += -tmp2[k, l]

            #diff_l2[i,j,lxi:,lxi:] += -tmp2[lxi:,:]
#            for k in range(dim-lxi):
#                for l in range(dim-lxi):
#                    out[i,j,<unsigned int>(lxi+k),<unsigned int>(lxi+l)] += Pn[i,k,l]


def compute_pred_err(N, dim, xn, f, A, zl, out):
    for i in range(N):
        out[i] = xn[i] - f[i] - A[i].dot(zl[i])


def compute_l2(N, lxi, dim, perr, Pn, A, Pl, M, out):
    for i in range(N):

        out[i] = perr[i].dot(perr[i].T) + A[i].dot(Pl[i]).dot(A[i].T)

        #Axi = A[i][:lxi]
        #Az = A[i][lxi:]

        #tmp = -Axi.dot(M[i])
        #out[i,lxi:,:lxi] += tmp.T
        #out[i,:lxi,lxi:] += tmp
        tmp = -A[i].dot(M[i])
        out[i, :, lxi:] += tmp
        out[i, lxi:, :] += tmp.T

        #tmp2 = Pn[i] - M[i].T.dot(Pl[i]) - Az.dot(M[i])
        #out[i, lxi:,lxi:] += tmp2
        out[i, lxi:, lxi:] += Pn[i]
