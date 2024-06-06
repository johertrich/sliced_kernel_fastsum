# This code belongs to the paper
#
# J. Hertrich, 2024.
# Fast Kernel Summation in High Dimensions via Slicing and Fourier Transforms.
# arxiv preprint 2401.08260
#
# Please cite the paper, if you use this code.

import torch
import math
import numpy as np
import scipy, scipy.special
import mpmath

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float

def differences(p,q):
    dim = p.shape[1]
    m_p, m_q = p.shape[0], q.shape[0]
    diff = torch.zeros((m_p,m_q,dim),dtype=dtype,device=device)
    diff = p.reshape(m_p,1,dim) - q.reshape(1,m_q,dim)
    return diff
    
def distance(p, q):
    diff = differences(p, q)
    out=torch.linalg.vector_norm(diff,ord=2,dim=2)
    return out
    
def Gaussian_kernel_mat(x,y,sigma_sq,scale=1.):
    # compute kernel matrix for Gaussian kernel
    dist=torch.sum(differences(x,y)**2,-1)
    return torch.exp(- dist / (2*sigma_sq*scale**2))
    
def Laplacian_kernel_mat(x,y,alpha,scale=1.):
    # compute kernel matrix for Gaussian kernel
    dist=torch.sqrt(torch.sum(differences(x,y)**2,-1))
    return torch.exp(- alpha*dist / scale)
    
def Matern_kernel_mat(x,y,alpha,nu,scale=1.):
    # compute kernel matrix for Gaussian kernel
    # does not support backprobagation
    if nu==1.5:
        return Matern32_kernel_mat(x,y,alpha,scale=scale)
    if nu==.5:
        return Laplacian_kernel_mat(x,y,alpha,scale=scale)
    # evaluate power series if nu not in {.5,1.5}
    dist=torch.sqrt(torch.sum(differences(x,y)**2,-1))
    out=matern_kernel_fun(dist/scale,alpha,nu,1)
    return out

def Matern32_kernel_mat(x,y,alpha,scale=1.):
    # compute kernel matrix for Gaussian kernel
    # does not support backprobagation
    dist=torch.sqrt(torch.sum(differences(x,y)**2,-1))
    arg=np.sqrt(3.)*dist/(scale*alpha)
    out=(1+arg)*torch.exp(-arg)
    return out

def Gaussian_der_kernel_mat(x,y,sigma_sq,scale=1.):
    # compute kernel matrix for the derivative of the Gaussian kernel
    diff=differences(x,y)
    dist=torch.sum(diff**2,-1)
    return -diff/(scale*sigma_sq)*torch.exp(- dist[:,:,None] / (2*sigma_sq*scale**2))

def Gaussian_kernel_fun_ft(grid,d,sigma_sq):
    # implementation of the Fourier transform of the one-dimensional counterpart of the Gaussian kernel
    # for numerical stability computations are done in the log-space
    k = grid
    args=2*math.pi**2*sigma_sq*k**2
    log_args=torch.log(args)
    log_args[args==0]=0
    factor=d*math.pi*torch.sqrt(sigma_sq/2)
    log_out=log_args*.5*(d-1)-args-scipy.special.loggamma((d+2)/2)
    out=torch.exp(log_out)
    if d>1:
        out[args==0]=0
    else:
        out[args==0]=1/scipy.special.gamma((d+2)/2)
    return out*factor

def Laplace_kernel_fun(grid,d,alpha):
    # implementation for the one-dimensional counterpart of the Laplace kernel.
    # This function is slow. Avoid calling it several times, use interpolations instead.
    abs_grid_np=np.abs(grid)
    mpfun=lambda x:float(mpmath.hyp1f2(.5*d,.5,.5,(alpha*x)**2/4)-np.sqrt(math.pi)*alpha*x*np.exp(scipy.special.loggamma((d+1)/2)-scipy.special.loggamma(d/2))*mpmath.hyp1f2(.5*d+.5,1.,1.5,(alpha*x)**2/4))
    mpfun_vec=np.vectorize(mpfun)
    out=mpfun_vec(abs_grid_np)
    return out


def matern_kernel_fun(x,alpha,nu,d):
    # implementation for the one-dimensional counterpart of the Matern kernel.
    out_all=torch.zeros_like(x)
    relevant=torch.logical_and((x/alpha) < 2.5,(x/alpha)>1e-5)
    x_relevant=x[relevant]
    
    # go to double precision...
    out=torch.zeros_like(x_relevant).type(torch.float64)
    factor=math.pi**1.5/(scipy.special.gamma(nu)*2**nu*np.sin(nu*math.pi))
    diff=torch.tensor(1.,dtype=torch.float64,device=device)
    n=0
    log_g_d2=scipy.special.loggamma(.5*d)
    log_x=torch.log(x_relevant).type(torch.float64)
    while (torch.max(torch.abs(diff))>1e-5 and n<100):
        if n+1>nu:
            log_factors1=scipy.special.loggamma((2*n+d)/2)+n*np.log(nu)-scipy.special.loggamma((2*n+1)/2)-log_g_d2-scipy.special.loggamma(n+1)-scipy.special.loggamma(n-nu+1)-(n-nu)*np.log(2.)-2*n*np.log(alpha)
            addi1=2*n*log_x+log_factors1
            diff1=torch.exp(addi1)
            sign=1.
        else:
            log_factors1=scipy.special.loggamma((2*n+d)/2)+n*np.log(nu)-scipy.special.loggamma((2*n+1)/2)-log_g_d2-scipy.special.loggamma(n+1)-(n-nu)*np.log(2.)-2*n*np.log(alpha)
            div=scipy.special.gamma(n-nu+1)
            sign=div/np.abs(div)
            addi1=2*n*log_x+log_factors1-np.log(div*sign)
            diff1=torch.exp(addi1)
            
        log_factors2=scipy.special.loggamma((2*n+2*nu+d)/2)+(n+nu)*np.log(nu)-log_g_d2-scipy.special.loggamma(n+1)-scipy.special.loggamma((2*n+2*nu+1)/2)-scipy.special.loggamma(n+nu+1)-n*np.log(2.)-2*(n+nu)*np.log(alpha)
        addi2=2*(n+nu)*log_x+log_factors2
        diff2=torch.exp(addi2)
        
        if sign<0:
            log_neg_diff=torch.logsumexp(torch.stack((addi1,addi2),-1),-1)
            diff=-torch.exp(log_neg_diff)
        else:
            diff=diff1-diff2
        out=out+diff
        n=n+1
        
    out=factor*out
    out_all[relevant]=out.type(dtype)
    out_all[(x/alpha) <= 1e-5]=1.
    return out_all

    
def Laplace_kernel_fun_1d(x,alpha,d):
    # implementation for the one-dimensional counterpart of the Laplace kernel.
    # does the same as matern_kernel_fun(x,1/alpha,.5,d) but faster and more stable
    out_all=torch.zeros_like(x)
    relevant=torch.logical_and((x*alpha) < 3.5,(x*alpha)>1e-5)
    x_relevant=x[relevant]*alpha
    
    # go to double precision...
    out=torch.zeros_like(x_relevant).type(torch.float64)
    diff=torch.tensor(1.,dtype=torch.float64,device=device)
    n=0
    log_g_d2=scipy.special.loggamma(.5*d)
    log_x=torch.log(x_relevant).type(torch.float64)
    while (torch.max(torch.abs(diff))>1e-5 and n<100):
        log_factor=.5*np.log(math.pi)+scipy.special.loggamma(.5*(n+d))-scipy.special.loggamma(n+1)-log_g_d2-scipy.special.loggamma(.5*(n+1))
        addi=n*log_x+log_factor
        if n%2==0:
            diff=torch.exp(addi)
        else:
            diff=-torch.exp(addi)
        out=out+diff
        n=n+1
    out=out.type(dtype)
    out_all[relevant]=out
    out_all[(x*alpha) <= 1e-5]=1.
    return out_all
   
