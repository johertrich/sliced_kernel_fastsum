# This code belongs to the paper
#
# J. Hertrich, 2024.
# Fast Kernel Summation in High Dimensions via Slicing and Fourier Transforms.
# arxiv ...
#
# Please cite the paper, if you use this code.

import numpy as np
import torch
from scipy.fft import fft
from kernels import *
import time
import pykeops.torch

nfft_loaded=True
try:
    import torch_nfft as tn
except:
    print('Module torch_nfft not found. Use NDFT!')
    nfft_loaded=False
    


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype=torch.float



def ndft_adjoint(x, pos, N=16,grid=None):
    # Adjoint discrete Fourier transform with nonequispaced data x located on pos
    # grid defines the indices of the Fourier coefficients which we want to compute
    # shared basis points means that we compute several Fourier transforms with the same pos 
    # (needed for SVGD, build matrix only once)
    device = pos.device
    x = x.to(torch.cfloat)

    if grid is None:
        grid1d = torch.arange(-N/2, N/2, dtype=torch.float, device=device)
        grid=grid1d[:,None]
    else:
        N=grid.shape[0]

    def single_ndft(x_part, pos_part):
        # build matrix
        fourier_tensor = torch.exp(2j * torch.pi * grid*pos_part.T)
        y = torch.matmul(fourier_tensor, x_part)
        return y.T

    assert x.shape[1]==pos.shape[1], "shape of weights and basis points must agree"
    return torch.cat([single_ndft(x[:,idx:idx+1], pos[:,idx:idx+1]) for idx in range(pos.shape[1])])
    
def ndft_forward(x, pos, P,grid=None):
    # Discrete Fourier transform with nonequispaced Fourier coefficients with indices grid given by x
    # pos defines the points, where we want to evaluate the function
    # shared basis points means that we compute several Fourier transforms with the same pos 
    # (needed for SVGD, build matrix only once)
    n = pos.shape[0]
    device = pos.device
    x = x.to(torch.cfloat)
    N=x.shape[1]

    if grid is None:
        grid1d = torch.arange(-N/2, N/2, dtype=torch.float, device=device)
        grid=grid1d[:,None]
    else:
        N=grid.shape[0]

    def single_ndft(x_part, pos_part):
        fourier_tensor = torch.exp(-2j * torch.pi * grid.T*pos_part)
        y = torch.matmul(fourier_tensor, x_part.T)
        return y.T
    
    return torch.cat([single_ndft(x[idx:idx+1,:], pos[:,idx:idx+1]) for idx in range(pos.shape[1])])

def fastsum(kernel_ft,x,y,x_weights=None,grid=None):
    # Fast summation with adjoint and forward NDFT
    if x_weights is None:
        x_weights=torch.ones_like(x)
    a = ndft_adjoint(x_weights,-x,N=kernel_ft.shape[1],grid=grid)
    d = kernel_ft*a
    fRF = ndft_forward(d,-y,x_weights.shape[1],grid=grid).real.T
    return fRF

def fastsum_fft(kernel_ft,x,y,x_weights=None):
    # Fast summation with adjoint and forward NFFT
    # Only available with CUDA, since torch_nfft is only implemented on CUDA.
    n=kernel_ft.shape[1]
    mx = x.shape[0]
    my = y.shape[0]
    p = x.shape[1] # p_slices
    batch_x = torch.arange(p, device=device).repeat_interleave(mx)
    x_flat = x.T.flatten().unsqueeze(1) 
    batch_x = torch.arange(p, device=device).repeat_interleave(mx)
    x_flat = x.T.flatten().unsqueeze(1) # (mx*p, 1)
    x_weights = x_weights.T
    a = tn.nfft_adjoint(x_weights.flatten(), -x_flat, bandwidth=n, batch=batch_x)
    d = kernel_ft*a
    batch_y = torch.arange(p, device=device).repeat_interleave(my)
    y_flat = y.T.flatten().unsqueeze(1) # (my*p, 1)
    fRF = tn.nfft_forward(d, -y_flat, real_output=True, batch=batch_y) #(p*my,) <- (p, n), (p*my, 1)
    fRF = fRF.reshape(p, my).T # (my, p)
    return fRF
    
def naive_kernel_sum(x,x_weights,y,kernel_mat_fun,batch_size=100):
    # Naive kernel summation via the kernel matrix
    d=x.shape[1]
    N=x.shape[0]
    naive_sum=0
    i=0
    # batching for fitting memory constraints
    while i<N:
        kernel_mat=kernel_mat_fun(x[i:i+batch_size],y)
        naive_sum+=torch.sum(kernel_mat*x_weights[i:i+batch_size,None],0)
        i+=batch_size
    return naive_sum

def fast_Gaussian_summation(x,y,x_weights,sigma_sq,P,n_ft,x_range=0.1):
    # Fast summation for the Gaussian kernel
    d=x.shape[1]
    # In 1D: sliced=unsliced => P=1 is sufficient
    if d==1:
        P=1

    # 1D Projections
    xi=torch.randn((P,d),dtype=dtype,device=device)
    xi=xi/torch.sqrt(torch.sum(xi**2,-1,keepdim=True))
    xi=xi.unsqueeze(1)
    x_proj=torch.nn.functional.conv1d(x.reshape(1,1,-1),xi,stride=d).reshape(P,-1).transpose(0,1)
    y_proj=torch.nn.functional.conv1d(y.reshape(1,1,-1),xi,stride=d).reshape(P,-1).transpose(0,1)

    
    # Rescaling
    xy_proj=torch.cat((x_proj,y_proj),0)
    x_max=torch.max(xy_proj)
    x_min=torch.min(xy_proj)
    x_mid=.5*(x_max+x_min)
    scale_factor=x_range/(x_max-x_min)
    sigma_max=0.05
    if d==2:
        # for small d=2 {_1}F1(d/2,1/2,-x^2/2) converges only very slowly to zero.
        # consequently, we need to rescale the kernel to reduce the error from Poissons summation formula.
        # In this case we will need more Fourier coefficients (since spatial localized kernels have a wide
        # spread in the Fourier domain. That might cause a slow-down for d=2.
        # Consider to append a zero-dimension to all data points and call the function for d=3.
        sigma_max=1e-5
    if sigma_sq*scale_factor**2>sigma_max:
        scale_factor=torch.sqrt(torch.tensor(sigma_max,dtype=dtype,device=device)/sigma_sq)
    x_proj=(x_proj-x_mid)*scale_factor
    y_proj=(y_proj-x_mid)*scale_factor
    sigma_real=sigma_sq*scale_factor**2
    
    
    # compute kernel Fourier coefficients
    grid1d = torch.arange((-n_ft+1)//2, (n_ft+1)//2, dtype=torch.float, device=device)
    kernel_ft=Gaussian_kernel_fun_ft(grid1d,d,sigma_real)

    # select grid of nonzero coefficients
    kernel_not_zero=kernel_ft>1e-8
    kernel_not_zero_inds=kernel_not_zero.nonzero()
    kernel_ft=kernel_ft[kernel_not_zero].reshape(1,-1)
    grid=grid1d[kernel_not_zero]
    grid=grid[:,None]
    # fast Fourier summation
    my_sum=fastsum(kernel_ft,x_proj,y_proj,x_weights=x_weights[:,None].tile(1,P),grid=grid)    
    # Mean over all projections
    return torch.mean(my_sum,1)

def fast_Gaussian_summation_batched(x,y,x_weights,sigma_sq,P,n_ft,x_range=0.1,batch_size=1000):
    # batching over the projections
    P_left=P
    out=0
    while P_left>batch_size:
        out=out+fast_Gaussian_summation(x,y,x_weights,sigma_sq,batch_size,n_ft,x_range=0.1)/P*batch_size
        P_left-=batch_size
    out=out+fast_Gaussian_summation(x,y,x_weights,sigma_sq,P_left,n_ft,x_range=0.1)/P*P_left
    return out
    
    
def energy_distance_1D(x,y):
    # 1D energy distance via cdf, see Szeleky 2002
    N=x.shape[1]
    M=y.shape[1]
    points,inds=torch.sort(torch.cat((x,y),1))
    x_where=torch.where(inds<N,1.,0.).type(dtype)
    F_x=torch.cumsum(x_where,-1)/N
    y_where=torch.where(inds>=N,1.,0.).type(dtype)
    F_y=torch.cumsum(y_where,-1)/M
    mmd=torch.sum(((F_x[:,:-1]-F_y[:,:-1])**2)*(points[:,1:]-points[:,:-1]),-1)
    return mmd

def compute_sliced_factor(d):
    # Compute the slicing constant within the negative distance kernel
    k=(d-1)//2
    fac=1.
    if (d-1)%2==0:
        for j in range(1,k+1):
            fac=2*fac*j/(2*j-1)
    else:
        for j in range(1,k+1):
            fac=fac*(2*j+1)/(2*j)
        fac=fac*math.pi/2
    return fac

def sliced_energy_distance(x,y,n_projections,sliced_factor=None):
    # Compute energy distance via slicing
    d=x.shape[1]
    if sliced_factor is None:
        sliced_factor=compute_sliced_factor(d)
    x=x.reshape(x.shape[0],-1)
    y=y.reshape(y.shape[0],-1)
    xi=torch.randn((n_projections,d),dtype=dtype,device=device)
    xi=xi/torch.sqrt(torch.sum(xi**2,-1,keepdim=True))
    xi=xi.unsqueeze(1)
    x_proj=torch.nn.functional.conv1d(x.reshape(1,1,-1),xi,stride=d).reshape(n_projections,-1)
    y_proj=torch.nn.functional.conv1d(y.reshape(1,1,-1),xi,stride=d).reshape(n_projections,-1)
    mmds=energy_distance_1D(x_proj,y_proj)
    return torch.mean(mmds)*sliced_factor
    
def sliced_MMD_Gauss(x,y,sigma_sq,P=1000,x_range=0.2,n_ft=1024):
    x_weights=torch.ones((x.shape[0],),dtype=dtype,device=device)/x.shape[0]
    y_weights=torch.ones((y.shape[0],),dtype=dtype,device=device)/y.shape[0]
    interaction_energy_x=torch.sum(fast_Gaussian_summation_batched(x,x,x_weights,sigma_sq,P,n_ft,x_range=x_range))
    interaction_energy_y=torch.sum(fast_Gaussian_summation_batched(y,y,y_weights,sigma_sq,P,n_ft,x_range=x_range))
    potential_energy=torch.sum(fast_Gaussian_summation_batched(x,y,x_weights,sigma_sq,P,n_ft,x_range=x_range))
    return .5*interaction_energy_x+.5*interaction_energy_y-potential_energy
    
    
def fastsum_energy_kernel_1D(x,x_weights,y):
    # Sorting algorithm for fast sumation with negative distance (energy) kernel
    N=x.shape[1]
    M=y.shape[1]
    P=x.shape[0]
    # Potential Energy
    sorted_yx,inds_yx=torch.sort(torch.cat((y,x),1))
    inds_yx=inds_yx+torch.arange(P,device=device).unsqueeze(1)*(N+M)
    inds_yx=torch.flatten(inds_yx)
    weights_sorted=torch.cat((torch.zeros_like(y),x_weights),1).flatten()[inds_yx].reshape(P,-1)
    pot0=torch.sum(weights_sorted*(sorted_yx-sorted_yx[:,0:1]),1,keepdim=True)
    yx_diffs=sorted_yx[:,1:]-sorted_yx[:,:-1]
    # Mults from cumsums shifted by 1
    mults_short=torch.sum(x_weights,-1,keepdim=True)-2*torch.cumsum(weights_sorted,1)[:,:-1]
    mults=torch.zeros_like(weights_sorted)
    mults[:,1:]=mults_short
    potential=torch.zeros_like(sorted_yx)
    potential[:,1:]=yx_diffs.clone()
    potential=pot0-torch.cumsum(potential*mults,1)
    out1=torch.zeros_like(sorted_yx).flatten()
    out1[inds_yx]=potential.flatten()
    out1=out1.reshape(P,-1)
    out1=out1[:,:M]
    return out1
    
def fast_energy_summation(x,y,x_weights,P,sliced_factor):
    # fast sum via slicing and sorting
    d=x.shape[1]

    # In 1D: sliced=unsliced => P=1 is sufficient
    if d==1:
        P=1

    # 1D Projections
    xi=torch.randn((P,d),dtype=dtype,device=device)
    xi=xi/torch.sqrt(torch.sum(xi**2,-1,keepdim=True))
    xi=xi.unsqueeze(1)
    x_proj=torch.nn.functional.conv1d(x.reshape(1,1,-1),xi,stride=d).reshape(P,-1).transpose(0,1)
    y_proj=torch.nn.functional.conv1d(y.reshape(1,1,-1),xi,stride=d).reshape(P,-1).transpose(0,1)
    fastsum_energy=fastsum_energy_kernel_1D(x_proj.transpose(0,1),x_weights[None,:].tile(P,1),y_proj.transpose(0,1)).transpose(0,1)
    return sliced_factor*torch.mean(-fastsum_energy,1) 

def fast_energy_summation_batched(x,y,x_weights,P,sliced_factor,batch_size=1000):
    # batching over the projections
    P_left=P
    out=0
    while P_left>batch_size:
        out=out+fast_energy_summation(x,y,x_weights,batch_size,sliced_factor)/P*batch_size
        P_left-=batch_size
    out=out+fast_energy_summation(x,y,x_weights,P_left,sliced_factor)/P*P_left
    return out    
    
def fast_Laplacian_summation(x,y,x_weights,alpha,P,n_ft,sliced_factor,x_range=0.1):
    # fastsum for the Laplacian kernel (with decomposition in energy kernel plus smooth kernel)
    d=x.shape[1]

    # In 1D: sliced=unsliced => P=1 is sufficient
    if d==1:
        P=1

    # 1D Projections
    xi=torch.randn((P,d),dtype=dtype,device=device)
    xi=xi/torch.sqrt(torch.sum(xi**2,-1,keepdim=True))
    xi=xi.unsqueeze(1)
    x_proj=torch.nn.functional.conv1d(x.reshape(1,1,-1),xi,stride=d).reshape(P,-1).transpose(0,1)
    y_proj=torch.nn.functional.conv1d(y.reshape(1,1,-1),xi,stride=d).reshape(P,-1).transpose(0,1)
    # Rescaling
    xy_proj=torch.cat((x_proj,y_proj),0)
    x_max=torch.max(xy_proj)
    x_min=torch.min(xy_proj)
    x_mid=.5*(x_max+x_min)
    scale_factor=x_range/(x_max-x_min)
    x_proj=(x_proj-x_mid)*scale_factor
    y_proj=(y_proj-x_mid)*scale_factor
    alpha_real=alpha/scale_factor
    
    fastsum_energy=fastsum_energy_kernel_1D(x_proj.transpose(0,1),x_weights[None,:].tile(P,1),y_proj.transpose(0,1)).transpose(0,1)
    fastsum_energy=alpha_real*fastsum_energy/sliced_factor

    
    # compute kernel Fourier coefficients
    h = torch.arange((-n_ft+1)//2, (n_ft+1)//2, device=device)
    perm = h%n_ft
    perm_inv = torch.argsort(h%n_ft)
    vect=Laplace_kernel_fun_1d(torch.abs(h/n_ft),alpha_real,d)+alpha_real*torch.abs(h/n_ft)/sliced_factor
    vect_perm = vect[perm_inv]
    kernel_ft= 1/n_ft * torch.fft.fft( vect_perm )[perm] 
    kernel_ft=kernel_ft
    
    # fast Fourier summation
    if nfft_loaded:
        my_sum=fastsum_fft(kernel_ft.reshape(1,-1),x_proj,y_proj,x_weights=x_weights[:,None].tile(1,P))
    else:
        # select grid of nonzero coefficients
        kernel_not_zero=torch.abs(kernel_ft)>0
        kernel_not_zero_inds=kernel_not_zero.nonzero()
        kernel_ft=kernel_ft[kernel_not_zero].reshape(1,-1)
        grid=h[kernel_not_zero]
        grid=grid[:,None]
        my_sum=fastsum(kernel_ft,x_proj,y_proj,x_weights=x_weights[:,None].tile(1,P))
    # Mean over all projections
    return torch.mean(my_sum-fastsum_energy,1)

def fast_Laplacian_summation_batched(x,y,x_weights,alpha,P,n_ft,sliced_factor,x_range=0.1,batch_size=1000):
    # batching over the projections
    P_left=P
    out=0
    while P_left>batch_size:
        out=out+fast_Laplacian_summation(x,y,x_weights,alpha,batch_size,n_ft,sliced_factor,x_range=0.1)/P*batch_size
        P_left-=batch_size
    out=out+fast_Laplacian_summation(x,y,x_weights,alpha,P_left,n_ft,sliced_factor,x_range=0.1)/P*P_left
    return out
    
def fast_Matern_summation(x,y,x_weights,alpha,P,n_ft,nu,x_range=0.1):
    # fastsum for the Matern kernel
    d=x.shape[1]

    # In 1D: sliced=unsliced => P=1 is sufficient
    if d==1:
        P=1

    # 1D Projections
    xi=torch.randn((P,d),dtype=dtype,device=device)
    xi=xi/torch.sqrt(torch.sum(xi**2,-1,keepdim=True))
    xi=xi.unsqueeze(1)
    x_proj=torch.nn.functional.conv1d(x.reshape(1,1,-1),xi,stride=d).reshape(P,-1).transpose(0,1)
    y_proj=torch.nn.functional.conv1d(y.reshape(1,1,-1),xi,stride=d).reshape(P,-1).transpose(0,1)

    # Rescaling
    xy_proj=torch.cat((x_proj,y_proj),0)
    x_max=torch.max(xy_proj)
    x_min=torch.min(xy_proj)
    x_mid=.5*(x_max+x_min)
    scale_factor=x_range/(x_max-x_min)
    x_proj=(x_proj-x_mid)*scale_factor
    y_proj=(y_proj-x_mid)*scale_factor
    alpha_real=alpha*scale_factor

    # compute kernel Fourier coefficients
    h = torch.arange((-n_ft+1)//2, (n_ft+1)//2, device=device)
    perm = h%n_ft
    perm_inv = torch.argsort(h%n_ft)
    vect=matern_kernel_fun(torch.abs(h)/n_ft,alpha_real.item(),nu,d)    
    vect_perm = vect[perm_inv] 
    kernel_ft= 1/n_ft * torch.fft.fft( vect_perm )[perm]
    kernel_ft=kernel_ft.reshape(1,-1)
    
    # fast Fourier summation
    if nfft_loaded:
        my_sum=fastsum_fft(kernel_ft,x_proj,y_proj,x_weights=x_weights[:,None].tile(1,P))
    else:
        my_sum=fastsum(kernel_ft,x_proj,y_proj,x_weights=x_weights[:,None].tile(1,P))

    # Mean over all projections
    return torch.mean(my_sum,1)

def fast_Matern_summation_batched(x,y,x_weights,alpha,P,n_ft,nu,x_range=0.1,batch_size=1000):
    # batching over the projections
    P_left=P
    out=0
    while P_left>batch_size:
        out=out+fast_Matern_summation(x,y,x_weights,alpha,batch_size,n_ft,nu,x_range=0.1)/P*batch_size
        P_left-=batch_size
    out=out+fast_Matern_summation(x,y,x_weights,alpha,P_left,n_ft,nu,x_range=0.1)/P*P_left
    return out
