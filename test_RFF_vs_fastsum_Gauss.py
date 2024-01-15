# This code belongs to the paper
#
# J. Hertrich, 2024.
# Fast Kernel Summation in High Dimensions via Slicing and Fourier Transforms.
# arxiv ...
#
# Please cite the paper, if you use this code.

import torch
from fastsum import *
import math
import numpy as np
import scipy.io
from kernels import *
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype=torch.float

def RFF_Gauss(x,y,x_weights,sigma_sq,P):
    d=x.shape[1]
    xi=torch.randn((P,d),dtype=dtype,device=device)
    xi=xi.unsqueeze(1)
    b=2*math.pi*torch.rand((P,),dtype=dtype,device=device)
    x_proj=torch.nn.functional.conv1d(x.reshape(1,1,-1),xi,stride=d).reshape(P,-1).transpose(0,1)/np.sqrt(sigma_sq)
    y_proj=torch.nn.functional.conv1d(y.reshape(1,1,-1),xi,stride=d).reshape(P,-1).transpose(0,1)/np.sqrt(sigma_sq)
    zeta_x=np.sqrt(2)*torch.cos(x_proj+b)
    zeta_y=np.sqrt(2)*torch.cos(y_proj+b)
    zeta_x_sum=torch.sum(zeta_x*x_weights[:,None],0)
    res=torch.mean(zeta_y*zeta_x_sum[None,:],-1)
    return res
    
def RFF_Gauss_batched(x,y,x_weights,sigma_sq,P,batch_size):
    # batching over the projections
    P_left=P
    out=0
    while P_left>batch_size:
        out=out+RFF_Gauss(x,y,x_weights,sigma_sq,batch_size)/P*batch_size
        P_left-=batch_size
    out=out+RFF_Gauss(x,y,x_weights,sigma_sq,P_left)/P*P_left
    return out
    

def kernel_sum_comparison(N,d,sigma_sq,P_list,runs=10,multiplicities=10,n_ft=200):
    errors_fastsum=[[] for P in P_list]
    errors_RFF=[[] for P in P_list]
    times_naive=[]
    times_fastsum=[[] for P in P_list]
    times_RFF=[[] for P in P_list]
    for _ in range(runs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        kernel_mat=lambda x,y: Gaussian_kernel_mat(x,y,sigma_sq)
        x=.1*torch.randn((N,d),dtype=dtype,device=device)
        y=.1*torch.randn((N,d),dtype=dtype,device=device)
        x_weights=torch.rand((N,),dtype=dtype,device=device)
        x_weights=torch.ones_like(x_weights)
        print('Naive')
        tic=time.time()
        for _ in range(1):
            out_naive=naive_kernel_sum(x,x_weights,y,kernel_mat,batch_size=1)
        toc=time.time()-tic
        times_naive.append(toc)
        for P_num,P in enumerate(P_list):
            print(f'P={P}')
            tic=time.time()
            for _ in range(multiplicities):
                out_fastsum=fast_Gaussian_summation_batched(x,y,x_weights,sigma_sq,P,n_ft,x_range=0.3,batch_size=100)
            toc=time.time()-tic
            times_fastsum[P_num].append(toc)
            error=torch.sum(torch.abs(out_naive-out_fastsum)).item()
            rel_error=error/(torch.sum(torch.abs(x_weights))*y.shape[0])
            errors_fastsum[P_num].append(rel_error.item())
            tic=time.time()
            for _ in range(multiplicities):
                out_RFF=RFF_Gauss_batched(x,y,x_weights,sigma_sq,10*P,batch_size=100)
            toc=time.time()-tic
            times_RFF[P_num].append(toc)
            error=torch.sum(torch.abs(out_naive-out_RFF)).item()
            rel_error=error/(torch.sum(torch.abs(x_weights))*y.shape[0])
            errors_RFF[P_num].append(rel_error.item())
    time_naive_mean=np.mean(times_naive)
    time_naive_std=np.std(times_naive)
    time_fastsum_mean=[np.mean(times_fastsum[P_num]) for P_num in range(len(P_list))]
    time_fastsum_std=[np.std(times_fastsum[P_num]) for P_num in range(len(P_list))]
    time_RFF_mean=[np.mean(times_RFF[P_num]) for P_num in range(len(P_list))]
    time_RFF_std=[np.std(times_RFF[P_num]) for P_num in range(len(P_list))]
    error_fastsum_mean=[np.mean(errors_fastsum[P_num]) for P_num in range(len(P_list))]
    error_fastsum_std=[np.std(errors_fastsum[P_num]) for P_num in range(len(P_list))]
    error_RFF_mean=[np.mean(errors_RFF[P_num]) for P_num in range(len(P_list))]
    error_RFF_std=[np.std(errors_RFF[P_num]) for P_num in range(len(P_list))]
    return time_naive_mean,time_naive_std,time_fastsum_mean,time_fastsum_std,error_fastsum_mean,error_fastsum_std,time_RFF_mean,time_RFF_std,error_RFF_mean,error_RFF_std



ds=[100,1000,10000]
for d in ds:
    res=kernel_sum_comparison(20000,d,1.,[10,10,50,100,200,500,1000,2000,5000,10000],runs=10,multiplicities=1,n_ft=1024)
    time_naive_mean,time_naive_std,time_fastsum_mean,time_fastsum_std,error_fastsum_mean,error_fastsum_std,time_RFF_mean,time_RFF_std,error_RFF_mean,error_RFF_std=res
    print(error_RFF_mean,error_fastsum_mean)
    print(time_RFF_mean,time_fastsum_mean)
    print(time_naive_mean)

    scipy.io.savemat(f'RFF_vs_fastsum_Gauss_{d}.mat',{'runtimes_RFF':np.array(time_RFF_mean)[1:],'errors_RFF':np.array(error_RFF_mean)[1:],'runtimes_fastsum':np.array(time_fastsum_mean)[1:],'errors_fastsum':np.array(error_fastsum_mean)[1:]})


