# This code belongs to the paper
#
# J. Hertrich, 2024.
# Fast Kernel Summation in High Dimensions via Slicing and Fourier Transforms.
# arxiv preprint 2401.08260
#
# Please cite the paper, if you use this code.

import torch
from fastsum import *
import math
import numpy as np
import scipy, scipy.special,scipy.io
from kernels import *
import time

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
dtype=torch.float


def kernel_sum_comparison(N,d,P_list,runs=10,multiplicities=10):
    sliced_factor=compute_sliced_factor(d)
    errors_fastsum=[[] for P in P_list]
    times_naive=[]
    times_fastsum=[[] for P in P_list]
    for _ in range(runs):
        kernel_mat=lambda x,y: -distance(x,y)
        x=.1*torch.randn((N,d),dtype=dtype,device=device)
        y=.1*torch.randn((N,d),dtype=dtype,device=device)
        x_weights=torch.rand((N,),dtype=dtype,device=device)
        print('Naive')
        torch.cuda.synchronize()
        tic=time.time()
        for _ in range(multiplicities):
            out_naive=naive_kernel_sum_keops_energy(x,x_weights,y,batch_size=1000)
        torch.cuda.synchronize()
        toc=time.time()-tic
        print(torch.sum(out_naive))
        times_naive.append(toc)
        for P_num,P in enumerate(P_list):
            print(f'P={P}')
            torch.cuda.synchronize()
            tic=time.time()
            for _ in range(multiplicities):
                out_fastsum=fast_energy_summation_batched(x,y,x_weights,P,sliced_factor,batch_size=int(1e8)//N)
            torch.cuda.synchronize()
            toc=time.time()-tic
            print(torch.sum(out_fastsum))
            times_fastsum[P_num].append(toc)
            error=torch.sum(torch.abs(out_naive-out_fastsum)).item()
            rel_error=error/(torch.sum(torch.abs(x_weights))*y.shape[0])
            errors_fastsum[P_num].append(rel_error.item())
    time_naive_mean=np.mean(times_naive)
    time_naive_std=np.std(times_naive)
    time_fastsum_mean=[np.mean(times_fastsum[P_num]) for P_num in range(len(P_list))]
    time_fastsum_std=[np.std(times_fastsum[P_num]) for P_num in range(len(P_list))]
    error_fastsum_mean=[np.mean(errors_fastsum[P_num]) for P_num in range(len(P_list))]
    error_fastsum_std=[np.std(errors_fastsum[P_num]) for P_num in range(len(P_list))]
    return time_naive_mean,time_naive_std,time_fastsum_mean,time_fastsum_std,error_fastsum_mean,error_fastsum_std



d=100
Ns=[10000,100000,1000000,10000000]
Ps=[10,100,1000,2000,5000]
times_naive=[]
times_fastsum=[]
for N in Ns:
    out=kernel_sum_comparison(N,d,Ps,runs=1,multiplicities=1)
    times_fastsum.append(out[2])
    times_naive.append(out[0])
    
times_fastsum=np.array(times_fastsum)
times_naive=np.array(times_naive)

scipy.io.savemat('Energy_vs_keops.mat',{'times_naive':times_naive,'times_fastsum':times_fastsum,'Ns':Ns,'Ps':Ps,'d':d})

