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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        tic=time.time()
        for _ in range(multiplicities):
            out_naive=naive_kernel_sum(x,x_weights,y,kernel_mat,batch_size=1000000//N)
        toc=time.time()-tic
        times_naive.append(toc)
        for P_num,P in enumerate(P_list):
            print(f'P={P}')
            tic=time.time()
            for _ in range(multiplicities):
                out_fastsum=fast_energy_summation_batched(x,y,x_weights,P,sliced_factor,batch_size=1000)
            toc=time.time()-tic
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


#########################################################################################################
#################### CREATE TABLES AND PLOT DATA ########################################################
#########################################################################################################

tab1=True
tab2=True
tab3=True

if tab1:
    # Table runtime and error vs N    
    Ns=list(range(2000,52000,2000))
    d=50
    alpha=1.
    Ps=[10,100,1000,2000,5000]
    results=[]
    for N in Ns:
        print(f'Run with N={N}')
        results.append(kernel_sum_comparison(N,d,Ps,runs=10,multiplicities=1))
    print(results)

    # Build table
    runtimes=[]
    errors=[]
    lines=[]
    lines_err=[]
    line=''
    for N in Ns:
        line=line+f'&N=${N}$'
    lines.append(line+'\\\\\n')
    lines_err.append(line+'\\\\\n')
    rt=[]
    line='Naive'
    for i in range(len(Ns)):
        line=line+'&${0:1.4f}\pm{1:1.4f}$'.format(results[i][0],results[i][1])
        rt.append(results[i][0])
    runtimes.append(rt)
    lines.append(line+'\\\\\n')
    for j in range(1,len(Ps)):
        rt=[]
        er=[]
        line=f'P={Ps[j]}'
        line_err=f'P={Ps[j]}'
        for i in range(len(Ns)):
            line=line+'&${0:1.4f}\pm{1:1.4f}$'.format(results[i][2][j],results[i][3][j])
            line_err=line_err+'&${0:1.4f}\pm{1:1.4f}$'.format(results[i][4][j],results[i][5][j])
            rt.append(results[i][2][j])
            er.append(results[i][4][j])
        runtimes.append(rt)
        errors.append(er)
        lines.append(line+'\\\\\n')
        lines_err.append(line_err+'\\\\\n')
    runtimes=np.array(runtimes)
    errors=np.array(errors)
    print(runtimes)
    with open('Energy_table_runtimes','w') as f:
        f.writelines(lines)
    with open('Energy_table_errors','w') as f:
        f.writelines(lines_err)
    scipy.io.savemat('Energy_results.mat',{'runtimes':runtimes,'errors':errors,'Ns':Ns,'Ps':Ps,'d':d})
    
if tab2:
    N=10000
    ds=[1]+list(range(20,220,20))
    alpha=1.
    Ps=[10,100,1000,2000,5000]
    results=[]
    for d in ds:
        print(f'Run with d={d}')
        results.append(kernel_sum_comparison(N,d,Ps,runs=10,multiplicities=1))
    print(results)

    # Build table
    runtimes=[]
    errors=[]
    lines=[]
    lines_err=[]
    line=''
    for d in ds:
        line=line+f'&d=${d}$'
    lines.append(line+'\\\\\n')
    lines_err.append(line+'\\\\\n')
    rt=[]
    line='Naive'
    for i in range(len(ds)):
        line=line+'&${0:1.4f}\pm{1:1.4f}$'.format(results[i][0],results[i][1])
        rt.append(results[i][0])
    runtimes.append(rt)
    lines.append(line+'\\\\\n')
    for j in range(1,len(Ps)):
        rt=[]
        er=[]
        line=f'P={Ps[j]}'
        line_err=f'P={Ps[j]}'
        for i in range(len(ds)):
            line=line+'&${0:1.4f}\pm{1:1.4f}$'.format(results[i][2][j],results[i][3][j])
            line_err=line_err+'&${0:1.4f}\pm{1:1.4f}$'.format(results[i][4][j],results[i][5][j])
            rt.append(results[i][2][j])
            er.append(results[i][4][j])
        runtimes.append(rt)
        errors.append(er)
        lines.append(line+'\\\\\n')
        lines_err.append(line_err+'\\\\\n')
    runtimes=np.array(runtimes)
    errors=np.array(errors)
    print(runtimes)
    with open('Energy_table_runtimes_base_d','w') as f:
        f.writelines(lines)
    with open('Energy_table_errors_base_d','w') as f:
        f.writelines(lines_err)
    scipy.io.savemat('Energy_results_d.mat',{'runtimes':runtimes,'errors':errors,'N':N,'Ps':Ps,'ds':ds})

if tab3:
    N=10000
    ds=[10,50,100,200]
    alpha=1.
    Ps=[10,50,100,200,500,1000,2000,5000,10000]
    results=[]
    for d in ds:
        print(f'Run with d={d}')
        results.append(kernel_sum_comparison(N,d,Ps,runs=10,multiplicities=1))
    print(results)

    # Build table
    runtimes=[]
    errors=[]
    lines=[]
    lines_err=[]
    line=''
    for d in ds:
        line=line+f'&d=${d}$'
    lines.append(line+'\\\\\n')
    lines_err.append(line+'\\\\\n')
    rt=[]
    line='Naive'
    for i in range(len(ds)):
        line=line+'&${0:1.4f}\pm{1:1.4f}$'.format(results[i][0],results[i][1])
        rt.append(results[i][0])
    runtimes.append(rt)
    lines.append(line+'\\\\\n')
    for j in range(1,len(Ps)):
        rt=[]
        er=[]
        line=f'P={Ps[j]}'
        line_err=f'P={Ps[j]}'
        for i in range(len(ds)):
            line=line+'&${0:1.4f}\pm{1:1.4f}$'.format(results[i][2][j],results[i][3][j])
            line_err=line_err+'&${0:1.4f}\pm{1:1.4f}$'.format(results[i][4][j],results[i][5][j])
            rt.append(results[i][2][j])
            er.append(results[i][4][j])
        runtimes.append(rt)
        errors.append(er)
        lines.append(line+'\\\\\n')
        lines_err.append(line_err+'\\\\\n')
    runtimes=np.array(runtimes)
    errors=np.array(errors)
    print(runtimes)
    with open('Energy_table_runtimes_base_P','w') as f:
        f.writelines(lines)
    with open('Energy_table_errors_base_P','w') as f:
        f.writelines(lines_err)
    scipy.io.savemat('Energy_results_P.mat',{'runtimes':runtimes,'errors':errors,'N':N,'Ps':Ps,'ds':ds})
