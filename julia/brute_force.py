# create brute-force ground truth with pykeops for time reasons
import torch
import pykeops.torch
import numpy as np
import h5py
import os
import pykeops

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype=torch.float

def naive_kernel_sum_keops_Gauss(x,x_weights,y,sigma_sq,batch_size=100):
    # Naive kernel summation via the kernel matrix
    # with pykeops
    d=x.shape[1]
    N=x.shape[0]
    naive_sum=0
    i=0
    y=pykeops.torch.LazyTensor(y[None,:,:])
    # batching for fitting memory constraints
    while i<N:
        x_batch=pykeops.torch.LazyTensor(x[i:i+batch_size,None,:])
        x_weights_batch=pykeops.torch.LazyTensor(x_weights[i:i+batch_size,None],axis=0)
        kernel_mat=(-.5*((x_batch-y)**2).sum(-1)/sigma_sq).exp()
        naive_sum+=(kernel_mat*x_weights_batch).sum(0)
        i+=batch_size
    return naive_sum.squeeze()

def naive_kernel_sum_keops_Lap(x,x_weights,y,alpha,batch_size=100):
    # Naive kernel summation via the kernel matrix
    # with pykeops
    d=x.shape[1]
    N=x.shape[0]
    naive_sum=0
    i=0
    y=pykeops.torch.LazyTensor(y[None,:,:])
    # batching for fitting memory constraints
    while i<N:
        x_batch=pykeops.torch.LazyTensor(x[i:i+batch_size,None,:])
        x_weights_batch=pykeops.torch.LazyTensor(x_weights[i:i+batch_size,None],axis=0)
        kernel_mat=(-alpha*((x_batch-y)**2).sum(-1).sqrt()).exp()
        naive_sum+=(kernel_mat*x_weights_batch).sum(0)
        i+=batch_size
    return naive_sum.squeeze()


ds=[100,1000,10000]
N=100000
sigma_sq=100.
alpha=.5
path="."


for d in ds:
    name=f"samples_{N}_{d}.h5"
    name_gauss=f"kernelsum_Gauss_{N}_{d}_{sigma_sq}.h5"
    name_lap=f"kernelsum_Laplace_{N}_{d}_{alpha}.h5"
    if os.path.isfile(path+name):
        with h5py.File(path+name,"r") as f:
            x=torch.tensor(np.array(f['x']),dtype=dtype,device=device)
            y=torch.tensor(np.array(f['y']),dtype=dtype,device=device)
            x_weights=torch.tensor(np.array(f['x_weights']),dtype=dtype,device=device)
    else:
        x=.1*torch.randn((N,d),dtype=dtype,device=device)
        y=.1*torch.randn((N,d),dtype=dtype,device=device)
        x_weights=torch.rand((N,),dtype=dtype,device=device)
        with h5py.File(path+name,"w") as f:
            f.create_dataset("x",data=x.detach().cpu().numpy())
            f.create_dataset("y",data=y.detach().cpu().numpy())
            f.create_dataset("x_weights",data=x_weights.detach().cpu().numpy())
    print(x.shape,y.shape,x_weights.shape)
    out_gauss=naive_kernel_sum_keops_Gauss(x,x_weights,y,sigma_sq,batch_size=10000)
    out_lap=naive_kernel_sum_keops_Lap(x,x_weights,y,alpha,batch_size=10000)
    with h5py.File(path+name_gauss,"w") as f:
        f.create_dataset("out_gauss",data=out_gauss.detach().cpu().numpy())
    with h5py.File(path+name_lap,"w") as f:
        f.create_dataset("out_lap",data=out_lap.detach().cpu().numpy())

