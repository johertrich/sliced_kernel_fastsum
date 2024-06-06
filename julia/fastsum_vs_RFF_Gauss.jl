# Comparsion of the fast kernel summation with RFF for the Gaussian kernel

using Distributions
using LinearAlgebra
using HDF5
# compare single threaded
BLAS.set_num_threads(1)

include("fastsum.jl")
include("kernels.jl")

experiment_number=parse(Int64,ARGS[1])
dimension_number=experiment_number%3
parameter_number=div(experiment_number,3)
if dimension_number==0
    ds=[100]
    runs=10
elseif dimension_number==1
    ds=[1000]
    runs=10
elseif dimension_number==2
    ds=[10000]
    runs=1
end
if parameter_number==0
    sigma_sq=1.
elseif parameter_number==1
    sigma_sq=5.
elseif parameter_number==2
    sigma_sq=10.
end


normal_distr=Normal()

# small run to compile functions
N=100
d=10
P=10
n_ft=1024

x_=.1 .* rand(normal_distr,N,d)
y_=.1 .* rand(normal_distr,N,d)
x_weights_=ones(N)

out_fastsum_= @time fastsum_gauss(x_,y_,x_weights_,sigma_sq,P,n_ft,0.1)
out_RFF_= @time RFF_Gauss(x_,y_,x_weights_,sigma_sq,P)
out_RFFM_= @time RFFM_Gauss(x_,y_,x_weights_,sigma_sq,P)


# actual comparison
N=100000
Ps=[200 500 1000 2000 5000 10000]

path="."

for d in ds

    fid = h5open(path*"samples_"*string(N)*"_"*string(d)*".h5","r")
    x=reshape(collect(fid["x"]),d,N)'
    y=reshape(collect(fid["y"]),d,N)'
    x_weights=collect(fid["x_weights"])
    close(fid)

    fid = h5open(path*"kernelsum_Gauss_"*string(N)*"_"*string(d)*"_"*string(sigma_sq)*".h5","r")
    out_naive=collect(fid["out_gauss"])
    close(fid)

    i=0

    times_fs=Float64[]
    times_rff=Float64[]
    times_rffm=Float64[]
    errors_fs=Float64[]
    errors_rff=Float64[]
    errors_rffm=Float64[]
    for P in Ps
        i=i+1
        for run in 1:runs
            tic=time()
            out_fastsum=fastsum_gauss(x,y,x_weights,sigma_sq,P,n_ft,0.1)
            toc=time()-tic
            push!(times_fs,toc)
            error=sum(abs.(out_naive-out_fastsum))/(sum(x_weights)*size(y,1))
            push!(errors_fs,error)
            tic=time()
            out_RFF=RFF_Gauss(x,y,x_weights,sigma_sq,P)
            toc=time()-tic
            push!(times_rff,toc)
            error=sum(abs.(out_naive-out_RFF))/(sum(x_weights)*size(y,1))
            push!(errors_rff,error)
            tic=time()
            out_RFFM=RFFM_Gauss(x,y,x_weights,sigma_sq,P)
            toc=time()-tic
            push!(times_rffm,toc)
            error=sum(abs.(out_naive-out_RFFM))/(sum(x_weights)*size(y,1))
            push!(errors_rffm,error)
        end
        fid = h5open(path*"results_Gauss_"*string(N)*"_"*string(d)*"_"*string(sigma_sq)*"_arr.h5","w")
        fid["times_fs"]=reshape(copy(times_fs),runs,i)
        fid["errors_fs"]=reshape(copy(errors_fs),runs,i)
        fid["times_rff"]=reshape(copy(times_rff),runs,i)
        fid["errors_rff"]=reshape(copy(errors_rff),runs,i)
        fid["times_rffm"]=reshape(copy(times_rffm),runs,i)
        fid["errors_rffm"]=reshape(copy(errors_rffm),runs,i)
        close(fid)
    end
end
