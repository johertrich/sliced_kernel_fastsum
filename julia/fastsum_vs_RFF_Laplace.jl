# Comparsion of the fast kernel summation with RFF for the Laplacian kernel

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
    alpha=0.05
elseif parameter_number==1
    alpha=0.25
elseif parameter_number==2
    alpha=0.5
end


normal_distr=Normal()

N=100
d=10
P=10
#400 for alpha=.5 and d=10000 otherwise 100
if experiment_number==8
    n_ft=400
else
    n_ft=100 
end
sliced_factor_=compute_sliced_factor(d)

# small run to compile functions
x_=.1 .* rand(normal_distr,N,d)
y_=.1 .* rand(normal_distr,N,d)
x_weights_=ones(N)

out_fastsum_= @time fastsum_laplace(x_,y_,x_weights_,alpha,P,sliced_factor_,n_ft,0.2)
out_RFF_= @time RFF_Laplace(x_,y_,x_weights_,alpha,P)
out_RFFM_= @time RFFM_Laplace(x_,y_,x_weights_,alpha,P)

# actual comparison
N=100000
Ps=[200 500 1000 2000 5000 10000]

path="."

for d in ds
    sliced_factor=compute_sliced_factor(d)
    fid = h5open(path*"samples_"*string(N)*"_"*string(d)*".h5","r")
    x=reshape(collect(fid["x"]),d,N)'
    y=reshape(collect(fid["y"]),d,N)'
    x_weights=collect(fid["x_weights"])
    close(fid)

    fid = h5open(path*"kernelsum_Laplace_"*string(N)*"_"*string(d)*"_"*string(alpha)*".h5","r")
    out_naive=collect(fid["out_lap"])
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
            out_fastsum=fastsum_laplace(x,y,x_weights,alpha,P,sliced_factor,n_ft,0.2)
            toc=time()-tic
            push!(times_fs,toc)
            error=sum(abs.(out_naive-out_fastsum))/(sum(x_weights)*size(y,1))
            push!(errors_fs,error)
            tic=time()
            out_RFF=RFF_Laplace(x,y,x_weights,alpha,2*P)
            toc=time()-tic
            push!(times_rff,toc)
            error=sum(abs.(out_naive-out_RFF))/(sum(x_weights)*size(y,1))
            push!(errors_rff,error)
            tic=time()
            out_RFFM=RFFM_Laplace(x,y,x_weights,alpha,2*P)
            toc=time()-tic
            push!(times_rffm,toc)
            error=sum(abs.(out_naive-out_RFFM))/(sum(x_weights)*size(y,1))
            push!(errors_rffm,error)
        end
        fid = h5open(path*"results_lap_"*string(N)*"_"*string(d)*"_"*string(alpha)*"_arr.h5","w")
        fid["times_fs"]=reshape(copy(times_fs),runs,i)
        fid["errors_fs"]=reshape(copy(errors_fs),runs,i)
        fid["times_rff"]=reshape(copy(times_rff),runs,i)
        fid["errors_rff"]=reshape(copy(errors_rff),runs,i)
        fid["times_rffm"]=reshape(copy(times_rffm),runs,i)
        fid["errors_rffm"]=reshape(copy(errors_rffm),runs,i)
        close(fid)
        println("P=",P)
        println("Fastsum time: ",mean(reshape(copy(times_fs),runs,i),dims=1)[i],"+-",std(reshape(copy(times_fs),runs,i),dims=1)[i])
        println("RFF time: ",mean(reshape(copy(times_rff),runs,i),dims=1)[i],"+-",std(reshape(copy(times_rff),runs,i),dims=1)[i])
        println("RFFM time: ",mean(reshape(copy(times_rffm),runs,i),dims=1)[i],"+-",std(reshape(copy(times_rffm),runs,i),dims=1)[i])
        println("Fastsum error: ",mean(reshape(copy(errors_fs),runs,i),dims=1)[i],"+-",std(reshape(copy(errors_fs),runs,i),dims=1)[i])
        println("RFF error: ",mean(reshape(copy(errors_rff),runs,i),dims=1)[i],"+-",std(reshape(copy(errors_rff),runs,i),dims=1)[i])
        println("RFFM error: ",mean(reshape(copy(errors_rffm),runs,i),dims=1)[i],"+-",std(reshape(copy(errors_rffm),runs,i),dims=1)[i])
    end
end
