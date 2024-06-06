using Distributions
using NFFT3
using FFTW

include("kernels.jl")

normal_distr=Normal()
uniform_distr=Uniform()

function ndft_adjoint(x,pos,grid)
    # adjoint NDFT
    fourier_matrix = exp.(2im * pi * grid * pos')
    return fourier_matrix*x
end

function fastsum_fft(kernel_ft,x,y,x_weights)
    # 1D fast Fourier summation with NFFT
    M=size(x_weights,1)
    N=size(kernel_ft)
    p=NFFT3.NFFT(N,M)
    p.x=-x
    p.f=x_weights
    NFFT3.nfft_adjoint(p)
    a=p.fhat
    d=kernel_ft.*a
    p.fhat=d
    p.x=-y
    NFFT3.trafo(p)
    out=real.(p.f)
    return out    
end

function ndft_forward(x,pos,grid)
    # forward NDFT
    fourier_matrix = exp.(-2im * pi * pos * grid')
    return fourier_matrix*x
end

function fastsum(kernel_ft,x,y,x_weights,grid)
    # 1D fast Fourier summation with the NDFT
    out = zeros(size(y))
    a=ndft_adjoint(x_weights,-x,grid)
    d=kernel_ft.*a
    summand=ndft_forward(d,-y,grid)
    out=real.(summand)
    return out
end

function naive_kernel_sum(x,x_weights,y,kernel)
    # Naive kernel summation via the kernel matrix
    d=size(x,2)
    N=size(x,1)
    M=size(y,1)
    naive_sum=zeros(M)
    # batching for fitting memory constraints
    for i in 1:N
        for j in 1:M
            naive_sum[j]+=x_weights[i]*kernel(x[i,:],y[j,:])
        end
    end
    return naive_sum
end



function fastsum_gauss(x,y,x_weights,sigma_sq,P,n_ft,x_range)
    # slicing for the Gaussian kernel
    d=size(x,2)    
    if size(x,2)==1
        P=1
    end
    out=zeros(size(y,1))
    for p in 1:P
        xi = rand(normal_distr,d)
        xi = xi ./ sqrt.(sum(xi.^2))
        x_proj=x*xi
        y_proj=y*xi
        
        x_max=findmax([x_proj y_proj])[1]
        x_min=findmin([x_proj y_proj])[1]
        x_mid=.5*(x_max+x_min)
        scale_factor=x_range/(x_max-x_min)
        sigma_max=0.025
        if d==2
            sigma_max=1e-5
        end
        if sigma_sq*scale_factor^2>sigma_max
            scale_factor=sqrt(sigma_max/sigma_sq)
        end
        x_proj=(x_proj.-x_mid).*scale_factor
        y_proj=(y_proj.-x_mid).*scale_factor
        sigma_real=sigma_sq*scale_factor^2
        
        
        grid1d=floor(.5*(-n_ft+1)):1.:0
        kernel_ft=Gaussian_kernel_fun_ft(grid1d,d,sigma_real)
        kernel_not_zero=kernel_ft.>1e-4
        kernel_ft=kernel_ft[kernel_not_zero]
        grid=grid1d[kernel_not_zero]
        # use symmetry
        append!(grid,-grid)
        append!(kernel_ft,kernel_ft)
        
        res = fastsum(kernel_ft,x_proj,y_proj,x_weights,grid)
        out=out+res
    end
    return out./P
end

   
function RFF_Gauss(x,y,x_weights,sigma_sq,P)
    # RFF for the Gaussian kernel
    d=size(x,2)
    out=zeros(size(y,1))
    for p in 1:P
        xi = rand(normal_distr,d)
        x_proj=x*xi./sqrt(sigma_sq)
        y_proj=y*xi./sqrt(sigma_sq)
        b=2*pi*rand(uniform_distr,1)
        zeta_x=sqrt(2)*cos.(x_proj.+b)
        zeta_y=sqrt(2)*cos.(y_proj.+b)
        zeta_x_sum=sum(zeta_x.*x_weights)
        res=zeta_y.*zeta_x_sum
        out=out+res
    end
    return out./P
end

function RFFM_Gauss(x,y,x_weights,sigma_sq,P)
    # RFF for the Gaussian kernel with different embedding
    d=size(x,2)
    out=zeros(size(y,1))
    for p in 1:P
        xi = rand(normal_distr,d)
        x_proj=x*xi./sqrt(sigma_sq)
        y_proj=y*xi./sqrt(sigma_sq)
        cos_x=cos.(x_proj)
        cos_y=cos.(y_proj)
        cos_x_sum=sum(cos_x.*x_weights)
        sin_x=sin.(x_proj)
        sin_y=sin.(y_proj)
        sin_x_sum=sum(sin_x.*x_weights)
        res=cos_y.*cos_x_sum+sin_y.*sin_x_sum
        out=out+res
    end
    return out./P
end



function fastsum_energy_1d(x,x_weights,y)
    # Sorting algorithm for fast sumation with negative distance (energy) kernel
    N=size(x,1)
    M=size(y,1)
    # Potential Energy
    yx=[y; x]
    inds_yx=sortperm(yx,rev=false)
    sorted_yx=yx[inds_yx]
    yx_weights=[zeros(M);x_weights]
    weights_sorted=yx_weights[inds_yx]
    pot0=sum(weights_sorted.*(sorted_yx.-sorted_yx[1]))
    yx_diffs=sorted_yx[2:M+N]-sorted_yx[1:M+N-1]
    # Mults from cumsums shifted by 1
    mults_short=sum(x_weights).-2*cumsum(weights_sorted)[1:M+N-1]
    mults=zeros(M+N)
    mults[2:N+M]=mults_short
    
    potential=zeros(M+N)
    potential[2:M+N]=yx_diffs
    potential=pot0.-cumsum(potential.*mults)
    out1=zeros(M+N)
    out1[inds_yx]=potential
    out1=out1[1:M]
    return out1
end




function fastsum_energy(x,y,x_weights,P,sliced_factor)
    # slicing for energy kernel
    d=size(x,2)
    if d==1
        P=1
    end
    out=zeros(size(y,1))
    for p in 1:P
    
        xi = rand(normal_distr,d)
        xi = xi ./ sqrt.(sum(xi.^2))
        x_proj=x*xi
        y_proj=y*xi
        fastsum_energy=fastsum_energy_1d(deepcopy(x_proj),deepcopy(x_weights),deepcopy(y_proj))
        out=out+fastsum_energy
    end
    return -sliced_factor/P * out
end

function compute_sliced_factor(d)
    # Compute the slicing constant within the negative distance kernel
    k=(d-1)//2
    fac=1.
    if (d-1)%2==0
        for j in 1:k
            fac=2*fac*j/(2*j-1)
        end
    else
        for j in 1:k
            fac=fac*(2*j+1)/(2*j)
        end
        fac=fac*pi/2
    end
    return fac
end


function fastsum_laplace(x,y,x_weights,alpha,P,sliced_factor,n_ft,x_range)
    # slicing for the Laplacian kernel
    d=size(x,2)
    if d==1
        P=1
    end    
    out=zeros(size(y,1))
    for p in 1:P
        xi = rand(normal_distr,d)
        xi = xi ./ sqrt.(sum(xi.^2))
        x_proj=x*xi
        y_proj=y*xi
        
        x_max=findmax([x_proj y_proj])[1]
        x_min=findmin([x_proj y_proj])[1]
        x_mid=.5*(x_max+x_min)
        scale_factor=x_range/(x_max-x_min)
        x_proj=(x_proj.-x_mid).*scale_factor
        y_proj=(y_proj.-x_mid).*scale_factor
        alpha_real=alpha/scale_factor
        
        fastsum_energy=fastsum_energy_1d(x_proj,x_weights,y_proj)
        fastsum_energy=alpha_real*fastsum_energy*sliced_factor
        
        h=collect(floor(.5*(-n_ft+1)):1:floor(.5*(n_ft-1)))
        vect=Laplace_kernel_fun_1d(abs.(h/n_ft),alpha_real,d)+alpha_real*abs.(h/n_ft)*sliced_factor
        
        vect_perm=ifftshift(vect)
        kernel_ft=1/n_ft * fftshift(fft(vect_perm))
        mysum=fastsum_fft(kernel_ft,x_proj,y_proj,x_weights)
        out=out+mysum-fastsum_energy
    end
    return out/P
end

function RFF_Laplace(x,y,x_weights,alpha,P)
    # RFF for the Laplacian kernel
    d=size(x,2)
    out=zeros(size(y,1))
    for p in 1:P
        chi=rand(Chisq(1))
        xi = rand(normal_distr,d)/sqrt(chi)
        x_proj=x*xi.*alpha
        y_proj=y*xi.*alpha
        b=2*pi*rand(uniform_distr,1)
        zeta_x=sqrt(2)*cos.(x_proj.+b)
        zeta_y=sqrt(2)*cos.(y_proj.+b)
        zeta_x_sum=sum(zeta_x.*x_weights)
        res=zeta_y.*zeta_x_sum
        out=out+res
    end
    return out./P
end

function RFFM_Laplace(x,y,x_weights,alpha,P)
    # RFF for the Laplacian kernel with different embedding
    d=size(x,2)
    out=zeros(size(y,1))
    for p in 1:P
        chi=rand(Chisq(1))
        xi = rand(normal_distr,d)/sqrt(chi)
        x_proj=x*xi.*alpha
        y_proj=y*xi.*alpha
        cos_x=cos.(x_proj)
        cos_y=cos.(y_proj)
        cos_x_sum=sum(cos_x.*x_weights)
        sin_x=sin.(x_proj)
        sin_y=sin.(y_proj)
        sin_x_sum=sum(sin_x.*x_weights)
        res=cos_y.*cos_x_sum+sin_y.*sin_x_sum
        out=out+res
    end
    return out./P
end
