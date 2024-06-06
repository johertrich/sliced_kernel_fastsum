using SpecialFunctions

function Gaussian_kernel_fun_ft(grid1d,d,sigma_sq)
    k=grid1d
    args=2 .* pi^2 .* sigma_sq .* k.^2
    log_args=log.(args)
    log_args[args.==0].=0
    factor=d*pi*sqrt(.5*sigma_sq)
    log_out=log_args*.5*(d-1).-args.-loggamma(.5*(d+2))
    out=exp.(log_out)
    if d>1
        out[args.==0].=0
    else
        out[args.==0].=1/gamma(.5*(d+2))
    end
    return out*factor
end

function Laplace_kernel_fun_1d(x,alpha,d)
    out_all=zeros(size(x))
    relevant= (x*alpha.<=3.5) .&& (alpha*x .> 1e-5)
    x_relevant = x[relevant]*alpha
    out=zeros(size(x_relevant))
    diff=ones(size(x_relevant))
    n=0
    log_g_d2=loggamma(.5*d)
    log_x=log.(x_relevant)
    while findmax(abs.(diff))[1]>1e-5 && n<100
        log_factor=.5*log(pi)+loggamma(.5*(n+d))-loggamma(n+1)-log_g_d2-loggamma(.5*(n+1))
        addi=n*log_x.+log_factor
        if n%2==0
            diff=exp.(addi)
        else
            diff=-exp.(addi)
        end
        out=out+diff
        n=n+1
    end
    out_all[relevant]=out
    out_all[x*alpha.<=1e-5].=1
    return out_all
end


