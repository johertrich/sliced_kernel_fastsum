# Fast Kernel Summation in High Dimensions via Slicing and Fourier Transforms

This repository reproduces the numerical examples from the paper "Fast Kernel Summation in High Dimensions via Slicing and Fourier Transforms" available at https://doi.org/10.1137/24M1632085 (a preprint is available at https://arxiv.org/abs/2401.08260 ).

**Note**: **The purpose of this repository is to reproduce the results from [this paper](https://doi.org/10.1137/24M1632085). A more up-to-date implementation of the fast kernel summation is available at [https://github.com/johertrich/simple_torch_NFFT](https://github.com/johertrich/simple_torch_NFFT).**

-------------------------------------------------------------------------------------

The code contains one implementation in PyTorch and one in Julia. In PyTorch, we use the package torch-NFFT available at https://github.com/dominikbuenger/torch_nfft for the NFFT.
Note that this package is GPU-only. For a CPU implementation we go back to the NDFT which might be significantly slower.
For Julia, we use the NFFT package [2].

The Python-scripts `test_fastsum_xxx.py` implement the numerical examples from the paper and the files `test_xxx_vs_keops.py` implment the comparison to PyKeOps [3]. The Julia-files `fastsum_vs_RFF_xxx.jl` implement the comparison to random Fourier features [1].

If you have any questions, feel free to contact me via the email address j.hertrich@ucl.ac.uk  

[1] A. Rahimi and B. Recht (2007).   
Random features for large-scale kernel machines.   
Advances in Neural Information Processing Systems

[2] J. Keiner, S. Kunis, and D. Potts (2009).  
Using NFFT3 - a software library for various nonequispaced fast Fourier transforms.  
ACM Transactions on Mathematical Software, 36, no. 19

[3] B. Charlier, J. Feydy, J. A. Glaunes, F.-D. Collin, and G. Durif (2021).  
Kernel operations on the GPU with autodiff, without memory overflows.  
Journal of Machine Learning Research

## Citation

```
@article{H2024,
  title={Fast Kernel Summation in High Dimensions via Slicing and {F}ourier transforms},
  author={Hertrich, Johannes},
  journal={SIAM Journal on Mathematics of Data Science},
  volume={6},
  number={4},
  pages={1109--1137},
  year={2024}
}
```



