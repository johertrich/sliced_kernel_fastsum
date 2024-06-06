# Fast Kernel Summation in High Dimensions via Slicing and Fourier Transforms

This is the implementation to the paper "Fast Kernel Summation in High Dimensions via Slicing and Fourier Transforms" available at https://arxiv.org/abs/2401.08260

The code contains one implementation in PyTorch and one in Julia. In PyTorch, we use the package torch-NFFT available at https://github.com/dominikbuenger/torch_nfft for the NFFT.
Note that this package is GPU-only. For a CPU implementation we go back to the NDFT which might be significantly slower.
For Julia, we use the NFFT package [3].

The Python-scripts `test_fastsum_xxx.py` implement the numerical examples from the paper and the files `test_xxx_vs_keops.py` implment the comparison to PyKeOps [4]. The Julia-files `fastsum_vs_RFF_xxx.jl` implement the comparison to random Fourier features [2].

If you have any questions, feel free to contact me via the email address j.hertrich@ucl.ac.uk  


## REFERENCES

[1] J. Hertrich (2024).  
Fast Kernel Summation in High Dimensions via Slicing and Fourier Transforms.  
Preprint available under https://arxiv.org/abs/2401.08260

[2] A. Rahimi and B. Recht (2007).   
Random features for large-scale kernel machines.   
Advances in Neural Information Processing Systems

[3] J. Keiner, S. Kunis, and D. Potts (2009).  
Using NFFT3 - a software library for various nonequispaced fast Fourier transforms.  
ACM Transactions on Mathematical Software, 36, no. 19

[4] B. Charlier, J. Feydy, J. A. Glaunes, F.-D. Collin, and G. Durif (2021).  
Kernel operations on the GPU with autodiff, without memory overflows.  
Journal of Machine Learning Research
