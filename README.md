# Fast Kernel Summation in High Dimensions via Slicing and Fourier Transforms

This is the implementation to the paper "Fast Kernel Summation in High Dimensions via Slicing and Fourier Transforms" available at https://arxiv.org/abs/xxxx.xxxxx 

The code is written in PyTorch. For the NFFT we use the package torch-NFFT available at https://github.com/dominikbuenger/torch_nfft
Note that this package is GPU-only. For a CPU implementation we go back to the NDFT which might be significantly slower.

The file `fastsum.py` implements the general methods. The scripts `test_fastsum_xxx.py` implement the numerical examples from the paper. The files `test_RFF_vs_fastsum_xxx.py` implement the comparison to random Fourier features [2].

If you have any questions, feel free to contact me via the email address j.hertrich@ucl.ac.uk  


## REFERENCES

[1] J. Hertrich (2024).  
Fast Kernel Summation in High Dimensions via Slicing and Fourier Transforms.  
Preprint available under https://arxiv.org/abs/xxxx.xxxxx

[2] A. Rahimi and B. Recht (2007).   
Random features for large-scale kernel machines.   
Advances in Neural Information Processing Systems
