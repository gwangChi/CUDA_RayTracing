# CUDA Version - GPU multithreaded Ray Tracing
After compilation, you need to pass two parameters to the kernel program. The first one is the number of rays, and the second one is the dimension of the 2d image pixels that you want to get. For example, ```./kernel 1000000000 1000``` lets the program compute 1 billion rays using the GPU you have and then writes the output as a 1000*1000 matrix in the 1000000000.dat file.

To adjust the number of threads per block and number of blocks, you can locate to line 60 and line 61 of the program (kernel.cu). Note that do not exceed the maximum values allowed for your GPU.

# Results
Running 1 billion rays on the TESLA V100 PCIE gpu on Midway3 takes 0.805091 seconds with 39063 blocks and 256 threads per block.
Running 10 billion rays on the TESLA V100 PCIE gpu on Midway3 takes 7.852918 seconds with 390625 blocks and 512 threads per block.
The running time for 10 billion rays does not seem to vary much when the number of rays looped through per thread ranges from order of tens to order of tens of thousands, with fixed threads per block. 

The following is generated with with 256 threads per block and 100 rays handled per thread in 0.805091 seconds:
<img src="https://github.com/mpcs51087/project-2-winter-2023-cripispuki/blob/main/Milestone2/1b.png" width=500>

The following is generated with with 512 threads per block and 50 rays handled per thread in 7.852918 seconds:
<img src="https://github.com/mpcs51087/project-2-winter-2023-cripispuki/blob/main/Milestone2/10b.png" width=500>

# Comparison with CPU performances
On my apple M1 processor, the serializing method took 6 minutes and 3.325 seconds. The increase in speed via GPU parallelization is about 400 fold.
