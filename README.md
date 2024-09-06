# CUDA Simple Vs Grid-Stride Loop Methodology (https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/) and its variation on number of threads per block.

## We observe that the threads per block are more or less irrelevant in the case of Grid-stride loop methodology. 

## However, simple kernel achieves higher global memory access bandwidth peak in comparison to grid-stride loop methodology. 

GPU: Tesla V100 - SXM2 - 32GB

Compilation: nvcc -arch=sm_70 simpleVsGridStrideLoop.cu