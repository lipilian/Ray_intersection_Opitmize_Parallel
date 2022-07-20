nvcc -c -o CUDA.o CUDA.cu
nvcc -c -o data.o data.cc
nvcc -o CUDA CUDA.o data.o