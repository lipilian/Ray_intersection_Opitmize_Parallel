#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <chrono>
#include "data.h"

#define NumZ 401
#define NumX 40
#define NumY 120

__constant__ float Zlevel_Constant[NumZ];

__global__ void intersectGPU(float *d_input, int *d_output, int size, const float Xmin, const float Ymin, const float deltaX, const float deltaY){
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid < size){
        float scale, x, y;
        int Xindex, Yindex;
        for(int i = 0; i < NumZ; i++){
            scale = Zlevel_Constant[i] / d_input[gid * 5 + 4];
            x = d_input[gid * 5] + scale * d_input[gid * 5 + 2];
            y = d_input[gid * 5 + 1] + scale * d_input[gid * 5 + 3];
            Xindex = round((x - Xmin)/deltaX);
            Yindex = round((y - Ymin)/deltaY);
            atomicAdd(&(d_output[i * NumX * NumY + Xindex * NumY + Yindex]), 1);
        }
    }
}


int main(){
    using namespace std;
    using namespace std::chrono;

    const float Xmin = -1.0f;
    const float Xmax = 4.0f;
    const float Ymin = -1.0f;
    const float Ymax = 14.0f;
    const float Zmin = -40.0f;
    const float Zmax = 40.0f;
    const float deltaX = (Xmax - Xmin)/NumX;
    const float deltaY = (Ymax - Ymin)/NumY;
    const float deltaZ = 80.0f/NumZ;
    float z[NumZ] = {0};
    for(int i = 0; i < NumZ; i++){
        z[i] = Zmin + i * deltaZ;
    }
    printf("\n ----- Ray intersection finding program -----------\n ");
    printf("\033[1m%5s %15s %15s %15s\033[0m\n", "Set", "Time, ms", "GB/s", "GFLOP/s"); 
    fflush(stdout);

    string filename = "./data/" + to_string(1) + ".txt";
    VectorData a = VectorData(filename);
    float *points = a.u;
    int length = a.length;
    double tTotal = 0.0;

    auto start = std::chrono::high_resolution_clock::now();
    int size = length * 5;
    int byte_size = size * sizeof(float);

 
    // create device input
    float *d_input;
    cudaMalloc((void**)&d_input, byte_size);
    std::cout << "length = " << length << std::endl;
    cudaMemcpy(d_input,points, byte_size, cudaMemcpyHostToDevice);
    std::cout << "length = " << length << std::endl;
    
    // create device output and host output
    int *h_output, *d_output;
    int output_size = NumZ * NumX * NumY;
    int output_byte_size = output_size * sizeof(int);
    h_output = (int*)calloc(output_size,sizeof(int)); // malloc h_output and initialize to 0
    cudaMalloc((void**)&d_output, output_byte_size); // device memory allocate
    cudaMemcpy(d_output,h_output, output_byte_size, cudaMemcpyHostToDevice);

    //dimension of block and grid
    dim3 DimGrid(ceil(length/1024), 1, 1);
    dim3 DimBlock(1024,1,1);

    std::cout << "---------Fill constant memory data with Z level information----------" << std::endl;
    float Zlevel_host[NumZ];
    for(int i = 0; i < NumZ; i++){
        Zlevel_host[i] = Zmin + i * (Zmax - Zmin)/(NumZ - 1);
    }
    cudaMemcpyToSymbol(Zlevel_Constant, Zlevel_host, NumZ * sizeof(float));

    // run ray intersection in GPU
    intersectGPU<<<DimGrid, DimBlock>>>(d_input, d_output, length, Xmin, Ymin, deltaX, deltaY);
    cudaDeviceSynchronize();

    auto stop = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_output, d_output, output_byte_size, cudaMemcpyDeviceToHost);

    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Run time by using cuda is " << duration.count() << " us" << std::endl;
    cudaFree(d_input);

    // save data to file to compile with other method
    std::ofstream myfile("CudaOutput.txt");
    if(myfile.is_open()){
        for(int i = 0; i < output_size; i++){
            myfile << h_output[i] << " ";
        }
        myfile.close();
    } else {
        std::cout << "Can't open the output file" << std::endl;
    }

    free(h_output);
    cudaFree(d_output);
    // since points pointer will be automatic free by deconstrcutor 
    cudaDeviceReset();

    return 0;
}