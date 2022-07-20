#!/bin/bash
module load openmpi/4.1.0-gcc-7.2.0

mpicxx -c -fopenmp --std=c++11 -c -o MPI.o MPI.cc
mpicxx -c -fopenmp --std=c++11 -c -o data.o data.cc
mpicxx -fopenmp -o mpi MPI.o data.o