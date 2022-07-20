#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <mpi.h>
#include <cstdio>
#include "data.h"

int main(int argc, char **argv){
    // MPI CODE
    using namespace std;
    int processors, me;
    MPI_Status status;
    int ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &processors);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &me);  

    const int NumZ = 401;
    const int NumX = 40;
    const int NumY = 120;
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
    
    string filename = "./data/" + to_string(1) + ".txt";
    VectorData a = VectorData(filename);
    double *points = a.u;
    int length = a.length;
    

    const double rowsPerProcess = double(length)/double(processors);
    const int myFirstRow = int(rowsPerProcess * me);
    const int myLastRow = int(rowsPerProcess * (me + 1));
    double tTotal = 0.0;
    for (int m = 0; m < 10; m++){
        
        int RayCounter_partial[NumZ * NumX * NumY] = {0};
        MPI_Barrier(MPI_COMM_WORLD);
        const double t0 = omp_get_wtime();
        #pragma omp parallel for
        for (int k = myFirstRow; k < myLastRow; k++){ 
            for (int p = 0; p < NumZ; p++){ 
                float scale = z[p] / points[5 * k + 4];
                float x = points[5 * k] + scale * points[5 * k + 2];
                float y = points[5 * k + 1] + scale * points[5 * k + 3];
                int Xindex = round((x - Xmin)/deltaX);
                int Yindex = round((y - Ymin)/deltaY);
                #pragma omp atomic
                RayCounter_partial[p * NumX * NumY + Xindex * NumY + Yindex]++;
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &RayCounter_partial, NumZ * NumX * NumY, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        const double t1 = omp_get_wtime();
        if(me == 0){
            cout << "Processing time: " << (t1 - t0)*1.0e3 << endl; 
            tTotal += (t1 - t0)*1.0e3;
        }
        
    }
    

    if(me == 0){
        printf("I am %d of %d processor\n", me, processors); //TODO print the running time
        cout << "Averaged time: " << tTotal/10 << endl;//tTotal/ 10 << endl;
    }

    //TODO: free allocated memory
    ierr = MPI_Finalize(); // end MPI

    return 0;
}