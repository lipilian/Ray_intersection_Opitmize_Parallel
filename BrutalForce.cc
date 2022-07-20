#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <omp.h>
#include "data.h"

using namespace std;


int main(){

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
    printf("\n ----- Ray intersection finding program -----------\n ");
    printf("\033[1m%5s %15s %15s %15s\033[0m\n", "Set", "Time, ms", "GB/s", "GFLOP/s"); 
    fflush(stdout);
    
    string filename = "./data/" + to_string(1) + ".txt";
    VectorData a = VectorData(filename);
    //double *points = a.u;
    float *points = a.u;
    int length = a.length;
    double tTotal = 0.0;
 
    for (int m = 0; m < 1; m++){
        int RayCounter[NumZ * NumX * NumY] = {0};
        const double t0 = omp_get_wtime();
        for (int k = 0; k < length; k++){ 
            for (int p = 0; p < NumZ; p++){ 
                float scale = z[p] / points[5 * k + 4];
                float x = points[5 * k] + scale * points[5 * k + 2];
                float y = points[5 * k + 1] + scale * points[5 * k + 3];
                int Xindex = round((x - Xmin)/deltaX);
                int Yindex = round((y - Ymin)/deltaY);
                //#pragma omp atomic
                RayCounter[p * NumX * NumY + Xindex * NumY + Yindex]++;
            }
        }
        const double t1 = omp_get_wtime();
        cout << "Processing time: " << (t1 - t0)*1.0e3 << endl; 
        tTotal += (t1 - t0)*1.0e3;

        if(m == 0){ // try to store result to compare with other method
            std::ofstream myfile("BrutalForceOutput.txt");
            if(myfile.is_open()){
                for(int i = 0; i < NumZ * NumX * NumY; i++){
                    myfile << RayCounter[i] << " ";
                }
                myfile.close();
            } else {
                std::cout << "Can't open the output file" << std::endl;
            }
        }
    }
    cout << "Averaged time: " << tTotal/ 10 << endl;
    return 0;
}