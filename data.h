#ifndef DATA_H
#define DATA_H

#include <iostream>
#include <string>
#include <fstream>


class VectorData {
    public:
        int length;
        std::string filename;
        float *u;

        VectorData(const std::string &filename);

        void getNumPoints();

        void readFile();

        ~VectorData();
};



#endif