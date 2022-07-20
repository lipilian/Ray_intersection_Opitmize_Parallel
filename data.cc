#include "data.h"

VectorData::VectorData(const std::string &filename) 
 : filename(filename) {
    VectorData::getNumPoints(); 
    VectorData::readFile();
}


VectorData::~VectorData(){
    //std::cout << "destructor is called" << std::endl;
    delete [] u;
}

void VectorData::getNumPoints(){
    std::ifstream in_file(filename);
    length = 0;
    if (in_file.is_open()){
        std::string line;
        while(getline(in_file, line)){
            length++;
        }
        //std::cout << "There are " << length << " rays" << std::endl;
    } 
    in_file.close();
}

void VectorData::readFile(){
    u = new float [length * 5];
    std::ifstream in_file(filename);
    int index = 0;
    std::string x, y, Vx, Vy, Vz;
    //std::getline(in_file, x , ',');
    //std::cout << x << std::endl;
    if(in_file.is_open()){
        while(index < length * 5){
            std::getline(in_file, x, ',');
            u[index] = std::stod(x);
            std::getline(in_file, y, ',');
            u[index + 1] = std::stod(y);
            std::getline(in_file, Vx, ',');
            u[index + 2] = std::stod(Vx);
            std::getline(in_file, Vy, ',');
            u[index + 3] = std::stod(Vy);
            std::getline(in_file, Vz, '\n');
            u[index + 4] = std::stod(Vz);
            index += 5;
        }
    } else {
        std::cout << "unable to open data file: " << filename << std::endl;
    }
    in_file.close();
}