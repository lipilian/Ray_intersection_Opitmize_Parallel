CXX=g++
CXXFLAGS=-c -fopenmp --std=c++11
LDFLAGS=-fopenmp



BRUTAL_OBJECTS = BrutalForce.o data.o
SIMD_OBJECTS = Simd.o data.o
OPENMP_OBJECTS = OpenMP.o data.o
MPI_OBJECTS = MPI.o data.o

brutalforce: $(BRUTAL_OBJECTS)
	$(CXX) $(LDFLAGS) -o brutalforce $(BRUTAL_OBJECTS)

simd: $(SIMD_OBJECTS)
	$(CXX) $(LDFLAGS) -o simd $(SIMD_OBJECTS)

openmp: $(OPENMP_OBJECTS)
	$(CXX) $(LDFLAGS) -o openmp $(OPENMP_OBJECTS)

run_brutalforce: brutalforce
	./brutalforce

run_simd: simd
	./simd

run_openmp: openmp
	./openmp

run_mpi: mpi
	./srun

clean: 
	rm -f *.o brutalforce simd openmp mpi *~